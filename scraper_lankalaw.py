import urllib.request
import ssl
import re
import io
import os
import csv
import time
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pypdf

# Setup SSL context for HTTPS requests
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
}

CSV_FILE = "sri_lanka_criminal_cases_corpus_v2.csv"
CSV_COLUMNS = [
    "Dataset ID",
    "Case Name",
    "Court Case No.",
    "Year",
    "Reporter",
    "Legal Topic / Law",
    "Case Facts / Issue",
    "Judgment / Holding",
    "Source URL"
]

CRIMINAL_KEYWORDS = [
    'penal code', 'criminal law', 'criminal procedure', 'code of criminal procedure', 
    'bribery', 'theft', 'murder', 'culpable homicide', 'assault', 'grievous hurt', 
    'bail', 'accused', 'sentence', 'conviction', 'burglary', 'robbery', 'stolen property', 
    'forgery', 'cheating', 'emergency regulations', 'insurrection', 'rape', 'kidnapping', 
    'unlawful assembly', 'magistrate', 'poisons, opium', 'narcotics', 'corruption', 
    'offensive weapons', 'extortion'
]

csv_lock = threading.Lock()
collected_count = 0
existing_ids = set()
existing_urls = set()

def get_html(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
                return resp.read().decode('utf-8', errors='ignore')
        except Exception:
            time.sleep(1)
    return ""

def get_all_collection_urls():
    """Retrieve all SLR and NLR volume URLs from lankalaw.net with SLR prioritized"""
    print("[*] Fetching volume list from lankalaw.net homepage...", flush=True)
    home_html = get_html("https://lankalaw.net")
    
    slr_links = set(re.findall(r'href=["\'](https?://(?:www\.)?lankalaw\.net/sri-lanka-law-reports-\d+/?)[^"\'\s]*["\']', home_html, re.I))
    nlr_links = set(re.findall(r'href=["\'](https?://(?:www\.)?lankalaw\.net/new-law-report[s]?-volume-\d+/?)[^"\'\s]*["\']', home_html, re.I))
    
    sorted_slr = sorted(list(slr_links), key=lambda x: int(re.search(r'reports-(\d+)', x).group(1)) if re.search(r'reports-(\d+)', x) else 0, reverse=True)
    sorted_nlr = sorted(list(nlr_links), key=lambda x: int(re.search(r'volume-(\d+)', x).group(1)) if re.search(r'volume-(\d+)', x) else 0, reverse=True)
    
    all_volumes = sorted_slr + sorted_nlr
    print(f"[*] Found {len(all_volumes)} total volumes ({len(sorted_slr)} SLR prioritized, {len(sorted_nlr)} NLR).", flush=True)
    return all_volumes

def parse_pdf_content(pdf_url, volume_label):
    """Download PDF, extract text, and extract structured legal fields supporting both NLR and SLR."""
    try:
        req = urllib.request.Request(pdf_url, headers=HEADERS)
        with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
            pdf_bytes = resp.read()
    except Exception:
        return None

    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        full_text = ""
        for page in reader.pages[:8]:
            t = page.extract_text() or ""
            full_text += t + "\n"
    except Exception:
        return None

    if len(full_text.strip()) < 100:
        return None

    cleaned_text = re.sub(r'[ \t]+', ' ', full_text)
    lower_text = cleaned_text.lower()

    # Determine if it's a criminal case
    matches = [k for k in CRIMINAL_KEYWORDS if k in lower_text]
    if len(matches) < 2 and 'penal code' not in lower_text and 'criminal' not in lower_text:
        return None

    filename = pdf_url.split('/')[-1].replace('.pdf', '')
    
    # 1. Extract Dataset ID
    slr_file_match = re.search(r'v(\d+)-sri-lr-(\d+)', filename, re.I)
    nlr_id_match = re.search(r'/(\d+)-(?:NLR-)?(?:NLR-)?V-(\d+)', pdf_url, re.I)

    if slr_file_match:
        v, num = slr_file_match.groups()
        dataset_id = f"{volume_label}-V{v}-{num.zfill(3)}"
    elif nlr_id_match:
        case_num, vol = nlr_id_match.groups()
        dataset_id = f"NLR-{vol}-{case_num.zfill(3)}"
    else:
        clean_fn = re.sub(r'[^A-Za-z0-9\-]', '', filename)[:14]
        dataset_id = f"{volume_label}-{clean_fn}"

    # 2. Extract Case Name
    case_name = "Unknown"
    lines = [l.strip() for l in cleaned_text.split('\n') if l.strip()]
    for l in lines[:6]:
        if re.search(r'\b[vV][sS]\.?\b', l):
            clean_n = re.split(r'\b\d{4}/|\b20\d\d\b|\b19\d\d\b', l)[0].strip()
            if len(clean_n) > 5:
                case_name = clean_n
                break
        elif 'Appellant' in l and 'Respondent' in l:
            case_name = l
            break

    if case_name == "Unknown":
        name_match = re.search(r'([A-Z0-9\.\s\-]{3,60},?\s*(?:Appellant|Applicant|Petitioner)s?,?\s*and\s*[A-Z0-9\.\s\-\(\)]{3,60},?\s*(?:Respondent)s?)', cleaned_text)
        if name_match:
            case_name = re.sub(r'^(?:[A-Z\.\s]+(?:J\.|S\.P\.J\.|C\.J\.)\s+)?', '', name_match.group(1).strip().replace('\n', ' '))

    # Skip index or table of contents pages
    if case_name == "Unknown" or len(case_name) < 4:
        return None

    # 3. Extract Year
    year = ""
    year_match = re.search(r'\b(19\d\d|20\d\d)\b', cleaned_text[:400])
    if year_match:
        year = year_match.group(1)
    elif 'SLR-' in volume_label:
        y_match = re.search(r'\d{4}', volume_label)
        if y_match:
            year = y_match.group(0)

    # 4. Extract Court Case No.
    court_no = ""
    court_match = re.search(r'((?:S\.?C\.?|C\.?A\.?|H\.?C\.?|M\.?C\.?|D\.?C\.?)\s*(?:Appeals?\s*)?(?:No\.?\s*)?[A-Z0-9\/\-]{3,35})', cleaned_text)
    if court_match:
        court_no = court_match.group(1).strip()

    # 5. Extract Legal Topic / Law
    legal_topic = "Criminal Law"
    for kw in ['Penal Code', 'Criminal Procedure', 'Bribery', 'Bail', 'Evidence', 'Poisons', 'Narcotics', 'Emergency Regulations', 'Administration of Justice Law']:
        if kw.lower() in lower_text:
            legal_topic = kw
            break

    # 6. Extract Holding
    judgment = ""
    held_match = re.search(r'Held\s*[:\-]\s*(.*?)(?:\n[A-Z\s]{4,}\s*J\.|\nAPPEAL from|\nAPPLICATION for|Cur\. adv\. vult\.)', cleaned_text, re.DOTALL | re.I)
    if held_match:
        judgment = re.sub(r'\s+', ' ', held_match.group(1)).strip()[:1500]

    # 7. Extract Narrative Facts
    facts_chunk = ""
    if court_no and court_no in cleaned_text:
        parts = cleaned_text.split(court_no, 1)
        if len(parts) > 1:
            facts_chunk = parts[1]
    elif year and year in cleaned_text:
        parts = cleaned_text.split(year, 1)
        if len(parts) > 1:
            facts_chunk = parts[1]
    else:
        facts_chunk = cleaned_text[:2500]

    if "Held" in facts_chunk:
        case_facts = facts_chunk.split("Held")[0].strip()
    else:
        case_facts = facts_chunk[:1200].strip()

    case_facts = re.sub(r'\s+', ' ', case_facts).strip()
    if len(case_facts) < 80:
        case_facts = cleaned_text[:800].replace('\n', ' ')

    return {
        "Dataset ID": dataset_id,
        "Case Name": case_name[:120],
        "Court Case No.": court_no[:60],
        "Year": year,
        "Reporter": f"Volume {volume_label}",
        "Legal Topic / Law": legal_topic,
        "Case Facts / Issue": case_facts[:2500],
        "Judgment / Holding": judgment[:1500] if judgment else "Conviction/Sentence affirmed or set aside based on legal points.",
        "Source URL": pdf_url
    }

def process_pdf_job(pdf_url, volume_label, target_cases):
    global collected_count, existing_ids, existing_urls
    try:
        with csv_lock:
            if collected_count >= target_cases or pdf_url in existing_urls:
                return None

        record = parse_pdf_content(pdf_url, volume_label)
        if not record or record["Case Name"] == "Unknown":
            return None

        with csv_lock:
            if collected_count >= target_cases or record["Dataset ID"] in existing_ids:
                return None

            with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerow(record)
                f.flush()

            existing_ids.add(record["Dataset ID"])
            existing_urls.add(pdf_url)
            collected_count += 1
            curr = collected_count

        print(f" [{curr}/{target_cases}] Added: {record['Dataset ID']} | {record['Case Name'][:32]} | {record['Legal Topic / Law']}", flush=True)
        return record
    except Exception as e:
        print(f"[!] Error on {pdf_url}: {e}", flush=True)
        return None

def run_scraper(target_cases=1000):
    global collected_count, existing_ids, existing_urls
    print(f"=======================================================", flush=True)
    print(f"[*] Starting Precision Sri Lankan Legal Corpus Scraper", flush=True)
    print(f"[*] Target Goal: {target_cases} Real Criminal Cases", flush=True)
    print(f"[*] Target File: {CSV_FILE}", flush=True)
    print(f"=======================================================\n", flush=True)

    completed_vol_labels = set()
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            reporter_counts = {}
            for r in reader:
                d_id = r.get("Dataset ID", "")
                url = r.get("Source URL", "")
                rep = r.get("Reporter", "")
                if d_id:
                    existing_ids.add(d_id)
                if url:
                    existing_urls.add(url)
                if rep:
                    reporter_counts[rep] = reporter_counts.get(rep, 0) + 1
                    
        for rep, cnt in reporter_counts.items():
            if cnt >= 20:
                completed_vol_labels.add(rep.replace("Volume ", "").strip())

        collected_count = len(existing_ids)
        print(f"[*] Resuming: {collected_count} cases already collected ({len(existing_urls)} URLs registered).", flush=True)
        print(f"[*] Completed volume labels skipped: {len(completed_vol_labels)} volumes", flush=True)

    # Write header if new
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()

    if collected_count >= target_cases:
        print(f"[+] Target of {target_cases} cases already met! Total: {collected_count}", flush=True)
        return

    volumes = get_all_collection_urls()

    for vol_url in volumes:
        if collected_count >= target_cases:
            break

        vol_label = "Vol"
        if 'reports-' in vol_url:
            year = re.search(r'reports-(\d+)', vol_url).group(1)
            vol_label = f"SLR-{year}"
        elif 'volume-' in vol_url:
            num = re.search(r'volume-(\d+)', vol_url).group(1)
            vol_label = f"NLR-{num}"

        if vol_label in completed_vol_labels:
            continue

        print(f"\n---> Scanning {vol_label}: {vol_url}", flush=True)
        html = get_html(vol_url)
        if not html:
            continue

        pdf_links = list(set(re.findall(r'href=["\'](https?://[^"\']+\.pdf)["\']', html)))
        new_pdf_links = [l for l in pdf_links if l not in existing_urls]
        print(f"     Found {len(pdf_links)} total PDFs ({len(new_pdf_links)} uncollected).", flush=True)

        if not new_pdf_links:
            continue

        # Concurrent processing with 6 worker threads
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(process_pdf_job, link, vol_label, target_cases) for link in new_pdf_links]
            for f in as_completed(futures):
                if collected_count >= target_cases:
                    break

    print(f"\n=======================================================", flush=True)
    print(f"[+] MASS SCRAPING FINISHED!", flush=True)
    print(f"[+] Total Sri Lankan Criminal Cases Collected: {collected_count}")
    print(f"[+] Corpus Saved At: {CSV_FILE}")
    print(f"=======================================================\n", flush=True)

if __name__ == "__main__":
    target = 1000
    if len(sys.argv) > 1:
        target = int(sys.argv[1])
    run_scraper(target)
