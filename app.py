"""
app.py
Interactive Streamlit Web Application for AI-Based Legal Case Law Recommendation
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from baseline_tfidf import TfidfCaseRecommender
from siamese_model import SiameseCaseRecommender

# Page Configuration
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;800&family=Inter:wght@400;500;600&display=swap');

    /* Typography */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
    }
    p, span, div, input, button, .stMarkdown {
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean headers */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        color: var(--primary-color);
        margin-bottom: 2rem;
        font-family: 'Playfair Display', serif !important;
    }

    /* Metric/Toggle Buttons (Cards) */
    .metric-card {
        background-color: var(--secondary-background-color);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        border: 1px solid rgba(128, 128, 128, 0.2);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--primary-color);
        margin: 0;
    }
    .metric-label {
        color: var(--text-color);
        opacity: 0.7;
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Cache Models
@st.cache_resource
def load_recommenders():
    tfidf_engine = TfidfCaseRecommender()
    siamese_engine = SiameseCaseRecommender()
    return tfidf_engine, siamese_engine

tfidf_engine, siamese_engine = load_recommenders()
corpus_df = tfidf_engine.df

# Sample benchmark scenarios
SAMPLE_QUERIES = {
    "Select a scenario...": "",
    "Airport Narcotics Concealment (Heroin in luggage)": "Accused was intercepted by customs at Bandaranaike International Airport carrying heroin concealed in a false bottom of luggage and shoes without mens rea or conscious knowledge.",
    "Circumstantial Evidence Murder Conviction": "The accused was convicted of murder under Section 296 based solely on circumstantial evidence, recovery of weapon under section 27, and having been seen last with the deceased.",
    "Public Servant Bribery Trap": "A public surveyor demanded a cash gratification of Rs. 450 to recommend a water permit for state land and was arrested in a marked money trap laid by bribery officers.",
    "Bail Remand Without Recording Reasons": "Magistrate made an order remanding the suspect in custody without stating grounds or recording reasons as required under Section 75.",
    "Unsworn Statement from the Dock": "The accused chose to make an unsworn statement from the dock rather than giving evidence under affirmation from the witness box, leading to trial judge misdirection.",
    "Weapon Training under Prevention of Terrorism Act": "Accused underwent weapons training as a member of an unlawful organization charged under Section 2 of Prevention of Terrorism Act PTA based on confession."
}

# Sidebar Controls
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <div style="font-size: 3rem;">⚖️</div>
            <h2 style="font-family: 'Playfair Display', serif; margin-top: 5px;">Legal AI Assistant</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### How can I help?")

    all_topics = ["All Areas of Law"] + sorted(list(corpus_df["Legal Topic / Law"].unique()))
    selected_topic = st.selectbox("Filter by Legal Domain", all_topics)

    top_k = st.slider("How many cases should I find?", min_value=3, max_value=10, value=5)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Quick Examples")
    scenario_pick = st.selectbox("Try a sample scenario:", list(SAMPLE_QUERIES.keys()))

# Hero Section
st.markdown('<div class="main-header">Hello! I\'m your Legal AI Assistant.</div>', unsafe_allow_html=True)

# Top Metrics Row (Restored Toggle Buttons / Custom Cards)
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f'''
        <div class="metric-card">
            <p class="metric-value">{len(corpus_df)}</p>
            <p class="metric-label">Cases Read</p>
        </div>
    ''', unsafe_allow_html=True)
with m2:
    st.markdown(f'''
        <div class="metric-card">
            <p class="metric-value">{corpus_df["Legal Topic / Law"].nunique()}</p>
            <p class="metric-label">Legal Areas</p>
        </div>
    ''', unsafe_allow_html=True)
with m3:
    st.markdown(f'''
        <div class="metric-card">
            <p class="metric-value">1974-2015</p>
            <p class="metric-label">Time Period</p>
        </div>
    ''', unsafe_allow_html=True)

st.write("")
st.divider()

# Tabbed Layout
tab_search, tab_analytics = st.tabs([
    "🔍 Find Precedents",
    "📚 Browse the Database"
])

def render_case_result(res: dict) -> None:
    \"\"\"
    Renders a single case result in the Streamlit UI using native container components.
    
    Args:
        res (dict): A dictionary containing the case metadata, similarity score, and facts.
    \"\"\"
    # Using Native Streamlit components for a flawless, responsive layout
    with st.container(border=True):
        st.markdown(f"### {res['rank']}. {res['case_name']}")
        
        # Tags / Metadata row
        col1, col2, col3, col4 = st.columns(4)
        col1.caption(f"**Match:** {res['similarity_pct']}")
        col2.caption(f"**Year:** {res['year']}")
        col3.caption(f"**Domain:** {res['legal_topic']}")
        col4.caption(f"**Court:** {res['court_no']}")
        
        st.markdown(f"> **What happened:** {res['case_facts'][:450]}...")
        
        with st.expander("Read my full analysis of this judgment"):
            st.markdown(f"**Full Narrative Facts:**\n\n{res['case_facts']}")
            st.divider()
            st.markdown(f"**The Court's Decision (Holding):**\n\n{res['holding']}")

def hybrid_search(query, top_k=5, topic_filter=None):
    """Combines Siamese neural semantic scores (40%) and TF-IDF lexical scores (60%)."""
    tfidf_res = {r["dataset_id"]: r for r in tfidf_engine.search(query, top_k=len(corpus_df), category_filter=topic_filter)}
    siamese_res = {r["dataset_id"]: r for r in siamese_engine.search(query, top_k=len(corpus_df), category_filter=topic_filter)}

    all_ids = set(tfidf_res.keys()).union(set(siamese_res.keys()))
    combined = []
    for d_id in all_ids:
        t_score = tfidf_res.get(d_id, {}).get("similarity_score", 0.0)
        s_score = siamese_res.get(d_id, {}).get("similarity_score", 0.0)
        
        final_score = 0.60 * t_score + 0.40 * s_score
        
        base_record = tfidf_res.get(d_id) or siamese_res.get(d_id)
        rec = dict(base_record)
        rec["similarity_score"] = round(final_score, 4)
        rec["similarity_pct"] = f"{round(final_score * 100, 1)}%"
        combined.append(rec)

    combined.sort(key=lambda x: x["similarity_score"], reverse=True)
    for idx, r in enumerate(combined[:top_k], start=1):
        r["rank"] = idx
    return combined[:top_k]

# TAB 1: Search & Recommendation
with tab_search:
    st.write("")
    st.subheader("Tell me about your case")
    default_text = SAMPLE_QUERIES.get(scenario_pick, "") if scenario_pick != "Select a scenario..." else ""
    
    user_query = st.text_area(
        "Describe what happened (the facts, the crime, or the legal issue you want me to look into):",
        value=default_text,
        height=150,
        placeholder="Example: The accused was caught with narcotics at the airport, but claims they didn't know it was in their bag...",
        label_visibility="collapsed"
    )

    search_clicked = st.button("Find Similar Cases", use_container_width=True, type="primary")

    if (search_clicked or default_text) and user_query.strip():
        cat_filter = None if selected_topic == "All Areas of Law" else selected_topic
        results = hybrid_search(user_query, top_k=top_k, topic_filter=cat_filter)

        st.write("")
        st.subheader("Here are the most relevant cases I found:")
        st.write("")

        if results:
            for res in results:
                render_case_result(res)
        else:
            st.warning("I couldn't find any historical cases matching that description. Try writing it differently or clearing the legal domain filter.")

# TAB 2: Corpus Database Explorer
with tab_analytics:
    st.write("")
    st.subheader("Browse the Database")
    filter_txt = st.text_input("Type a keyword or case name to search the entire database:", label_visibility="collapsed", placeholder="Search by keyword, case name, or domain...")
    
    filtered_df = corpus_df
    if filter_txt:
        filtered_df = corpus_df[
            corpus_df["Case Name"].str.contains(filter_txt, case=False, na=False) |
            corpus_df["Case Facts / Issue"].str.contains(filter_txt, case=False, na=False) |
            corpus_df["Legal Topic / Law"].str.contains(filter_txt, case=False, na=False)
        ]
    
    st.dataframe(
        filtered_df[["Dataset ID", "Case Name", "Year", "Legal Topic / Law", "Court Case No.", "Reporter"]],
        use_container_width=True,
        hide_index=True
    )
