import streamlit as st

st.set_page_config(page_title="Pipeline Documentation", layout="wide")

st.title("Pipeline Documentation")

documents = {
    "General Documentation": "README.md",
    "manage.py": "docs/manage.md",
    "Data Ingestion": "docs/data_ingestion.md",
    "Data Cleaning": "docs/data_cleaning.md",
    "Feature Engineering": "docs/feature_engineering.md",
    "Model Training": "docs/model_training.md",
    "Backtesting": "docs/trading_logic.md",
}

col1, col2 = st.columns([1, 3])

with col1:
    selected_title = st.radio("Documents", list(documents.keys()))

with col2:
    with open(documents[selected_title], "r", encoding="utf-8") as markdown_file:
        st.markdown(markdown_file.read())