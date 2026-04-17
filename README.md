# SI507 Final Project

This repository contains the course-only version of **Stock and News Network Explorer**.

It is organized for two use cases:

- course submission
- Streamlit Cloud deployment

## Main features

- real market data from local snapshots
- graph-based analysis with `stock`, `sector`, and `topic` nodes
- CLI workflow
- unified Streamlit site with:
  - analysis dashboard
  - interactive graph explorer
    - explorer presets
    - switchable style profiles:
      - AWS + Bloom
      - AWS + Sigma
      - Kumu + Bloom
    - focus-node neighborhood view
    - node inspector and structure panels
- LLM-based news impact assessment
- tests and study documents

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the CLI:

```bash
python main.py
```

Run the web app:

```bash
streamlit run query_app.py
```

## Streamlit Cloud

Use these values:

- Repository: your GitHub repo
- Branch: `main`
- Main file path: `query_app.py`

Optional secret for the `New News Analysis` feature:

```toml
OPENAI_API_KEY = "your_openai_api_key"
```

## Study path

Read the files in `study_docs/` from `01` to `10`.

## Note

This project is not a stock prediction system and not an investment recommendation tool.  
The graph is the primary structure for exploring relationships among stocks, sectors, and news topics.
