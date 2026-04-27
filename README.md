# SI507 Final Project

This repository contains the course-only version of **Stock and News Network Explorer**.

It is organized for two use cases:

- course submission
- Streamlit Cloud deployment

## Recommended grading path

If you are reviewing the project, start here:

1. `SUBMISSION_SUMMARY.md`
2. `DESIGN_DECISIONS_AND_COMPARISONS.md`
3. this `README.md`

Those files explain the project goal, graph design, interaction modes, design
decisions, comparisons, and how to run the system.

## Main features

- real market data from local snapshots
- graph-based analysis with `stock`, `sector`, and `topic` nodes
- CLI workflow
- unified Streamlit site with analysis dashboard and interactive graph explorer
- graph explorer presets, style profiles, search, shortest-path highlight, and inspector panels
- LLM-based news impact assessment
- explicit design-decision and comparison report
- pytest test suite

## Local data snapshot

The repository already includes enough cached data to run the main project:

- 50 stock price tables
- 50 sector metadata files
- one merged news JSON file with 41,913 deduplicated articles in `data/raw/news/merged_seed_news.json`
- one cached LLM impact run for the `News Impact` tab

## Quick start

From the project folder, install dependencies:

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

Verify the project:

```bash
python -m compileall -q .
pytest -q
```

Expected test result:

```text
72 passed
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

## Note

This project is not a stock prediction system and not an investment recommendation tool.  
The graph is the primary structure for exploring relationships among stocks, sectors, and news topics.
