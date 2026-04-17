# Streamlit Deployment Notes

This folder is the GitHub upload version for the course query app.

## Recommended Streamlit deployment target

Deploy the query app first:

- Main file path: `query_app.py`

This is the cleaner course-facing app.

## Streamlit Community Cloud form

Fill the deployment form like this:

- Repository: `<your-github-username>/<your-repo-name>`
- Branch: `main` or `master`
- Main file path: `query_app.py`

## Optional secret

If you want the **New News Analysis** tab to call OpenAI, add this in Advanced settings:

```toml
OPENAI_API_KEY = "your_openai_api_key"
```

If you do not add this secret, the main query app still works. Only the new-news LLM feature will be unavailable.

## What is included

- Query Streamlit app
- Interactive graph app
- Core graph-analysis modules
- Local data snapshot:
  - 50 stock price tables
  - 50 sector metadata files
  - merged news payload
  - one LLM impact run for the News Impact tab

## What is intentionally excluded

- tests
- study docs
- slides
- temporary files
- optional semantic retrieval modules

