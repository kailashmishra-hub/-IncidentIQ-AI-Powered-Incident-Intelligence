# RAG with Multiple Repositories

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** pipeline that loads data from **single/multiple JSON/Txt files stored in a folder** and enables semantic search over the combined dataset.

The goal of this repository is to show how structured data from multiple sources can be transformed into embeddings and queried using natural language.

RAG enhances Large Language Models (LLMs) by retrieving relevant external data before generating a response, improving accuracy and reducing hallucinations.

---

# Project Overview

This project implements a simple RAG workflow with examples to depict Incidents_List scenarios , Retrieves information from multiple JSONs.

## Similar Incident Finder (Streamlit)

`RAG_05_SimilarIncidentsFinder.py` provides a simple web interface that searches
only user-uploaded CSV or Excel incident data. The files in `Incidents_List` are
not indexed or searched by the Streamlit application.

Upload one or more CSV/Excel files from the sidebar to use your own incident data.
Column names become field labels, and every non-empty spreadsheet row is converted
to one complete document and one `text-embedding-3-small` vector. Rows are never
split, overlapped, or combined. Results below the selected minimum-similarity score
are excluded.

The sidebar provides two date scopes: **Last 6 months** (the default) and
**Whole repository**. The six-month window ends on the entered current-issue
creation date. Common date columns such as `Created Date`, `Created At`,
`Incident Date`, `Issue Date`, `Opened Date`, and `Date` are detected
automatically; undated rows are excluded from the six-month search.

After a search returns matches, an **Incident Copilot** panel appears beside the
result. It retains the current investigated issue but can answer from every row in
the uploaded repository, including questions about other incidents, assignees,
months, frequency, and trends. Each row is counted as one occurrence, and the
conversation resets whenever the upload or search filters change.

```powershell
pip install -r requirements.txt
streamlit run RAG_05_SimilarIncidentsFinder.py
```

Enter the OpenAI API key in the app sidebar. Alternatively, set the
`OPENAI_API_KEY` environment variable or add it to `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-key-here"
```
