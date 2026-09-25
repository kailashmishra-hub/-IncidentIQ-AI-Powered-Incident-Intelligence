"""Streamlit app for finding similar historical incidents with RAG."""

import os
import re
from datetime import date
from io import BytesIO

import pandas as pd
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

PROMPT = ChatPromptTemplate.from_template(
    """
You help an IT support engineer identify similar historical incidents.
Use only the incident rows in the supplied context. Explain only the supplied
matches. Do not mention or provide a "closest available" incident. Do not invent
incident IDs or details.

Current incident:
{question}

Historical incident reports:
{context}

For each selected match, return:
1. Similar Incident ID
2. Short Description
3. Detailed Description
4. Root Cause
5. Impact
6. Severity
7. Reported By
8. Created Date
9. Resolved Time
10. Status
11. Why it is similar

Preserve the values from the incident row. Write "Not available" for any field
that is absent or empty. Present incidents in the supplied order, which is newest
Created Date first.
"""
)

KNOWN_INCIDENT_COLUMNS = {
    "incidentid", "incidentnumber", "shortdescription", "detaileddescription",
    "rootcause", "impact", "severity", "reportedby", "createddate",
    "resolvedtime", "status",
}


def normalize_column_name(value) -> str:
    """Normalize spreadsheet headers for reliable matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def promote_incident_header(raw_table: pd.DataFrame) -> pd.DataFrame:
    """Detect and promote the row containing the real incident column headers."""
    if raw_table.empty:
        return raw_table

    best_row_index = 0
    best_score = -1
    for row_index in range(min(20, len(raw_table))):
        normalized_cells = {
            normalize_column_name(value)
            for value in raw_table.iloc[row_index].tolist()
            if pd.notna(value)
        }
        score = len(normalized_cells & KNOWN_INCIDENT_COLUMNS)
        if score > best_score:
            best_score = score
            best_row_index = row_index

    if best_score < 2:
        raise ValueError(
            "No valid incident header row was found. The workbook appears to be "
            "a dashboard or headerless export. Use a flat table with columns such "
            "as Incident_ID, Short_Description, Detailed_Description, Root_Cause, "
            "Impact, Severity, Reported_By, Created Date, Resolved_Time, and Status."
        )

    headers = [
        str(value).strip() if pd.notna(value) else f"Unnamed_{column_index}"
        for column_index, value in enumerate(raw_table.iloc[best_row_index].tolist())
    ]
    table = raw_table.iloc[best_row_index + 1 :].copy()
    table.columns = headers
    return table.dropna(how="all").reset_index(drop=True)


def row_to_text(row: pd.Series) -> str:
    """Turn one spreadsheet row into one labelled embedding document."""
    fields = []
    for column, value in row.items():
        if pd.notna(value) and str(value).strip():
            fields.append(f"{column}: {str(value).strip()}")
    return "\n".join(fields)


def approved_fields_from_row(row: pd.Series) -> str:
    """Extract result-card fields directly from the original spreadsheet row."""
    normalized_row = {
        re.sub(r"[^a-z0-9]", "", str(column).lower()): value
        for column, value in row.items()
    }
    field_definitions = [
        ("Incident ID", ("incidentid", "incidentnumber", "ticketid", "number", "id")),
        ("Short Description", ("shortdescription", "summary", "title")),
        ("Detailed Description", ("detaileddescription", "description", "details")),
        ("Root Cause", ("rootcause",)),
        ("Impact", ("impact",)),
        ("Severity", ("severity", "priority")),
        ("Reported By", ("reportedby", "reporter", "openedby")),
        ("Created Date", ("createddate", "createdat", "creationdate", "openeddate")),
        (
            "Resolved Time",
            ("resolvedtime", "resolveddate", "resolvedat", "resolutiondate", "closedat"),
        ),
        ("Status", ("status", "state")),
    ]
    lines = []
    for display_name, aliases in field_definitions:
        value = ""
        # Prefer exact header aliases before suffix matching so a generic alias
        # such as "description" cannot override "Detailed_Description".
        for alias in aliases:
            column_value = normalized_row.get(alias)
            if column_value is not None and pd.notna(column_value):
                if str(column_value).strip():
                    value = str(column_value).strip()
                    break
        if not value:
            for alias in aliases:
                for column_name, column_value in normalized_row.items():
                    if column_name.endswith(alias) and pd.notna(column_value):
                        if str(column_value).strip():
                            value = str(column_value).strip()
                            break
                if value:
                    break
        lines.append(f"{display_name}: {value or 'Not available'}")
    return "\n\n".join(lines)


def result_display_text(content: str) -> str:
    """Show only the approved incident fields in retrieved result cards."""
    parsed_fields = {}
    field_pattern = re.compile(
        r"^([^:\n]+):\s*(.*?)(?=^[^:\n]+:\s|\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    for label, value in field_pattern.findall(content):
        normalized = re.sub(r"[^a-z0-9]", "", label.lower())
        parsed_fields[normalized] = value.strip()

    display_fields = [
        ("Incident ID", ("incidentid", "id")),
        ("Short Description", ("shortdescription", "title")),
        ("Detailed Description", ("detaileddescription", "description")),
        ("Root Cause", ("rootcause",)),
        ("Impact", ("impact",)),
        ("Severity", ("severity", "priority")),
        ("Reported By", ("reportedby", "reporter")),
        ("Created Date", ("createddate", "createdat", "creationdate")),
        (
            "Resolved Time",
            ("resolvedtime", "resolveddate", "resolvedat", "resolutiondate"),
        ),
        ("Status", ("status",)),
    ]
    lines = []
    for display_name, aliases in display_fields:
        value = next(
            (parsed_fields[alias] for alias in aliases if parsed_fields.get(alias)),
            "Not available",
        )
        lines.append(f"{display_name}: {value}")
    return "\n\n".join(lines)


def safe_original_row_text(content: str) -> str:
    """Return original row fields while removing fields excluded from results."""
    kept_lines = []
    skip_continuation = False
    for line in content.splitlines():
        if ":" in line:
            label = line.split(":", 1)[0]
            normalized = normalize_column_name(label)
            skip_continuation = normalized in {"relatedincident", "relatedincidents"}
            if skip_continuation:
                continue
        if not skip_continuation:
            kept_lines.append(line)
    return "\n".join(kept_lines).strip()


def has_any_mapped_value(display_text: str) -> bool:
    """Check whether at least one approved field was mapped successfully."""
    values = [
        line.split(":", 1)[1].strip()
        for line in display_text.splitlines()
        if ":" in line
    ]
    return any(value and value != "Not available" for value in values)


def display_text_to_record(display_text: str) -> dict[str, str]:
    """Convert a result card into a row suitable for CSV export."""
    record = {}
    current_field = None
    for line in display_text.splitlines():
        if ":" in line:
            field, value = line.split(":", 1)
            current_field = field.strip()
            record[current_field] = value.strip()
        elif current_field and line.strip():
            record[current_field] = f"{record[current_field]}\n{line.strip()}".strip()
    return record


def extract_created_date(row: pd.Series) -> str:
    """Extract an ISO date from a commonly named creation-date column."""
    date_columns = {
        "date", "timestamp", "created", "createddate", "createdat",
        "creationdate", "issuedate", "incidentdate",
        "issuecreateddate", "issuecreatedat", "openeddate", "openedat",
        "reporteddate", "reportedat", "datecreated",
    }
    for column, value in row.items():
        normalized = re.sub(r"[^a-z0-9]", "", str(column).lower())
        if normalized in date_columns and pd.notna(value):
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.notna(parsed):
                return parsed.date().isoformat()
    return ""


def extract_resolved_date(row: pd.Series) -> str:
    """Extract an ISO date/time from a commonly named resolution field."""
    date_columns = {
        "resolveddate", "resolvedtime", "resolveddatetime", "resolvedat",
        "resolutiondate", "resolutiontime", "resolutionat",
        "closeddate", "closedat", "completeddate", "completedat",
        "dateclosed", "dateresolved",
    }
    for column, value in row.items():
        normalized = re.sub(r"[^a-z0-9]", "", str(column).lower())
        if normalized in date_columns and pd.notna(value):
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.notna(parsed):
                return parsed.isoformat(sep=" ")
    return ""


def read_uploaded_incidents(uploaded_files) -> list[Document]:
    """Read CSV/XLSX files while preserving one row = one incident."""
    documents = []
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        if uploaded_file.name.lower().endswith(".csv"):
            raw_tables = {"CSV": pd.read_csv(BytesIO(file_bytes), header=None)}
        else:
            raw_tables = pd.read_excel(
                BytesIO(file_bytes), sheet_name=None, header=None
            )
        tables = {}
        for sheet_name, raw_table in raw_tables.items():
            try:
                tables[sheet_name] = promote_incident_header(raw_table)
            except ValueError:
                # Workbooks commonly include dashboard/summary sheets. They are
                # intentionally ignored because they do not contain incident rows.
                continue
        if not tables:
            raise ValueError(
                f"No incident table was found in {uploaded_file.name}. Expected a "
                "sheet with headers such as Incident ID, Title, Severity, Status, "
                "Created Date, and Detailed Description."
            )

        for sheet_name, table in tables.items():
            for row_index, row in table.iterrows():
                content = row_to_text(row)
                if content:
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": uploaded_file.name,
                                "sheet": str(sheet_name),
                                "row": int(row_index) + 2,
                                "created_date": extract_created_date(row),
                                "resolved_date": extract_resolved_date(row),
                                "display_text": approved_fields_from_row(row),
                            },
                        )
                    )
    return documents


def serialize_documents(
    documents: list[Document],
) -> tuple[tuple[str, str, int, str, str, str, str], ...]:
    """Create an immutable cache input without merging any incident rows."""
    return tuple(
        (
            str(doc.metadata.get("source", "")),
            str(doc.metadata.get("sheet", "")),
            int(doc.metadata.get("row", 0)),
            str(doc.metadata.get("created_date", "")),
            str(doc.metadata.get("resolved_date", "")),
            str(doc.metadata.get("display_text", "")),
            doc.page_content,
        )
        for doc in documents
    )


def labelled_field_value(text: str, aliases: tuple[str, ...]) -> str:
    """Return a field value from the labelled one-row document text."""
    normalized_aliases = {normalize_column_name(alias) for alias in aliases}
    for line in text.splitlines():
        label, separator, value = line.partition(":")
        if separator and normalize_column_name(label) in normalized_aliases:
            return value.strip()
    return ""


def weighted_retrieval_text(text: str) -> str:
    """Emphasize issue description fields while preserving one vector per row."""
    title = labelled_field_value(text, ("Title", "Short Description", "Summary"))
    details = labelled_field_value(
        text, ("Detailed Description", "Description", "Details")
    )
    emphasized_sections = [text]
    # Repetition changes term weight inside this row's single embedding; it does
    # not create extra documents, chunks, overlap, or vectors.
    if title:
        emphasized_sections.extend([f"Primary incident title: {title}"] * 3)
    if details:
        emphasized_sections.extend([f"Primary incident details: {details}"] * 2)
    return "\n".join(emphasized_sections)


def meaningful_tokens(text: str) -> set[str]:
    """Normalize words for a small lexical signal alongside semantic search."""
    normalized = re.sub(r"\blog[ -]?in\b", "login", text.lower())
    normalized = re.sub(r"\bsign[ -]?in\b", "login", normalized)
    normalized = re.sub(
        r"\b(?:cannot|can't|unable to|not able to)\s+(?:login|authenticate)\b",
        "authenticationfailure login",
        normalized,
    )
    normalized = re.sub(
        r"\b(?:login|authentication)\s+(?:failure|failures|failed)\b",
        "authenticationfailure login",
        normalized,
    )
    normalized = re.sub(r"\b(?:users?|customers?|end users?)\b", "customer", normalized)
    normalized = re.sub(r"\b(?:apps?|applications?|banking portals?)\b", "application", normalized)
    stop_words = {
        "a", "an", "and", "are", "for", "in", "is", "of", "on", "the",
        "to", "was", "were", "with",
    }
    return {
        token.rstrip("s")
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) > 2 and token not in stop_words
    }


def lexical_coverage(query: str, incident: str) -> float:
    """Measure how many meaningful query terms occur in an incident row."""
    query_tokens = meaningful_tokens(query)
    if not query_tokens:
        return 0.0
    return len(query_tokens & meaningful_tokens(incident)) / len(query_tokens)


def field_weighted_lexical_coverage(query: str, incident: str) -> float:
    """Prioritize title and detailed description over secondary row fields."""
    title = labelled_field_value(
        incident, ("Title", "Short Description", "Summary")
    )
    details = labelled_field_value(
        incident, ("Detailed Description", "Description", "Details")
    )
    overall_score = lexical_coverage(query, incident)
    title_score = lexical_coverage(query, title) if title else overall_score
    detail_score = lexical_coverage(query, details) if details else overall_score
    return 0.50 * title_score + 0.35 * detail_score + 0.15 * overall_score


def authentication_availability_query(text: str) -> bool:
    """Identify queries about legitimate users being unable to authenticate."""
    normalized = text.lower()
    return bool(
        re.search(
            r"(?:cannot|can't|unable to|not able to)\s+(?:log[ -]?in|sign[ -]?in|authenticate)",
            normalized,
        )
        or re.search(
            r"(?:login|sign[ -]?in|authentication)\s+(?:failure|failures|failed|outage|error)",
            normalized,
        )
    )


def intent_adjustment(query: str, incident: str) -> float:
    """Reward matching login outages and reject security-only login events."""
    if not authentication_availability_query(query):
        return 0.0

    normalized = incident.lower()
    availability_markers = (
        "unable to verify",
        "unable to login",
        "unable to log in",
        "login failures",
        "authentication failure",
        "failed login attempts",
        "login attempts failed",
        "could not login",
        "could not log in",
        "generic error",
        "denial-of-service",
        "denial of service",
        "ddos",
        "login page unavailable",
    )
    security_markers = (
        "unauthorized access",
        "unauthorized login",
        "attacker",
        "compromised account",
        "compromised credential",
        "credential stuffing",
        "brute force",
        "data breach",
        "privileged administrator",
        "privileged account",
        "successful login from",
    )
    has_availability_signal = any(marker in normalized for marker in availability_markers)
    has_security_signal = any(marker in normalized for marker in security_markers)

    # Security incidents may mention many login attempts, but they are not service
    # availability matches when the user asks about legitimate users being blocked.
    if has_security_signal:
        return -0.40
    if has_availability_signal:
        return 0.12
    return 0.0


def hybrid_relevance_score(query: str, incident: str, semantic_score: float) -> tuple[float, float, float]:
    """Combine semantic, lexical, and intent signals into a bounded score."""
    word_score = field_weighted_lexical_coverage(query, incident)
    adjustment = intent_adjustment(query, incident)
    score = 0.65 * semantic_score + 0.35 * word_score + adjustment
    return max(0.0, min(1.0, score)), word_score, adjustment


def filter_by_six_month_window(
    documents: list[Document], issue_created_date: date
) -> list[Document]:
    """Keep incidents from the six calendar months before the current issue."""
    window_end = pd.Timestamp(issue_created_date)
    window_start = window_end - pd.DateOffset(months=6)
    filtered = []
    for document in documents:
        created_date = pd.to_datetime(
            document.metadata.get("created_date", ""), errors="coerce"
        )
        if pd.notna(created_date) and window_start <= created_date <= window_end:
            filtered.append(document)
    return filtered


def creation_date_sort_key(match: tuple[Document, float]) -> tuple[int, float]:
    """Sort dated incidents newest-first and leave undated rows at the end."""
    document, score = match
    created_date = pd.to_datetime(
        document.metadata.get("created_date", ""), errors="coerce"
    )
    timestamp = created_date.value if pd.notna(created_date) else -1
    return timestamp, score


@st.cache_resource(show_spinner=False)
def build_vector_store(
    api_key: str, records: tuple[tuple[str, str, int, str, str, str, str], ...]
) -> FAISS:
    """Create one vector per complete incident row."""
    documents = [
        Document(
            page_content=weighted_retrieval_text(content),
            metadata={
                "source": source,
                "sheet": sheet,
                "row": row,
                "created_date": created_date,
                "resolved_date": resolved_date,
                "display_text": display_text,
                "original_content": content,
            },
        )
        for source, sheet, row, created_date, resolved_date, display_text, content in records
    ]
    if not documents:
        raise ValueError("No non-empty incident rows were found.")

    # These reports are already split into small chunks. Disabling the automatic
    # context-length check avoids tiktoken downloading an encoding file at runtime,
    # which is commonly blocked on corporate networks.
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        check_embedding_ctx_length=False,
    )
    # There is intentionally no text splitter: each spreadsheet row becomes exactly
    # one vector, so content can never overlap with an adjacent incident row.
    return FAISS.from_documents(documents, embeddings)


def find_similar_incidents(
    query: str,
    api_key: str,
    documents: list[Document],
    similarity_threshold: float,
) -> tuple[str, list[tuple[Document, float]]]:
    """Return only incident rows that pass the configured similarity threshold."""
    vector_store = build_vector_store(api_key, serialize_documents(documents))
    # Evaluate every indexed incident row. The match threshold below—not a fixed
    # top-k cap—decides which incidents are returned.
    candidates = vector_store.similarity_search_with_score(query, k=len(documents))
    # OpenAI embeddings are unit-normalized. FAISS returns squared L2 distance,
    # which converts to cosine similarity as: cosine = 1 - distance / 2.
    scored_matches = []
    for doc, distance in candidates:
        # Search vectors use weighted text, while results and prompts retain the
        # original incident row without repeated fields.
        doc.page_content = str(doc.metadata.get("original_content", doc.page_content))
        semantic_score = max(0.0, min(1.0, 1.0 - float(distance) / 2.0))
        hybrid_score, word_score, adjustment = hybrid_relevance_score(
            query, doc.page_content, semantic_score
        )
        # Semantic retrieval supplies meaning; lexical coverage protects short,
        # explicit symptom queries such as "unable to login" from being diluted by
        # the longer full incident row.
        doc.metadata["semantic_score"] = semantic_score
        doc.metadata["lexical_score"] = word_score
        doc.metadata["intent_adjustment"] = adjustment
        scored_matches.append((doc, hybrid_score))
    scored_matches.sort(key=lambda item: item[1], reverse=True)
    matches = [
        (doc, score)
        for doc, score in scored_matches
        if score >= similarity_threshold
    ]
    # Final result order is chronological rather than similarity-ranked: newest
    # incidents first, oldest last, and rows without a creation date at the end.
    matches.sort(key=creation_date_sort_key, reverse=True)

    if not matches:
        return "I couldn't find a similar past incident in the repository.", []

    context = "\n\n---\n\n".join(
        f"Similarity: {score:.0%}\n{doc.page_content}" for doc, score in matches
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    answer = (PROMPT | model | StrOutputParser()).invoke(
        {"question": query, "context": context}
    )
    return answer, matches


def chat_with_repository(
    question: str,
    api_key: str,
    repository_documents: list[Document],
    current_issue: str,
    history: list[dict[str, str]],
) -> str:
    """Answer follow-ups using every row from the uploaded incident repository."""
    incident_context = "\n\n--- INCIDENT ROW ---\n\n".join(
        (
            f"Created date: {doc.metadata.get('created_date') or 'Unknown'}\n"
            f"Resolved time: {doc.metadata.get('resolved_date') or 'Unknown'}\n"
            f"{doc.page_content}"
        )
        for doc in repository_documents
    )
    prior_conversation = "\n".join(
        f"{message['role'].title()}: {message['content']}" for message in history[-8:]
    )
    system_prompt = f"""
You are an incident-analysis assistant with access to the entire user-uploaded
incident repository below. The current investigated issue is also supplied so you
can resolve phrases such as "this issue", "she", "he", and "that month" using the
conversation. Search across all repository rows when answering follow-ups; do not
restrict answers to the original similarity-search matches. Each distinct row is
one incident occurrence. Do not infer incidents, people, dates, or facts that are
not present. If the repository lacks the answer, say so clearly.

CURRENT INVESTIGATED ISSUE:
{current_issue or 'No current issue description is available.'}

COMPLETE UPLOADED INCIDENT REPOSITORY:
{incident_context}
"""
    user_prompt = f"""
Recent conversation:
{prior_conversation or 'No previous follow-up questions.'}

Current follow-up question:
{question}
"""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    response = model.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    return str(response.content)


def get_configured_api_key() -> str:
    """Prefer secrets/environment variables, then allow manual entry."""
    try:
        configured_key = st.secrets.get("OPENAI_API_KEY", "")
    except FileNotFoundError:
        configured_key = ""
    configured_key = configured_key or os.getenv("OPENAI_API_KEY", "")

    if configured_key:
        st.sidebar.success("OpenAI API key is configured")
        return configured_key
    return st.sidebar.text_input(
        "OpenAI API key",
        type="password",
        help="Used only to create embeddings and generate the answer.",
    )


def main() -> None:
    st.set_page_config(
        page_title="IncidentIQ — AI-Powered Incident Intelligence",
        page_icon="🔎",
        layout="wide",
    )
    st.title("🔎 IncidentIQ — AI-Powered Incident Intelligence")
    st.caption("Describe a new issue and find related incidents from the local repository.")

    st.session_state.setdefault("search_answer", "")
    st.session_state.setdefault("search_matches", [])
    st.session_state.setdefault("searched_issue", "")
    st.session_state.setdefault("results_chat", [])
    st.session_state.setdefault("upload_signature", ())
    st.session_state.setdefault("filter_signature", None)
    st.session_state.setdefault("uploader_version", 0)
    st.session_state.setdefault("current_issue_query", "")

    with st.sidebar:
        if st.button("↻ Reset workflow", use_container_width=True):
            st.session_state.search_answer = ""
            st.session_state.search_matches = []
            st.session_state.searched_issue = ""
            st.session_state.results_chat = []
            st.session_state.upload_signature = ()
            st.session_state.filter_signature = None
            st.session_state.current_issue_query = ""
            st.session_state.uploader_version += 1
            st.rerun()
        st.header("Setup")
        api_key = get_configured_api_key()
        uploaded_files = st.file_uploader(
            "Incident data (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            help="Each non-empty row is treated as one complete incident.",
            key=f"incident_upload_{st.session_state.uploader_version}",
        )
        upload_signature = tuple(
            (uploaded_file.name, uploaded_file.size) for uploaded_file in uploaded_files
        )
        if upload_signature != st.session_state.upload_signature:
            st.session_state.upload_signature = upload_signature
            st.session_state.search_answer = ""
            st.session_state.search_matches = []
            st.session_state.searched_issue = ""
            st.session_state.results_chat = []
        similarity_threshold = st.slider(
            "Minimum match score", 0.0, 1.0, 0.45, 0.05,
            key="minimum_match_score_v2",
            help=(
                "Hybrid score: 70% semantic similarity and 30% matching symptom "
                "words. Rows below this score are excluded."
            ),
        )
        search_scope = st.radio(
            "Search period",
            ["Last 6 months", "Whole repository"],
            help="The six-month option uses each incident row's creation date.",
        )
        issue_created_date = st.date_input(
            "Current issue created date",
            value=date.today(),
            disabled=search_scope == "Whole repository",
            help="The six-month search window ends on this date.",
        )
        filter_signature = (
            search_scope,
            issue_created_date.isoformat(),
            float(similarity_threshold),
        )
        if st.session_state.filter_signature is None:
            st.session_state.filter_signature = filter_signature
        elif filter_signature != st.session_state.filter_signature:
            st.session_state.filter_signature = filter_signature
            st.session_state.search_answer = ""
            st.session_state.search_matches = []
            st.session_state.searched_issue = ""
            st.session_state.results_chat = []

    try:
        documents = read_uploaded_incidents(uploaded_files) if uploaded_files else []
    except Exception as exc:
        st.error(f"The incident file could not be read: {exc}")
        return

    dated_count = sum(bool(doc.metadata.get("created_date")) for doc in documents)
    searchable_documents = (
        filter_by_six_month_window(documents, issue_created_date)
        if search_scope == "Last 6 months"
        else documents
    )

    with st.sidebar:
        st.metric("Incident rows available", len(documents))
        st.metric("Rows in selected period", len(searchable_documents))
        if uploaded_files:
            st.success("Using uploaded tabular data")
        else:
            st.warning("Upload a CSV or Excel file to search incidents.")
        if uploaded_files and search_scope == "Last 6 months" and dated_count == 0:
            st.warning(
                "No recognized creation-date column was found. Use `Created Date`, "
                "`Created At`, `Incident Date`, or `Opened Date`, or select the "
                "whole repository."
            )

    with st.form("incident_search"):
        query = st.text_area(
            "Describe the current incident",
            placeholder="Example: Users cannot log in because the database connection is failing.",
            height=130,
            key="current_issue_query",
        )
        search_button_column, reset_button_column = st.columns([2, 1])
        with search_button_column:
            submitted = st.form_submit_button(
                "Find similar incidents", type="primary", use_container_width=True
            )
        with reset_button_column:
            reset_results = st.form_submit_button(
                "Reset results", use_container_width=True
            )

    if reset_results:
        st.session_state.search_answer = ""
        st.session_state.search_matches = []
        st.session_state.searched_issue = ""
        st.session_state.results_chat = []
        st.rerun()

    if submitted:
        if not api_key:
            st.warning("Add your OpenAI API key in the sidebar to continue.")
            return
        if not query.strip():
            st.warning("Please describe the incident first.")
            return
        if not uploaded_files or not documents:
            st.warning("Upload a non-empty CSV or Excel incident file first.")
            return
        if not searchable_documents:
            st.warning(
                "No dated incident rows fall within the selected six-month period. "
                "Check the date column or select **Whole repository**."
            )
            return

        try:
            with st.spinner("Searching historical incidents..."):
                answer, matches = find_similar_incidents(
                    query.strip(), api_key, searchable_documents, similarity_threshold
                )
            st.session_state.search_answer = answer
            st.session_state.search_matches = matches
            st.session_state.searched_issue = query.strip()
            st.session_state.results_chat = []
        except Exception as exc:
            st.error(f"The search could not be completed: {exc}")
            return

    if not st.session_state.search_answer:
        st.info("Enter an incident description, then select **Find similar incidents**.")
        return

    result_column, chat_column = st.columns([1.35, 1], gap="large")

    with result_column:
        match_count = len(st.session_state.search_matches)
        st.subheader(f"Matched incidents: {match_count}")
        if st.session_state.search_matches:
            display_matches = sorted(
                st.session_state.search_matches,
                key=creation_date_sort_key,
                reverse=True,
            )
            export_rows = []
            for match, _score in display_matches:
                export_text = match.metadata.get("display_text")
                export_text = export_text or result_display_text(match.page_content)
                if not has_any_mapped_value(export_text):
                    export_text = safe_original_row_text(match.page_content)
                export_rows.append(display_text_to_record(export_text))
            export_csv = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇ Export results",
                data=export_csv,
                file_name="matched_incidents.csv",
                mime="text/csv",
                use_container_width=True,
            )
            for match, _score in display_matches:
                with st.container(border=True):
                    display_text = match.metadata.get("display_text")
                    display_text = display_text or result_display_text(match.page_content)
                    if not has_any_mapped_value(display_text):
                        display_text = safe_original_row_text(match.page_content)
                    st.text(display_text)

            with st.expander("View AI-generated analysis"):
                st.markdown(st.session_state.search_answer)
        else:
            st.markdown(st.session_state.search_answer)

    with chat_column:
        st.subheader("🤖 Incident Copilot")
        if not st.session_state.search_matches:
            st.info("Chat becomes available when the search returns incidents.")
        else:
            st.caption("Answers can use every row in the uploaded repository.")
            for message in st.session_state.results_chat:
                avatar = "🤖" if message["role"] == "assistant" else "🧑‍💻"
                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])

            with st.form("results_chat_form", clear_on_submit=True):
                follow_up = st.text_input(
                    "Ask a follow-up",
                    placeholder="How frequently has this issue occurred?",
                )
                ask_submitted = st.form_submit_button("Ask", type="primary")

            if ask_submitted and follow_up.strip():
                if not api_key:
                    st.warning("Add your OpenAI API key to use the chat.")
                else:
                    try:
                        with st.spinner("Analyzing the retrieved incidents..."):
                            chat_answer = chat_with_repository(
                                follow_up.strip(),
                                api_key,
                                documents,
                                st.session_state.searched_issue,
                                st.session_state.results_chat,
                            )
                        st.session_state.results_chat.extend(
                            [
                                {"role": "user", "content": follow_up.strip()},
                                {"role": "assistant", "content": chat_answer},
                            ]
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"The follow-up could not be answered: {exc}")


if __name__ == "__main__":
    main()
