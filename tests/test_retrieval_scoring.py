import RAG_05_SimilarIncidentsFinder as app


QUERY = "Service requests fail during submission"

RELEVANT_INCIDENT = """
Incident ID: SAMPLE-1
Title: Service request submission failures
Detailed Description: Customers receive an error whenever they submit a request.
Root Cause: A downstream connection pool was exhausted.
"""

METADATA_ONLY_MATCH = """
Incident ID: SAMPLE-2
Title: Scheduled reporting maintenance
Detailed Description: The reporting job completed normally.
Related Incidents: Service requests fail during submission
"""


def test_embedding_text_emphasizes_title_and_details_without_creating_chunks():
    weighted = app.weighted_retrieval_text(RELEVANT_INCIDENT)

    assert weighted.count("Primary incident title:") == 3
    assert weighted.count("Primary incident details:") == 2
    assert weighted.startswith(RELEVANT_INCIDENT)


def test_lexical_score_prioritizes_title_and_detailed_description():
    relevant_score = app.field_weighted_lexical_coverage(
        QUERY, RELEVANT_INCIDENT
    )
    metadata_score = app.field_weighted_lexical_coverage(
        QUERY, METADATA_ONLY_MATCH
    )
    assert relevant_score > metadata_score


def test_hybrid_score_is_generic_and_bounded():
    score, lexical_score = app.hybrid_relevance_score(
        QUERY, RELEVANT_INCIDENT, semantic_score=0.75
    )

    assert 0.0 <= score <= 1.0
    assert 0.0 <= lexical_score <= 1.0
