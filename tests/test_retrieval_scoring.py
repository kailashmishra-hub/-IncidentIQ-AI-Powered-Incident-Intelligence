import RAG_05_SimilarIncidentsFinder as app


QUERY = "Users cannot login to the application"

LOGIN_OUTAGE = """
Incident ID: INC-2026-003
Title: Mobile banking app login failures - authentication token service outage
Detailed Description: Customers attempting to log into the mobile application
received 'Unable to verify your credentials' for all login attempts.
Root Cause: The OAuth token service exhausted its Redis connections.
"""

SECURITY_EVENT = """
Incident ID: INC-2026-007
Title: Unauthorized access attempt detected on internal admin console
Detailed Description: Failed login attempts against a privileged administrator
account were followed by a successful login from an unusual IP address.
Root Cause: The privileged account and VPN credential were compromised by an attacker.
"""

DDOS_LOGIN_OUTAGE = """
Incident ID: INC-2026-013
Title: Distributed denial-of-service attack against online banking login page
Detailed Description: Customers could not reach the online banking login page
during the DDoS attack.
"""


def test_login_outage_scores_above_security_event_at_equal_semantic_similarity():
    outage_score, _, outage_adjustment = app.hybrid_relevance_score(
        QUERY, LOGIN_OUTAGE, semantic_score=0.70
    )
    security_score, _, security_adjustment = app.hybrid_relevance_score(
        QUERY, SECURITY_EVENT, semantic_score=0.70
    )

    assert outage_adjustment > 0
    assert security_adjustment < 0
    assert outage_score > security_score
    assert outage_score >= 0.45
    assert security_score < 0.45


def test_login_synonyms_are_normalized_for_lexical_matching():
    query_tokens = app.meaningful_tokens(QUERY)
    incident_tokens = app.meaningful_tokens(LOGIN_OUTAGE)

    assert "authenticationfailure" in query_tokens
    assert "authenticationfailure" in incident_tokens
    assert "customer" in query_tokens
    assert "application" in query_tokens


def test_ddos_that_blocks_login_is_treated_as_availability_incident():
    score, _, adjustment = app.hybrid_relevance_score(
        QUERY, DDOS_LOGIN_OUTAGE, semantic_score=0.55
    )

    assert adjustment > 0
    assert score >= 0.45


def test_embedding_text_emphasizes_title_and_details_without_creating_chunks():
    weighted = app.weighted_retrieval_text(LOGIN_OUTAGE)

    assert weighted.count("Primary incident title:") == 3
    assert weighted.count("Primary incident details:") == 2
    assert weighted.startswith(LOGIN_OUTAGE)


def test_lexical_score_prioritizes_title_and_detailed_description():
    metadata_only_match = """
Incident ID: INC-X
Title: Scheduled account maintenance
Detailed Description: Routine maintenance completed normally.
Related Incidents: Users cannot login to the application
"""

    relevant_score = app.field_weighted_lexical_coverage(QUERY, LOGIN_OUTAGE)
    metadata_score = app.field_weighted_lexical_coverage(QUERY, metadata_only_match)
    assert relevant_score > metadata_score
