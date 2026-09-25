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
