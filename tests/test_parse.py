from audit_llm_raters.scoring.parse import parse_score


def test_parse_valid_json_and_fences():
    assert parse_score('{"score": 4, "reason": "Clear."}', 1, 6).score == 4
    assert parse_score('```json\n{"score": 3, "reason": "Clear."}\n```', 1, 6).status == "ok"


def test_parse_failure_taxonomy():
    cases = [
        ("", "empty_response"),
        ("I cannot assess this essay.", "refusal"),
        ("not json", "invalid_json"),
        ('{"reason": "No score"}', "missing_score"),
        ('{"score": 3.0}', "non_integer_score"),
        ('{"score": true}', "non_integer_score"),
        ('{"score": 9}', "out_of_range"),
    ]
    for text, status in cases:
        assert parse_score(text, 1, 6).status == status
