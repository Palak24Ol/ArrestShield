from arrestshield.entities import entity_type_counts, extract_entities


def test_extracts_operational_entities_and_redacts_sensitive_values() -> None:
    text = (
        "I am calling from CBI. Transfer Rs 25,000 to victim.name@okaxis using PhonePe. "
        "Call +91 98765 43210 and use OTP 482911. Visit https://fake.example/pay."
    )
    entities = extract_entities(text)
    counts = entity_type_counts(entities)
    assert counts["authority_organisation"] == 1
    assert counts["monetary_amount"] == 1
    assert counts["upi_id"] == 1
    assert counts["payment_app"] == 1
    assert counts["phone_number"] == 1
    assert counts["otp_code"] == 1
    assert counts["url"] == 1

    public = [entity.public_dict() for entity in entities]
    serialized = repr(public)
    assert "victim.name@okaxis" not in serialized
    assert "98765 43210" not in serialized
    assert "482911" not in serialized
    assert "***@okaxis" in serialized
    assert "***3210" in serialized


def test_context_prevents_arbitrary_numbers_from_becoming_accounts() -> None:
    text = "We will meet at 123456789012 on 20260812. My account number is 1234 5678 9012."
    entities = extract_entities(text)
    accounts = [entity for entity in entities if entity.entity_type == "bank_account_candidate"]
    assert len(accounts) == 1
    assert accounts[0].normalized_value == "123456789012"


def test_email_and_upi_are_disambiguated() -> None:
    entities = extract_entities("Write to help@example.org or pay shield@ybl")
    counts = entity_type_counts(entities)
    assert counts == {"email": 1, "upi_id": 1}


def test_url_span_excludes_sentence_punctuation() -> None:
    text = "Open https://example.org/pay). Then stop."
    entity = next(item for item in extract_entities(text) if item.entity_type == "url")
    assert text[entity.start : entity.end] == "https://example.org/pay"

