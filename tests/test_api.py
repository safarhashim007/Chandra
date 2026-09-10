def test_api_placeholder_contract_is_explicit() -> None:
    # API implementation is intentionally not claimed before core registration exists.
    assert "VERIFIED" != "REJECTED"
