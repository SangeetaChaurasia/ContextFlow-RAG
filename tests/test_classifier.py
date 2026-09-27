from query_classifier import classify_question

def test_order_question():
    assert classify_question(
        "What is the status of ORD-83921?"
    ) == "order"


def test_account_question():
    assert classify_question(
        "How do I reset my password?"
    ) == "account"


def test_refund_question():
    assert classify_question(
        "How long does a refund take?"
    ) == "refund"


def test_shipping_question():
    assert classify_question(
        "Can I track my order?"
    ) == "shipping"


def test_general_question():
    assert classify_question(
        "What are your business hours?"
    ) == "general"
