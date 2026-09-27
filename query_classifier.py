import re


def classify_question(question):
    question_lower = question.lower()

    if re.search(r"\bORD-\d+\b", question, re.IGNORECASE):
        return "order"

    order_keywords = [
        "order status",
        "where is my order",
        "when was it shipped",
        "when did it ship",
        "tracking number",
    ]

    for keyword in order_keywords:
        if keyword in question_lower:
            return "order"

    password_keywords = [
        "password",
        "forgot password",
        "reset password",
    ]

    for keyword in password_keywords:
        if keyword in question_lower:
            return "account"

    refund_keywords = [
        "refund",
        "money back",
        "return",
    ]

    for keyword in refund_keywords:
        if keyword in question_lower:
            return "refund"

    shipping_keywords = [
        "shipping",
        "shipment",
        "delivery",
        "track order",
        "track my order",
        "tracking",
    ]

    for keyword in shipping_keywords:
        if keyword in question_lower:
            return "shipping"

    return "general"
