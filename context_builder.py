def build_context(results):
    context_parts = []

    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]
        title = metadata.get(
            "title",
            "Unknown"
        )

        category = metadata.get(
            "category",
            "Unknown"
        )

        text = metadata.get(
            "text",
            ""
        )

        context_parts.append(
            f"""
SOURCE {index}
Title: {title}
Category: {category}

{text}
""".strip()
        )

    return "\n\n---\n\n".join(
        context_parts
    )
