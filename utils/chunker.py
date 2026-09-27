def chunk_text(
   text: str,
   chunk_size: int = 500,
   chunk_overlap: int = 100,
) -> list[str]:

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be less than chunk_size"
        )

    words = text.split()

    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1

        if current_length + word_length > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            overlap_words = []

            overlap_length = 0

            for previous_word in reversed(current_chunk):
                word_length = len(previous_word) + 1

                if overlap_length + word_length > chunk_overlap:
                    break

                overlap_words.insert(0, previous_word)
                overlap_length += word_length

            current_chunk = overlap_words + [word]

            current_length = sum(
                len(item) + 1
                for item in current_chunk
            )

        else:
            current_chunk.append(word)
            current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
