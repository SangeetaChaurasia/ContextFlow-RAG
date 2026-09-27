import re
from collections import Counter

def tokenize(text):

    return re.findall(
        r"\b[\w-]+\b",
        text.lower()
    )

def build_vocabulary(documents):
    vocabulary = {}

    for document in documents:
        for token in tokenize(document):
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)

    return vocabulary

def sparse_encode(text, vocabulary):
    tokens = tokenize(text)

    counts = Counter(tokens)

    indices = []
    values = []

    for token, count in counts.items():
        if token not in vocabulary:
            continue

        if re.search(r"\d", token) or "-" in token:
            weight = float(count) * 5.0
        else:
            weight = float(count)

        indices.append(vocabulary[token])
        values.append(weight)

    return {
        "indices": indices,
        "values": values,
    }
