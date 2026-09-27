import torch

from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer


MODEL_NAME = "naver/splade-cocondenser-ensembledistil"


print("Loading SPLADE model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForMaskedLM.from_pretrained(
    MODEL_NAME
)

model.eval()

print("SPLADE model loaded successfully.")


def encode_sparse(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    logits = outputs.logits

    attention_mask = inputs["attention_mask"]

    values = torch.log1p(
        torch.relu(logits)
    )

    values = (
        values
        * attention_mask.unsqueeze(-1)
    )

    values = values.max(
        dim=1
    ).values.squeeze()

    non_zero = torch.nonzero(
        values > 0
    ).squeeze()

    if non_zero.numel() == 0:

        return {
            "indices": [],
            "values": [],
        }

    if non_zero.dim() == 0:
        non_zero = non_zero.unsqueeze(0)

    sparse_indices = (
        non_zero.tolist()
    )

    sparse_values = (
        values[non_zero].tolist()
    )

    return {
        "indices": sparse_indices,
        "values": sparse_values,
    }
