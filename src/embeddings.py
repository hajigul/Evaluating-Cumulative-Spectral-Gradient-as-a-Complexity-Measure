import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm

from .config import DEVICE, BERT_MODEL_NAME, BATCH_SIZE


def get_bert_embedding(text_batch, tokenizer, model):
    """Get [CLS] embeddings for a batch of texts."""
    inputs = tokenizer(
        text_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def compute_hr_embeddings(triplets, unique_heads, unique_relations):
    """Pre-compute BERT embeddings for all unique heads and relations."""
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
    model.eval()

    print("Computing head embeddings...")
    head_embeddings = {}
    for i in tqdm(range(0, len(unique_heads), BATCH_SIZE), desc="Heads"):
        batch = unique_heads[i:i + BATCH_SIZE]
        embeddings = get_bert_embedding(batch, tokenizer, model)
        for h, emb in zip(batch, embeddings):
            head_embeddings[h] = emb

    print("Computing relation embeddings...")
    relation_embeddings = {}
    for i in tqdm(range(0, len(unique_relations), BATCH_SIZE), desc="Relations"):
        batch = unique_relations[i:i + BATCH_SIZE]
        embeddings = get_bert_embedding(batch, tokenizer, model)
        for r, emb in zip(batch, embeddings):
            relation_embeddings[r] = emb

    return head_embeddings, relation_embeddings