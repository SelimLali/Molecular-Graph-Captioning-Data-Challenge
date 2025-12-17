import pickle
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

MAX_LEN = 128
MODEL_NAME = "roberta-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mean_pool(last_hidden, attn_mask):
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)  # [B,T,1]
    summed = (last_hidden * mask).sum(dim=1)             # [B,H]
    denom = mask.sum(dim=1).clamp_min(1e-9)              # [B,1]
    return summed / denom

@torch.no_grad()
def embed_batch(texts, tok, mdl):
    enc = tok(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = mdl(**enc)
    emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
    emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.cpu().to(torch.float32)

def main():
    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModel.from_pretrained(MODEL_NAME, add_pooling_layer=False).to(DEVICE).eval()

    for split in ["train", "validation"]:
        pkl_path = f"data/{split}_graphs.pkl"
        out_path = f"data/{split}_textemb_{MODEL_NAME}_mean.pt"

        with open(pkl_path, "rb") as f:
            graphs = pickle.load(f)

        ids = [str(g.id) for g in graphs]
        texts = [g.description for g in graphs]

        id2emb = {}
        bs = 64 if DEVICE.type != "cpu" else 16

        for i in tqdm(range(0, len(texts), bs), total=(len(texts)+bs-1)//bs):
            batch_ids = ids[i:i+bs]
            batch_txt = texts[i:i+bs]
            embs = embed_batch(batch_txt, tok, mdl)
            for k, e in zip(batch_ids, embs):
                id2emb[k] = e

        torch.save(id2emb, out_path)
        print(f"Saved {len(id2emb)} embeddings to {out_path}")

if __name__ == "__main__":
    main()
