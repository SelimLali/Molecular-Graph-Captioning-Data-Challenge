import pickle
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from train_retrieval_clip_gine import RetrievalModel, TRAIN_GRAPHS, TRAIN_TEXTEMB, DEVICE

TEST_GRAPHS = "data/test_graphs.pkl"
#K_MEDOID set to the best value seen on validation set.
K_MEDOID = 20  # I tried 5/10/20; and pick the best from local_eval 

def collate_graphs(batch):
    return Batch.from_data_list(batch)

def load_id2desc(pkl_path):
    with open(pkl_path, "rb") as f:
        graphs = pickle.load(f)
    return {str(g.id): g.description for g in graphs}

@torch.no_grad()
def encode_graphs(model, graphs, batch_size=64):
    dl = DataLoader(graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
    embs = []
    for bg in dl:
        bg = bg.to(DEVICE)
        embs.append(model.graph_enc(bg).cpu())
    embs = torch.cat(embs, dim=0)
    return F.normalize(embs, dim=-1)

@torch.no_grad()
def project_train_text(model, train_id2emb):
    train_ids = list(train_id2emb.keys())
    mat = torch.stack([train_id2emb[i] for i in train_ids], dim=0).to(DEVICE)
    mat = model.text_proj(mat)
    mat = F.normalize(mat, dim=-1).cpu()
    return train_ids, mat

@torch.no_grad()
def retrieve_medoid(train_text_emb, sims_row, k):
    topk = torch.topk(sims_row, k=k, largest=True).indices
    cand = train_text_emb[topk]
    mean = F.normalize(cand.mean(dim=0, keepdim=True), dim=-1)
    pick = torch.argmax((cand @ mean.t()).squeeze(1)).item()
    return topk[pick].item()

def main():
    train_id2emb = torch.load(TRAIN_TEXTEMB)
    train_id2desc = load_id2desc(TRAIN_GRAPHS)

    text_dim = next(iter(train_id2emb.values())).numel()
    model = RetrievalModel(text_dim=text_dim).to(DEVICE)
    model.load_state_dict(torch.load("retrieval_v3_best.pt", map_location=DEVICE))
    model.eval()

    train_ids, train_text_emb = project_train_text(model, train_id2emb)

    with open(TEST_GRAPHS, "rb") as f:
        test_graphs = pickle.load(f)
    test_ids = [str(g.id) for g in test_graphs]

    test_g_emb = encode_graphs(model, test_graphs, batch_size=64)

    sims = test_g_emb @ train_text_emb.t()

    out = []
    for i, tid in enumerate(test_ids):
        if K_MEDOID <= 1:
            idx = int(torch.argmax(sims[i]).item())
        else:
            idx = retrieve_medoid(train_text_emb, sims[i], k=K_MEDOID)
        rid = train_ids[idx]
        out.append({"ID": tid, "description": train_id2desc[rid]})

    df = pd.DataFrame(out)
    df.to_csv("submission_retrieval_v3.csv", index=False)
    print("Saved: submission_retrieval_v3.csv")

if __name__ == "__main__":
    main()
