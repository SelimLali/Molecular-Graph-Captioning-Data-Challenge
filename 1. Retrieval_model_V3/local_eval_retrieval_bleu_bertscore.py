#(retrieval model V3 with medoid reranking)
#Evaluation on validation set :
#retrieve from TRAIN captions
#score vs VAL ground truth
#reports 1-NN and medoid k={5,10,20}

import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sacrebleu import corpus_bleu
from bert_score import score as bertscore
from torch_geometric.data import Batch

from train_retrieval_clip_gine import (
    RetrievalModel, TRAIN_GRAPHS, VAL_GRAPHS, TRAIN_TEXTEMB, DEVICE
)

def collate_graphs(batch):
    return Batch.from_data_list(batch)

def load_id2desc(pkl_path):
    with open(pkl_path, "rb") as f:
        graphs = pickle.load(f)
    return {str(g.id): g.description for g in graphs}

@torch.no_grad()
def encode_graphs(model, graphs_pkl, batch_size=64):
    with open(graphs_pkl, "rb") as f:
        graphs = pickle.load(f)
    ids = [str(g.id) for g in graphs]
    dl = DataLoader(graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)

    embs = []
    model.eval()
    for bg in dl:
        bg = bg.to(DEVICE)
        embs.append(model.graph_enc(bg).cpu())
    embs = torch.cat(embs, dim=0)
    return ids, F.normalize(embs, dim=-1)

@torch.no_grad()
def project_train_text(model, train_id2emb):
    train_ids = list(train_id2emb.keys())
    mat = torch.stack([train_id2emb[i] for i in train_ids], dim=0).to(DEVICE)
    mat = model.text_proj(mat)
    mat = F.normalize(mat, dim=-1).cpu()
    return train_ids, mat

@torch.no_grad()
def retrieve_medoid(train_text_emb, sims_row, k):
    # sims_row: [Ntrain]
    topk = torch.topk(sims_row, k=k, largest=True).indices  # [k]
    cand = train_text_emb[topk]                             # [k, d]
    mean = F.normalize(cand.mean(dim=0, keepdim=True), dim=-1)  # [1,d]
    pick = torch.argmax((cand @ mean.t()).squeeze(1)).item()
    return topk[pick].item()

def evaluate(preds, refs):
    bleu = corpus_bleu(preds, [refs]).score
    P, R, F1 = bertscore(preds, refs, lang="en", model_type="roberta-base", verbose=False)
    berts_f1 = float(F1.mean().item())
    return bleu, berts_f1

def main():
    train_id2emb = torch.load(TRAIN_TEXTEMB)
    train_id2desc = load_id2desc(TRAIN_GRAPHS)
    val_id2desc   = load_id2desc(VAL_GRAPHS)

    text_dim = next(iter(train_id2emb.values())).numel()
    model = RetrievalModel(text_dim=text_dim).to(DEVICE)
    model.load_state_dict(torch.load("retrieval_v3_best.pt", map_location=DEVICE))
    model.eval()

    train_ids, train_text_emb = project_train_text(model, train_id2emb)

    val_ids, val_g_emb = encode_graphs(model, VAL_GRAPHS, batch_size=64)

    sims = val_g_emb @ train_text_emb.t()

    refs = [val_id2desc[vid] for vid in val_ids]

    # 1-NN
    nn_idx = sims.argmax(dim=-1).tolist()
    preds_1nn = [train_id2desc[train_ids[i]] for i in nn_idx]
    bleu, bsf1 = evaluate(preds_1nn, refs)
    print("Local Retrieval Eval on VAL (retrieve-from-TRAIN):")
    print(f"  1-NN:        BLEU={bleu:.3f} | BERTScoreF1={bsf1:.4f}")

    # medoid rerank
    for k in [5, 10, 20]:
        idxs = []
        for i in range(sims.size(0)):
            idxs.append(retrieve_medoid(train_text_emb, sims[i], k=k))
        preds = [train_id2desc[train_ids[i]] for i in idxs]
        bleu, bsf1 = evaluate(preds, refs)
        print(f"  Medoid@{k}:   BLEU={bleu:.3f} | BERTScoreF1={bsf1:.4f}")

if __name__ == "__main__":
    main()
