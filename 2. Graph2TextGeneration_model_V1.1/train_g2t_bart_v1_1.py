import os, math, pickle, random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_batch

# CONFIG (v1.1)
TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS   = "data/validation_graphs.pkl"

MODEL_NAME   = "facebook/bart-base"
MAX_TOK_LEN  = 160

BATCH_SIZE   = 64
ACCUM_STEPS  = 1
EPOCHS       = 20

LR_GRAPH     = 2e-4      # graph+adapter
LR_BART      = 5e-5      # safe for BART
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
CLIP_NORM    = 1.0
SEED         = 42

# Graph encoder
GNN_HIDDEN   = 384
GNN_LAYERS   = 4
DROPOUT      = 0.1

# Latent graph tokens
LATENT_TOKENS = 64
LATENT_HEADS  = 8

# Training tricks
FREEZE_BART_EPOCHS = 2   # I train graph->tokens first, then unfreeze BART
USE_AMP            = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# Feature maps
x_map: Dict[str, List[Any]] = {
    'atomic_num': list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW','CHI_OTHER',
        'CHI_TETRAHEDRAL','CHI_ALLENE','CHI_SQUAREPLANAR','CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'num_radical_electrons': list(range(0, 5)),
    'hybridization': ['UNSPECIFIED','S','SP','SP2','SP3','SP3D','SP3D2','OTHER'],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}
e_map: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED','SINGLE','DOUBLE','TRIPLE','QUADRUPLE','QUINTUPLE','HEXTUPLE',
        'ONEANDAHALF','TWOANDAHALF','THREEANDAHALF','FOURANDAHALF','FIVEANDAHALF',
        'AROMATIC','IONIC','HYDROGEN','THREECENTER','DATIVEONE','DATIVE','DATIVEL',
        'DATIVER','OTHER','ZERO',
    ],
    'stereo': ['STEREONONE','STEREOANY','STEREOZ','STEREOE','STEREOCIS','STEREOTRANS'],
    'is_conjugated': [False, True],
}

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)


# Data
class GraphTextDataset(Dataset):
    def __init__(self, graph_pkl: str):
        with open(graph_pkl, "rb") as f:
            self.graphs = pickle.load(f)
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx: int):
        g = self.graphs[idx]
        return g, g.description

@dataclass
class BatchData:
    graphs: Batch
    labels: torch.Tensor

def collate_fn(batch: List[Tuple[Any, str]], tokenizer):
    graphs, texts = zip(*batch)
    bg = Batch.from_data_list(list(graphs))

    enc = tokenizer(
        list(texts),
        truncation=True,
        max_length=MAX_TOK_LEN,
        padding=True,
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return BatchData(graphs=bg, labels=labels)


# Embeddings
class AtomBondEmbedding(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.atom_keys = list(x_map.keys())
        self.atom_embs = nn.ModuleList([nn.Embedding(len(x_map[k]), emb_dim) for k in self.atom_keys])
        self.bond_keys = list(e_map.keys())
        self.bond_embs = nn.ModuleList([nn.Embedding(len(e_map[k]), emb_dim) for k in self.bond_keys])
        for emb in list(self.atom_embs) + list(self.bond_embs):
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self, x, edge_attr):
        h = 0
        for j, emb in enumerate(self.atom_embs):
            h = h + emb(x[:, j])
        e = 0
        for j, emb in enumerate(self.bond_embs):
            e = e + emb(edge_attr[:, j])
        return h, e


# Graph -> latent tokens
class GraphToLatents(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.embed = AtomBondEmbedding(GNN_HIDDEN)

        self.in_proj = nn.Sequential(
            nn.Linear(GNN_HIDDEN, GNN_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(GNN_LAYERS):
            mlp = nn.Sequential(
                nn.Linear(GNN_HIDDEN, GNN_HIDDEN),
                nn.ReLU(),
                nn.Linear(GNN_HIDDEN, GNN_HIDDEN),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=GNN_HIDDEN))
            self.norms.append(nn.LayerNorm(GNN_HIDDEN))

        self.node_to_dmodel = nn.Linear(GNN_HIDDEN, d_model)

        self.latent_q = nn.Parameter(torch.randn(LATENT_TOKENS, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=LATENT_HEADS, dropout=DROPOUT, batch_first=True
        )

        # Adapter can help match BART embedding statistics
        self.adapter = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(4 * d_model, d_model),
        )
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, batch: Batch):
        x = batch.x.long()
        edge_attr = batch.edge_attr.long()

        h, e = self.embed(x, edge_attr)
        h = self.in_proj(h)

        for conv, ln in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, batch.edge_index, e)
            h = F.relu(h)
            h = F.dropout(h, p=DROPOUT, training=self.training)
            h = ln(h + h_res)

        h_dense, mask = to_dense_batch(h, batch.batch)          # [B,Nmax,H], [B,Nmax]
        h_dense = self.node_to_dmodel(h_dense)                  # [B,Nmax,d]
        key_padding_mask = ~mask                                # True for PAD

        B = h_dense.size(0)
        q = self.latent_q.unsqueeze(0).expand(B, -1, -1)        # [B,L,d]

        lat, _ = self.cross_attn(q, h_dense, h_dense, key_padding_mask=key_padding_mask, need_weights=False)

        lat = self.out_ln(lat + self.adapter(lat))
        lat_mask = torch.ones((B, LATENT_TOKENS), dtype=torch.long, device=lat.device)
        return lat, lat_mask

class Graph2TextBART(nn.Module):
    def __init__(self, bart: BartForConditionalGeneration):
        super().__init__()
        self.bart = bart
        self.graph2lat = GraphToLatents(d_model=bart.config.d_model)

    def forward(self, graphs: Batch, labels: torch.Tensor):
        lat, lat_mask = self.graph2lat(graphs)
        out = self.bart(inputs_embeds=lat, attention_mask=lat_mask, labels=labels)
        return out.loss

    @torch.no_grad()
    def generate(self, graphs: Batch, **gen_kwargs):
        lat, lat_mask = self.graph2lat(graphs)
        return self.bart.generate(inputs_embeds=lat, attention_mask=lat_mask, **gen_kwargs)

def set_bart_trainable(bart: BartForConditionalGeneration, trainable: bool):
    for p in bart.parameters():
        p.requires_grad = trainable

@torch.no_grad()
def eval_loss(model, loader, amp_enabled: bool):
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        graphs = batch.graphs.to(DEVICE)
        labels = batch.labels.to(DEVICE)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            loss = model(graphs, labels)
        tot += float(loss.item()) * labels.size(0)
        n += labels.size(0)
    return tot / max(1, n)

def main():
    set_seed(SEED)
    print("DEVICE =", DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    bart = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    bart.resize_token_embeddings(len(tokenizer))

    model = Graph2TextBART(bart=bart).to(DEVICE)

    train_ds = GraphTextDataset(TRAIN_GRAPHS)
    val_ds   = GraphTextDataset(VAL_GRAPHS)

    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=2, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=2, pin_memory=True
    )

    # Two LR groups
    graph_params = list(model.graph2lat.parameters())
    bart_params  = [p for p in model.bart.parameters()]

    opt = torch.optim.AdamW(
        [
            {"params": graph_params, "lr": LR_GRAPH},
            {"params": bart_params,  "lr": LR_BART},
        ],
        weight_decay=WEIGHT_DECAY
    )

    steps_per_epoch = math.ceil(len(train_dl) / ACCUM_STEPS)
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = int(WARMUP_RATIO * total_steps)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE.type == "cuda"))

    best = float("inf")
    opt.zero_grad(set_to_none=True)

    for ep in range(1, EPOCHS + 1):
        # Freeze schedule
        if ep <= FREEZE_BART_EPOCHS:
            set_bart_trainable(model.bart, False)
        else:
            set_bart_trainable(model.bart, True)

        model.train()
        running, n = 0.0, 0

        for step, batch in enumerate(train_dl, start=1):
            graphs = batch.graphs.to(DEVICE)
            labels = batch.labels.to(DEVICE)

            with torch.amp.autocast(device_type="cuda", enabled=scaler.is_enabled()):
                loss = model(graphs, labels)
                loss_scaled = loss / ACCUM_STEPS

            scaler.scale(loss_scaled).backward()

            if step % ACCUM_STEPS == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

            running += float(loss.item()) * labels.size(0)
            n += labels.size(0)

        train_loss = running / max(1, n)
        val_loss = eval_loss(model, val_dl, amp_enabled=scaler.is_enabled())
        print(f"Epoch {ep:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": MODEL_NAME,
                    "max_tok_len": MAX_TOK_LEN,
                    "latent_tokens": LATENT_TOKENS,
                },
                "g2t_v1_1_best.pt"
            )
            print("  -> saved g2t_v1_1_best.pt")

    print("Done.")

if __name__ == "__main__":
    main()
