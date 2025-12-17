#model V3 includes:
#node+edge categorical embeddings
#GINEConv
#attention pooling
#hidden=384
#global graph features
#text projection head
#learnable temperature
#batch size 96 (and optional grad accumulation)

import os
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, GlobalAttention, global_mean_pool

# CONFIG
TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS   = "data/validation_graphs.pkl"

TRAIN_TEXTEMB = "data/train_textemb_roberta-base_mean.pt"
VAL_TEXTEMB   = "data/validation_textemb_roberta-base_mean.pt"

DEVICE = torch.device("cpu")  # CPU (because PyG+MPS may crash)

BATCH_SIZE = 96       
ACCUM_STEPS = 1        
EPOCHS = 45
LR = 2e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1

HIDDEN = 384
LAYERS = 4
POOLING = "attn"  # "attn" or "mean"

# Feature maps
x_map = {
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
e_map = {
    'bond_type': [
        'UNSPECIFIED','SINGLE','DOUBLE','TRIPLE','QUADRUPLE','QUINTUPLE','HEXTUPLE',
        'ONEANDAHALF','TWOANDAHALF','THREEANDAHALF','FOURANDAHALF','FIVEANDAHALF',
        'AROMATIC','IONIC','HYDROGEN','THREECENTER','DATIVEONE','DATIVE','DATIVEL',
        'DATIVER','OTHER','ZERO',
    ],
    'stereo': ['STEREONONE','STEREOANY','STEREOZ','STEREOE','STEREOCIS','STEREOTRANS'],
    'is_conjugated': [False, True],
}

BOND_SINGLE = 1
BOND_DOUBLE = 2
BOND_TRIPLE = 3
BOND_AROMATIC = 12


# DATA
class PreprocessedGraphDataset(Dataset):
    def __init__(self, graph_path: str, textemb_path: str = None):
        with open(graph_path, "rb") as f:
            self.graphs = pickle.load(f)
        self.ids = [str(g.id) for g in self.graphs]
        self.textemb = torch.load(textemb_path) if textemb_path is not None else None

    def __len__(self): return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        if self.textemb is None:
            return g
        return g, self.textemb[str(g.id)]

def collate_fn(batch):
    if isinstance(batch[0], tuple):
        graphs, embs = zip(*batch)
        return Batch.from_data_list(list(graphs)), torch.stack(list(embs), dim=0)
    return Batch.from_data_list(batch)


# MODEL
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

def scatter_sum(values_1d, index, dim_size):
    out = torch.zeros(dim_size, device=values_1d.device, dtype=values_1d.dtype)
    return out.scatter_add_(0, index, values_1d)

class MolEncoder(nn.Module):
    def __init__(self, out_dim: int, hidden=384, layers=4, dropout=0.1, pooling="attn"):
        super().__init__()
        self.dropout = dropout
        self.pooling = pooling

        self.embed = AtomBondEmbedding(emb_dim=hidden)

        self.in_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=hidden))
            self.norms.append(nn.LayerNorm(hidden))

        if pooling == "attn":
            self.readout = GlobalAttention(gate_nn=nn.Linear(hidden, 1))
        else:
            self.readout = None

        self.global_dim = 64
        self.global_mlp = nn.Sequential(
            nn.Linear(8, self.global_dim),
            nn.ReLU(),
            nn.Linear(self.global_dim, self.global_dim),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden + self.global_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def compute_global_feats(self, batch: Batch):
        device = batch.x.device
        num_graphs = batch.num_graphs

        x = batch.x.long()
        N = x.size(0)
        node_graph = batch.batch  # [N]

        ones_n = torch.ones(N, device=device)
        n_nodes = scatter_sum(ones_n, node_graph, num_graphs)

        arom = scatter_sum((x[:, 7] == 1).float(), node_graph, num_graphs)
        ring = scatter_sum((x[:, 8] == 1).float(), node_graph, num_graphs)
        arom_frac = arom / n_nodes.clamp_min(1.0)
        ring_frac = ring / n_nodes.clamp_min(1.0)

        # edges
        if batch.edge_index is None or batch.edge_index.numel() == 0:
            n_edges = torch.zeros(num_graphs, device=device)
            frac_single = frac_double = frac_triple = frac_arom = torch.zeros(num_graphs, device=device)
        else:
            E = batch.edge_index.size(1)
            edge_graph = batch.batch[batch.edge_index[0]]  # [E]
            ones_e = torch.ones(E, device=device)
            n_edges = scatter_sum(ones_e, edge_graph, num_graphs)

            bt = batch.edge_attr[:, 0].long()  # bond_type index
            frac_single = scatter_sum((bt == BOND_SINGLE).float(), edge_graph, num_graphs) / n_edges.clamp_min(1.0)
            frac_double = scatter_sum((bt == BOND_DOUBLE).float(), edge_graph, num_graphs) / n_edges.clamp_min(1.0)
            frac_triple = scatter_sum((bt == BOND_TRIPLE).float(), edge_graph, num_graphs) / n_edges.clamp_min(1.0)
            frac_arom   = scatter_sum((bt == BOND_AROMATIC).float(), edge_graph, num_graphs) / n_edges.clamp_min(1.0)

        feats = torch.stack([
            torch.log1p(n_nodes),
            torch.log1p(n_edges),
            arom_frac,
            ring_frac,
            frac_single,
            frac_double,
            frac_triple,
            frac_arom,
        ], dim=1)  # [G,8]

        return feats

    def forward(self, batch: Batch):
        x = batch.x.long()
        edge_attr = batch.edge_attr.long()

        h, e = self.embed(x, edge_attr)
        h = self.in_proj(h)

        for conv, ln in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, batch.edge_index, e)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = ln(h + h_res)

        if self.pooling == "attn":
            g = self.readout(h, batch.batch)
        else:
            g = global_mean_pool(h, batch.batch)

        global_feats = self.compute_global_feats(batch)             # [G,8]
        global_vec = self.global_mlp(global_feats)                  # [G,global_dim]
        g = torch.cat([g, global_vec], dim=-1)                      # [G, hidden+global_dim]

        g = self.out_proj(g)
        g = F.normalize(g, dim=-1)
        return g

class TextProjector(nn.Module):
    def __init__(self, dim: int, hidden_mult=1):
        super().__init__()
        h = dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(dim, h),
            nn.ReLU(),
            nn.Linear(h, dim),
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, dim=-1)

class RetrievalModel(nn.Module):
    """
    Wraps graph encoder + text projector + learnable temperature.
    """
    def __init__(self, text_dim: int):
        super().__init__()
        self.graph_enc = MolEncoder(out_dim=text_dim, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT, pooling=POOLING)
        self.text_proj = TextProjector(text_dim, hidden_mult=1)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07), dtype=torch.float32))

    def scale(self):
        # CLIP style: keep in a sane range
        return self.logit_scale.exp().clamp(1.0, 100.0)

def clip_loss(g, t, logit_scale):
    g = F.normalize(g, dim=-1)
    t = F.normalize(t, dim=-1)
    logits = (g @ t.t()) * logit_scale
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_g2t = F.cross_entropy(logits, labels)
    loss_t2g = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_g2t + loss_t2g)

# TRAIN
def main():
    print("DEVICE =", DEVICE)

    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, TRAIN_TEXTEMB)
    val_ds   = PreprocessedGraphDataset(VAL_GRAPHS, VAL_TEXTEMB)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # infer text dim
    text_dim = next(iter(torch.load(TRAIN_TEXTEMB).values())).numel()

    model = RetrievalModel(text_dim=text_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best = float("inf")
    for ep in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()

        tot, n = 0.0, 0
        for step, (graphs, text_emb) in enumerate(train_dl, start=1):
            graphs = graphs.to(DEVICE)
            text_emb = text_emb.to(DEVICE)

            g = model.graph_enc(graphs)
            t = model.text_proj(text_emb)

            loss = clip_loss(g, t, model.scale())
            (loss / ACCUM_STEPS).backward()

            if step % ACCUM_STEPS == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

            bs = graphs.num_graphs
            tot += loss.item() * bs
            n += bs

        train_loss = tot / n

        # quick val loss (contrastive)
        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for graphs, text_emb in val_dl:
                graphs = graphs.to(DEVICE)
                text_emb = text_emb.to(DEVICE)
                g = model.graph_enc(graphs)
                t = model.text_proj(text_emb)
                loss = clip_loss(g, t, model.scale())
                bs = graphs.num_graphs
                vtot += loss.item() * bs
                vn += bs

        val_loss = vtot / vn
        print(f"Epoch {ep:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | scale={model.scale().item():.2f}")

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "retrieval_v3_best.pt")
            print("  -> saved retrieval_v3_best.pt")

if __name__ == "__main__":
    main()
