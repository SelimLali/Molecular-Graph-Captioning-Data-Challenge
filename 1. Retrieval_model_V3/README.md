
# Molecule–Text Retrieval V3

Improved retrieval model for the **Molecular Graph Captioning** challenge.

This model builds on the teacher’s baseline GCN+MSE retrieval system with several upgrades:
- **RoBERTa-base** text embeddings (better aligned with BERTScore).
- **GINEConv** graph encoder that uses atom and bond features.
- **Learned projection heads** for both graph and text embeddings.
- **Contrastive CLIP-style loss** with in-batch negatives and a learnable temperature.
- **Attention-based pooling** over graph nodes.
- Optional **train+validation retrieval bank** and **top-\(k\) medoid selection** at inference.

The result is a significantly stronger retrieval model, achieving a Kaggle score of about **0.60** on the
hidden test split.

---

## 1. Installation

From inside `1. Retrieval_model_V3/`:

```bash
pip install -r requirements.txt
```

This will install:
- `torch`, `torch_geometric`
- `transformers` (for RoBERTa)
- `pandas`, `numpy`, `tqdm`, and other utilities.

---

## 2. Data Setup

Place the preprocessed graphs under `data/`:

```text
1. Retrieval_model_V3/
├── data/
│   ├── train_graphs.pkl
│   ├── validation_graphs.pkl
│   └── test_graphs.pkl
└── ...
```

You can inspect the structure of these graphs using:

```bash
python inspect_graph_data.py
```

This will print basic statistics such as the number of graphs, nodes, edges, and the presence of captions.

---

## 3. Generate Text Embeddings (RoBERTa)

Before training the retriever, we precompute text embeddings for train/validation captions using RoBERTa-base:

```bash
python generate_text_embeddings_roberta.py
```

This script:
- Loads `train_graphs.pkl` and `validation_graphs.pkl`.
- Extracts the `description` field of each graph.
- Encodes descriptions using RoBERTa-base with mean-pooling over token embeddings.
- Saves the resulting tensors to:
  - `data/train_textemb_roberta-base_mean.pt`
  - `data/validation_textemb_roberta-base_mean.pt`

These embeddings are treated as the *initial text space* for training the retrieval model.

---

## 4. Train the Retrieval Model

Train the GNN-based retriever with a CLIP-style contrastive loss:

```bash
python train_retrieval_clip_gine.py
```

What this script does:

- Builds a GNN encoder with:
  - Atom embeddings (based on the 9 categorical features)
  - Bond embeddings (based on the 3 categorical edge features)
  - Several GINEConv layers
  - Attention pooling over nodes
  - A projection head to map to the shared embedding space
- Builds a small MLP projection head on top of RoBERTa embeddings.
- Uses a **symmetric infoNCE / CLIP-like loss** with in-batch negatives:
  - Text-to-graph and graph-to-text directions, combined.
  - A learnable temperature parameter.

Training hyperparameters include:
- Batch size (e.g., 96 or 128, depending on GPU memory).
- Number of epochs (e.g., 40–50).
- Adam optimizer with standard weight decay.
- A learnable logit scale (temperature) that grows over training.

The best checkpoint (lowest validation loss) is saved as:

```text
retrieval_v3_best.pt
```

---

## 5. Local Evaluation on Validation (BLEU / BERTScore)

To evaluate retrieval-based captions on the validation set:

```bash
python local_eval_retrieval_bleu_bertscore.py
```

This script:
1. Loads `retrieval_v3_best.pt` and the RoBERTa embeddings.
2. Builds a retrieval bank from the **training** captions.
3. For each validation graph:
   - Encodes its graph embedding.
   - Retrieves top-\(k\) nearest captions based on cosine similarity in the shared space.
   - Either uses **1-NN** or a **medoid@k** strategy to select the final caption.
4. Computes:
   - BLEU (sacreBLEU)
   - BERTScore F1 (with RoBERTa-base)
   - Optional “proxy” score combining the two.

You can adjust `k` (e.g. 5, 10, 20) inside the script to study how medoid selection affects metrics.

---

## 6. Generate Kaggle Submission

To produce a Kaggle-ready submission file:

```bash
python make_submission_retrieval.py
```

This script:
- Loads the trained model (`retrieval_v3_best.pt`)
- Builds a retrieval bank from **train** (and optionally **validation**) captions.
- For each test graph in `data/test_graphs.pkl`:
  - Computes its graph embedding.
  - Retrieves top-\(k\) candidates and applies the chosen selection rule (often medoid@10).
  - Writes a row `ID,description` to the CSV.

The resulting file is:

```text
submission_retrieval_v3.csv
```

You can upload this file directly to the Kaggle competition.

---

## 7. Notes and Tips

- For better performance, ensure the scripts run with `DEVICE=cuda` when a GPU is available.
- You can change batch size, hidden dimensions, number of GINE layers, or temperature initialization
  directly in `train_retrieval_clip_gine.py`.
- For further analysis and comparison with other retrieval variants, use the global plotting script
  in the project root (`1. plot_metrics_Retrieval_models.py`).
