"""
Plot of ALTEGRAD Molecular Graph Captioning — Graph2Text generation runs.

Metrics (loss curves, val BLEU/BERTScore, Kaggle scores)
- Training curves (train/val loss) per model
- Best val loss per model
- Val BLEU / BERTScore and Kaggle score comparisons
- Decoding grid-search analysis for V1.1 (proxy/metrics vs max_new_tokens,...)

Run:
  python Graph2Text generation runs_metrics.py

Outputs:
  figures_generation_models/*.png
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt



# 0) Hard-coded experiment logs

runs = {
    "V0 (GPT-2 prefix)": {
        "train_loss": [1.4232, 1.0128, 0.8853, 0.8018, 0.7441, 0.6921, 0.6466, 0.6082, 0.5743, 0.5416, 0.5136, 0.4878, 0.4652, 0.4434, 0.4223],
        "val_loss":   [1.0325, 0.8915, 0.8175, 0.7668, 0.7317, 0.7055, 0.6909, 0.6697, 0.6585, 0.6497, 0.6400, 0.6351, 0.6317, 0.6313, 0.6339],
        "val_bleu": 24.500,
        "val_bertscore_f1": 0.8915,
        "kaggle": 0.545,
    },
    "V1 (GPT-2, decode tweak)": {
        "train_loss": [1.9159, 1.0077, 0.8296, 0.7264, 0.6517, 0.5924, 0.5424, 0.4995, 0.4626, 0.4312, 0.4049, 0.3845],
        "val_loss":   [1.0513, 0.8585, 0.7808, 0.7357, 0.7049, 0.6888, 0.6751, 0.6720, 0.6614, 0.6589, 0.6577, 0.6572],
        "val_bleu": 23.000,
        "val_bertscore_f1": 0.8831,
        "kaggle": 0.520,
    },
    "V1.1 (best training, before decode tuning)": {
        "train_loss": [4.8078, 4.0868, 3.9197, 1.6077, 1.1155, 0.9674, 0.8737, 0.8055, 0.7493, 0.7024,
                       0.6631, 0.6275, 0.5956, 0.5672, 0.5416, 0.5172, 0.4955, 0.4763, 0.4585, 0.4425,
                       0.4285, 0.4161, 0.4056, 0.3972, 0.3904],
        "val_loss":   [3.6023, 3.4802, 3.1159, 1.0966, 0.9389, 0.8474, 0.7924, 0.7526, 0.7197, 0.6948,
                       0.6744, 0.6616, 0.6443, 0.6348, 0.6281, 0.6233, 0.6144, 0.6115, 0.6124, 0.6109,
                       0.6057, 0.6113, 0.6088, 0.6104, 0.6112],
        "val_bleu": 32.158,
        "val_bertscore_f1": 0.9140,
        "kaggle": 0.602,
    },
    "V1.1 (tuned decoding)": {
        "train_loss": None,
        "val_loss": None,
        "val_bleu": 37.288,
        "val_bertscore_f1": 0.9188,
        "kaggle": 0.619,
    },
    "V1.2 (BART-large, many latent tokens)": {
        "train_loss": [4.6021, 3.9404, 4.1667, 1.4781, 1.0574, 0.9378, 0.8615, 0.8047, 0.7614, 0.7232,
                       0.6921, 0.6647, 0.6410, 0.6190, 0.6001, 0.5828, 0.5670, 0.5526, 0.5405, 0.5291,
                       0.5182, 0.5088, 0.5003, 0.4934, 0.4874, 0.4817, 0.4780, 0.4744],
        "val_loss":   [7.7620, 9.9615, 5.0755, 1.2231, 1.0861, 1.0814, 1.0364, 0.9784, 1.0036, 0.9527,
                       1.0829, 1.1401, 1.0813, 1.1246, 1.2492, 1.3266, 1.1457, 1.2011, 1.2196, 1.2537,
                       1.3353, 1.3346, 1.3556, 1.3873, 1.4265, 1.4133, 1.4423, 1.4658],
        "val_bleu": None,
        "val_bertscore_f1": None,
        "kaggle": 0.496,
    },
    "V2 (RAG: retriever + BART-large generator)": {
        # Run with 2 phases: retriever training and generator training
        "retriever_train_loss": [4.6002, 3.1524, 2.4709, 2.0338, 1.7637, 1.5748, 1.4006, 1.2749, 1.1644, 1.0598,
                                 0.9764, 0.9059, 0.8517, 0.7803, 0.7306],
        "generator_train_loss": [1.9562, 1.1360, 0.9147, 0.8311, 0.7807, 0.7438, 0.7162, 0.6962, 0.6810, 0.6712],
        "generator_val_loss":   [1.5651, 0.9167, 0.8310, 0.7920, 0.7604, 0.7495, 0.7289, 0.7157, 0.7104, 0.7112],
        "val_bleu": None,
        "val_bertscore_f1": None,
        "kaggle": 0.582,
    },
}

# Decoding grid search logs (model V1.1)
decode_grid_v1_1 = [
    # beams=5, max_new=80
    {"beams":5,"max_new":80,"lenp":0.8,"nrep":3,"bleu":30.523,"f1":0.9140,"proxy":0.6096},
    {"beams":5,"max_new":80,"lenp":0.8,"nrep":4,"bleu":31.858,"f1":0.9164,"proxy":0.6175},
    {"beams":5,"max_new":80,"lenp":0.9,"nrep":3,"bleu":30.620,"f1":0.9140,"proxy":0.6101},
    {"beams":5,"max_new":80,"lenp":0.9,"nrep":4,"bleu":31.981,"f1":0.9164,"proxy":0.6181},
    {"beams":5,"max_new":80,"lenp":1.0,"nrep":3,"bleu":30.773,"f1":0.9140,"proxy":0.6108},
    {"beams":5,"max_new":80,"lenp":1.0,"nrep":4,"bleu":32.208,"f1":0.9165,"proxy":0.6193},
    {"beams":5,"max_new":80,"lenp":1.1,"nrep":3,"bleu":30.859,"f1":0.9140,"proxy":0.6113},
    {"beams":5,"max_new":80,"lenp":1.1,"nrep":4,"bleu":32.222,"f1":0.9163,"proxy":0.6193},

    # beams=5, max_new=90
    {"beams":5,"max_new":90,"lenp":0.8,"nrep":3,"bleu":31.900,"f1":0.9148,"proxy":0.6169},
    {"beams":5,"max_new":90,"lenp":0.8,"nrep":4,"bleu":33.755,"f1":0.9175,"proxy":0.6275},
    {"beams":5,"max_new":90,"lenp":0.9,"nrep":3,"bleu":32.035,"f1":0.9148,"proxy":0.6176},
    {"beams":5,"max_new":90,"lenp":0.9,"nrep":4,"bleu":33.871,"f1":0.9175,"proxy":0.6281},
    {"beams":5,"max_new":90,"lenp":1.0,"nrep":3,"bleu":32.208,"f1":0.9148,"proxy":0.6184},
    {"beams":5,"max_new":90,"lenp":1.0,"nrep":4,"bleu":34.040,"f1":0.9175,"proxy":0.6289},
    {"beams":5,"max_new":90,"lenp":1.1,"nrep":3,"bleu":32.324,"f1":0.9148,"proxy":0.6190},
    {"beams":5,"max_new":90,"lenp":1.1,"nrep":4,"bleu":34.160,"f1":0.9174,"proxy":0.6295},

    # beams=5, max_new=110
    {"beams":5,"max_new":110,"lenp":0.8,"nrep":3,"bleu":33.821,"f1":0.9156,"proxy":0.6269},
    {"beams":5,"max_new":110,"lenp":0.8,"nrep":4,"bleu":35.924,"f1":0.9185,"proxy":0.6389},
    {"beams":5,"max_new":110,"lenp":0.9,"nrep":3,"bleu":33.974,"f1":0.9156,"proxy":0.6277},
    {"beams":5,"max_new":110,"lenp":0.9,"nrep":4,"bleu":36.106,"f1":0.9185,"proxy":0.6398},
    {"beams":5,"max_new":110,"lenp":1.0,"nrep":3,"bleu":34.175,"f1":0.9157,"proxy":0.6287},
    {"beams":5,"max_new":110,"lenp":1.0,"nrep":4,"bleu":36.283,"f1":0.9185,"proxy":0.6406},
    {"beams":5,"max_new":110,"lenp":1.1,"nrep":3,"bleu":34.339,"f1":0.9157,"proxy":0.6296},
    {"beams":5,"max_new":110,"lenp":1.1,"nrep":4,"bleu":36.376,"f1":0.9183,"proxy":0.6410},

    # beams=5, max_new=130
    {"beams":5,"max_new":130,"lenp":0.8,"nrep":3,"bleu":34.600,"f1":0.9157,"proxy":0.6309},
    {"beams":5,"max_new":130,"lenp":0.8,"nrep":4,"bleu":36.727,"f1":0.9189,"proxy":0.6431},
    {"beams":5,"max_new":130,"lenp":0.9,"nrep":3,"bleu":34.760,"f1":0.9157,"proxy":0.6317},
    {"beams":5,"max_new":130,"lenp":0.9,"nrep":4,"bleu":36.969,"f1":0.9189,"proxy":0.6443},
    {"beams":5,"max_new":130,"lenp":1.0,"nrep":3,"bleu":34.946,"f1":0.9158,"proxy":0.6326},
    {"beams":5,"max_new":130,"lenp":1.0,"nrep":4,"bleu":37.183,"f1":0.9189,"proxy":0.6454},
    {"beams":5,"max_new":130,"lenp":1.1,"nrep":3,"bleu":35.105,"f1":0.9158,"proxy":0.6334},
    {"beams":5,"max_new":130,"lenp":1.1,"nrep":4,"bleu":37.288,"f1":0.9188,"proxy":0.6458}, 

    # beams=8, max_new=80
    {"beams":8,"max_new":80,"lenp":0.8,"nrep":3,"bleu":30.443,"f1":0.9135,"proxy":0.6090},
    {"beams":8,"max_new":80,"lenp":0.8,"nrep":4,"bleu":31.934,"f1":0.9164,"proxy":0.6179},
    {"beams":8,"max_new":80,"lenp":0.9,"nrep":3,"bleu":30.516,"f1":0.9135,"proxy":0.6093},
    {"beams":8,"max_new":80,"lenp":0.9,"nrep":4,"bleu":32.004,"f1":0.9164,"proxy":0.6182},
    {"beams":8,"max_new":80,"lenp":1.0,"nrep":3,"bleu":30.719,"f1":0.9135,"proxy":0.6103},
    {"beams":8,"max_new":80,"lenp":1.0,"nrep":4,"bleu":32.183,"f1":0.9163,"proxy":0.6190},
    {"beams":8,"max_new":80,"lenp":1.1,"nrep":3,"bleu":30.881,"f1":0.9135,"proxy":0.6112},
    {"beams":8,"max_new":80,"lenp":1.1,"nrep":4,"bleu":32.352,"f1":0.9163,"proxy":0.6199},
]



# 1) Utilities

OUTDIR = "figures_generation_models"
os.makedirs(OUTDIR, exist_ok=True)

def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def best_min(x):
    if x is None:
        return None
    return float(np.min(np.array(x, dtype=float)))

def maybe_list(v):
    return v if v is not None else []

def lineplot_train_val(title, train, val, outpath):
    epochs = np.arange(1, max(len(train), len(val)) + 1)
    plt.figure()
    if train is not None:
        plt.plot(np.arange(1, len(train) + 1), train, label="train_loss")
    if val is not None:
        plt.plot(np.arange(1, len(val) + 1), val, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    savefig(outpath)



# 2) Training curves

# Per-model curves
for name, d in runs.items():
    if d.get("train_loss") is not None or d.get("val_loss") is not None:
        if d.get("train_loss") is not None or d.get("val_loss") is not None:
            lineplot_train_val(
                title=f"Training curve — {name}",
                train=d.get("train_loss"),
                val=d.get("val_loss"),
                outpath=os.path.join(OUTDIR, f"loss_curve_{name.replace(' ', '_').replace('/', '_')}.png"),
            )

# model V2 has 2 phases: retriever + generator
v2 = runs["V2 (RAG: retriever + BART-large generator)"]
plt.figure()
plt.plot(np.arange(1, len(v2["retriever_train_loss"]) + 1), v2["retriever_train_loss"], label="retriever_train_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("V2 retriever training curve")
plt.legend()
savefig(os.path.join(OUTDIR, "loss_curve_V2_retriever.png"))

lineplot_train_val(
    title="V2 generator training curve (RAG)",
    train=v2["generator_train_loss"],
    val=v2["generator_val_loss"],
    outpath=os.path.join(OUTDIR, "loss_curve_V2_generator.png"),
)

# Overlay of validation losses
plt.figure()
for name, d in runs.items():
    vl = d.get("val_loss")
    if vl is not None:
        plt.plot(np.arange(1, len(vl) + 1), vl, label=name)

plt.plot(np.arange(1, len(v2["generator_val_loss"]) + 1), v2["generator_val_loss"], label="V2 (RAG) — generator val_loss")
plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.title("Validation loss comparison (note: different architectures/scales)")
plt.legend(fontsize=8)
savefig(os.path.join(OUTDIR, "val_loss_overlay.png"))



# 3) Summary bars: best val loss, val BLEU/F1, Kaggle

# Best validation loss (min) per model
names = []
best_vloss = []
for name, d in runs.items():
    vl = d.get("val_loss")
    if vl is not None:
        names.append(name)
        best_vloss.append(best_min(vl))

# include V2 generator best val_loss 
names.append("V2 (RAG) — generator")
best_vloss.append(best_min(v2["generator_val_loss"]))

plt.figure()
x = np.arange(len(names))
plt.bar(x, best_vloss)
plt.xticks(x, names, rotation=40, ha="right")
plt.ylabel("Best validation loss (min over epochs)")
plt.title("Best validation loss across models")
savefig(os.path.join(OUTDIR, "best_val_loss_bar.png"))


# Val BLEU / BERTScore F1 bars
metric_names = []
val_bleu = []
val_f1 = []
for name, d in runs.items():
    if d.get("val_bleu") is not None and d.get("val_bertscore_f1") is not None:
        metric_names.append(name)
        val_bleu.append(d["val_bleu"])
        val_f1.append(d["val_bertscore_f1"])

# BLEU bar
plt.figure()
x = np.arange(len(metric_names))
plt.bar(x, val_bleu)
plt.xticks(x, metric_names, rotation=40, ha="right")
plt.ylabel("Validation BLEU (sacrebleu)")
plt.title("Validation BLEU comparison")
savefig(os.path.join(OUTDIR, "val_bleu_bar.png"))

# BERTScore bar
plt.figure()
x = np.arange(len(metric_names))
plt.bar(x, val_f1)
plt.xticks(x, metric_names, rotation=40, ha="right")
plt.ylabel("Validation BERTScore F1")
plt.title("Validation BERTScore comparison")
savefig(os.path.join(OUTDIR, "val_bertscore_bar.png"))

# Kaggle leaderboard bars
k_names = []
k_scores = []
for name, d in runs.items():
    if d.get("kaggle") is not None:
        k_names.append(name)
        k_scores.append(d["kaggle"])

plt.figure()
x = np.arange(len(k_names))
plt.bar(x, k_scores)
plt.xticks(x, k_names, rotation=40, ha="right")
plt.ylabel("Kaggle score (test)")
plt.title("Kaggle score by model (generation)")
savefig(os.path.join(OUTDIR, "kaggle_score_bar.png"))


# 4) Decoding grid-search analysis (model V1.1)

# Filter to beams=5 only
grid = [r for r in decode_grid_v1_1 if r["beams"] == 5]

# Best proxy vs max_new_tokens for nrep=3 and nrep=4 (best over length_penalty)
def best_over_lenp(max_new, nrep):
    candidates = [r for r in grid if r["max_new"] == max_new and r["nrep"] == nrep]
    if not candidates:
        return None
    best = max(candidates, key=lambda x: x["proxy"])
    return best

max_news = sorted({r["max_new"] for r in grid})
best_nrep3 = [best_over_lenp(m, 3) for m in max_news]
best_nrep4 = [best_over_lenp(m, 4) for m in max_news]

plt.figure()
plt.plot(max_news, [b["proxy"] if b else np.nan for b in best_nrep3], marker="o", label="best proxy (nrep=3)")
plt.plot(max_news, [b["proxy"] if b else np.nan for b in best_nrep4], marker="o", label="best proxy (nrep=4)")
plt.xlabel("max_new_tokens")
plt.ylabel("Proxy score (your BLEU/F1 composite)")
plt.title("V1.1 decoding: proxy improves with longer max_new_tokens")
plt.legend()
savefig(os.path.join(OUTDIR, "v1_1_proxy_vs_maxnew.png"))

# Best BLEU and F1 vs max_new_tokens
plt.figure()
plt.plot(max_news, [b["bleu"] if b else np.nan for b in best_nrep4], marker="o")
plt.xlabel("max_new_tokens")
plt.ylabel("BLEU")
plt.title("V1.1 decoding: best BLEU (nrep=4, best over length_penalty)")
savefig(os.path.join(OUTDIR, "v1_1_bleu_vs_maxnew_best_nrep4.png"))

plt.figure()
plt.plot(max_news, [b["f1"] if b else np.nan for b in best_nrep4], marker="o")
plt.xlabel("max_new_tokens")
plt.ylabel("BERTScore F1")
plt.title("V1.1 decoding: best BERTScore F1 (nrep=4, best over length_penalty)")
savefig(os.path.join(OUTDIR, "v1_1_f1_vs_maxnew_best_nrep4.png"))

# Scatter: proxy vs BLEU for beams=5 (tradeoff / correlation)
plt.figure()
plt.scatter([r["bleu"] for r in grid], [r["proxy"] for r in grid])
plt.xlabel("BLEU")
plt.ylabel("Proxy")
plt.title("V1.1 decoding grid: proxy vs BLEU (beams=5)")
savefig(os.path.join(OUTDIR, "v1_1_proxy_vs_bleu_scatter.png"))


# 5) Tiny summary

print(f"Saved figures to: {OUTDIR}\n")
print("Quick summary (Kaggle):")
for name, d in runs.items():
    if d.get("kaggle") is not None:
        print(f"  {name:40s} -> {d['kaggle']:.3f}")

print("\nDone.")
