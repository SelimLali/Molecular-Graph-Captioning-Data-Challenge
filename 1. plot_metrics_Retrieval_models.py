import os
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Raw metrics from retrieval models runs
# =========================================================

# Kaggle scores
kaggle_scores = {
    "Baseline": 0.489,
    "V2": 0.513,
    "V3": 0.604,
    "V3 bis (train+val bank)": 0.602,  # ~0.602-0.603 observed after fixing bank
}

# Local validation (retrieve-from-train) metrics
local_val = {
    "V2 1-NN": {"BLEU": 20.483, "BERT_F1": 0.8801},
    "V3 1-NN": {"BLEU": 34.781, "BERT_F1": 0.9109},
    "V3 Medoid@20": {"BLEU": 37.553, "BERT_F1": 0.9162},
    "V3 bis 1-NN": {"BLEU": 36.415, "BERT_F1": 0.9137},
    "V3 bis Medoid@10": {"BLEU": 38.757, "BERT_F1": 0.9173},
}

# Medoid tuning curves (k -> metrics)
medoid_v3 = {
    1:  (34.781, 0.9109),  # 1-NN
    5:  (37.269, 0.9153),
    10: (37.349, 0.9158),
    20: (37.553, 0.9162),
}

medoid_v3bis = {
    1:  (36.415, 0.9137),  # 1-NN
    5:  (38.628, 0.9172),
    10: (38.757, 0.9173),
    20: (38.480, 0.9173),
    25: (38.153, 0.9169),
    30: (37.661, 0.9162),
    35: (36.959, 0.9152),
}

# Training logs: retrieval model V2 (20 epochs)
v2_epochs = list(range(1, 21))
v2_train_loss = [
    3.0633, 2.8874, 2.8332, 2.8000, 2.7746,
    2.7546, 2.7379, 2.7236, 2.7118, 2.7013,
    2.6925, 2.6842, 2.6784, 2.6713, 2.6651,
    2.6588, 2.6532, 2.6487, 2.6449, 2.6397
]
v2_val_loss = [
    3.5725, 3.4997, 3.4560, 3.4420, 3.4152,
    3.3966, 3.3811, 3.3736, 3.3663, 3.3556,
    3.3450, 3.3418, 3.3436, 3.3355, 3.3264,
    3.3249, 3.3252, 3.3174, 3.3196, 3.3177
]

# Training logs: retrieval model V3 (45 epochs)
v3_epochs = list(range(1, 46))
v3_train_loss = [
    3.0546, 1.7733, 1.3553, 1.1045, 0.9324,
    0.8068, 0.7046, 0.6251, 0.5670, 0.5110,
    0.4734, 0.4389, 0.4076, 0.3766, 0.3594,
    0.3364, 0.3183, 0.3017, 0.2804, 0.2770,
    0.2631, 0.2556, 0.2476, 0.2400, 0.2288,
    0.2217, 0.2207, 0.2082, 0.2045, 0.1961,
    0.1921, 0.1798, 0.1807, 0.1776, 0.1769,
    0.1727, 0.1721, 0.1634, 0.1612, 0.1590,
    0.1511, 0.1510, 0.1423, 0.1465, 0.1392
]
v3_val_loss = [
    1.5827, 1.1375, 0.9121, 0.7867, 0.6431,
    0.5781, 0.5090, 0.5001, 0.4350, 0.4025,
    0.3869, 0.3553, 0.3724, 0.3387, 0.3506,
    0.3117, 0.3176, 0.3368, 0.3256, 0.3030,
    0.2988, 0.3112, 0.2989, 0.3216, 0.3400,
    0.3583, 0.3132, 0.2623, 0.3030, 0.3190,
    0.2820, 0.2909, 0.3040, 0.2977, 0.2760,
    0.2679, 0.3116, 0.2756, 0.2526, 0.2432,
    0.2609, 0.2960, 0.2831, 0.2474, 0.2809
]
v3_scale = [
    14.68, 15.50, 16.45, 17.46, 18.50,
    19.57, 20.66, 21.75, 22.90, 24.06,
    25.19, 26.34, 27.48, 28.70, 29.86,
    31.09, 32.32, 33.49, 34.80, 36.00,
    37.13, 38.21, 39.17, 40.14, 41.20,
    42.16, 42.93, 43.81, 44.59, 45.35,
    46.14, 46.94, 47.48, 48.02, 48.14,
    48.69, 49.00, 49.57, 50.09, 50.48,
    50.87, 51.40, 51.85, 51.91, 52.30
]

# Training logs: retrieval model V3 bis (40 epochs)
v3b_epochs = list(range(1, 41))
v3b_train_loss = [
    3.9951, 2.5537, 2.0067, 1.6927, 1.4576,
    1.2865, 1.1433, 1.0315, 0.9313, 0.8525,
    0.7910, 0.7273, 0.6789, 0.6291, 0.5860,
    0.5395, 0.5062, 0.4826, 0.4560, 0.4254,
    0.3990, 0.3838, 0.3608, 0.3411, 0.3238,
    0.3131, 0.2952, 0.2780, 0.2702, 0.2577,
    0.2603, 0.2424, 0.2329, 0.2266, 0.2207,
    0.2097, 0.2090, 0.1992, 0.1896, 0.1875
]
v3b_val_loss = [
    2.3147, 1.4961, 1.2157, 1.0072, 0.8723,
    0.7899, 0.7011, 0.6385, 0.5650, 0.5145,
    0.4876, 0.4561, 0.4260, 0.3974, 0.3772,
    0.3842, 0.3509, 0.3379, 0.3234, 0.3085,
    0.3448, 0.2966, 0.2858, 0.2791, 0.2744,
    0.2723, 0.2622, 0.2770, 0.2667, 0.2561,
    0.2624, 0.2537, 0.2482, 0.2177, 0.2437,
    0.2506, 0.2367, 0.2295, 0.2454, 0.2347
]
v3b_scale = [
    14.37, 14.56, 14.86, 15.20, 15.58,
    15.97, 16.36, 16.77, 17.17, 17.57,
    17.98, 18.38, 18.79, 19.22, 19.66,
    20.10, 20.53, 20.97, 21.41, 21.86,
    22.33, 22.77, 23.23, 23.69, 24.17,
    24.63, 25.09, 25.59, 26.07, 26.54,
    27.01, 27.49, 27.99, 28.48, 28.99,
    29.50, 30.00, 30.52, 31.05, 31.57
]

# =========================================================
# Plot helpers
# =========================================================
def ensure_dir(path="retrieval_figs"):
    os.makedirs(path, exist_ok=True)
    return path

def savefig(fig, filename):
    outdir = ensure_dir()
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved", path)

# =========================================================
# 1) Kaggle scores across retrieval model versions
# =========================================================
def plot_kaggle_scores():
    names = list(kaggle_scores.keys())
    vals = [kaggle_scores[k] for k in names]

    fig = plt.figure()
    plt.bar(names, vals)
    plt.ylabel("Kaggle score")
    plt.title("Leaderboard score across retrieval model versions")
    plt.xticks(rotation=20, ha="right")
    savefig(fig, "kaggle_scores.png")

# =========================================================
# 2) Local VAL metrics across key variants
# =========================================================
def plot_local_val_metrics(metric="BLEU"):
    names = list(local_val.keys())
    vals = [local_val[k][metric] for k in names]

    fig = plt.figure()
    plt.bar(names, vals)
    plt.ylabel(metric)
    plt.title(f"Local validation ({metric}) across variants")
    plt.xticks(rotation=25, ha="right")
    savefig(fig, f"local_val_{metric.lower()}.png")

# =========================================================
# 3) Training curves (loss vs epoch)
# =========================================================
def plot_training_curve(name, epochs, train_loss, val_loss):
    fig = plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Contrastive loss")
    plt.title(f"Training curve: {name}")
    plt.legend()
    savefig(fig, f"train_curve_{name.replace(' ', '_').lower()}.png")

# =========================================================
# 4) Logit scale vs epoch (retrieval model V3 vs V3 bis)
# =========================================================
def plot_scale_comparison():
    fig = plt.figure()
    plt.plot(v3_epochs, v3_scale, label="V3 scale")
    plt.plot(v3b_epochs, v3b_scale, label="V3 bis scale")
    plt.xlabel("Epoch")
    plt.ylabel("Learned logit scale (exp(logit_scale))")
    plt.title("Learned temperature/scale over training")
    plt.legend()
    savefig(fig, "logit_scale_v3_vs_v3bis.png")

# =========================================================
# 5) Medoid-k tuning curves (choosing the best k)
# =========================================================
def plot_medoid_curve(which="v3", metric="BLEU"):
    if which == "v3":
        curve = medoid_v3
        title = "V3: Medoid@k tuning (VAL retrieve-from-TRAIN)"
        fname = f"medoid_tuning_v3_{metric.lower()}.png"
    else:
        curve = medoid_v3bis
        title = "V3 bis: Medoid@k tuning (VAL retrieve-from-TRAIN)"
        fname = f"medoid_tuning_v3bis_{metric.lower()}.png"

    ks = sorted(curve.keys())
    ys = [curve[k][0] if metric == "BLEU" else curve[k][1] for k in ks]

    fig = plt.figure()
    plt.plot(ks, ys, marker="o")
    plt.xlabel("k (top-k candidates)")
    plt.ylabel(metric)
    plt.title(title)
    savefig(fig, fname)

# =========================================================
# 6) Correlation: local best BLEU vs Kaggle score
# =========================================================
def plot_bleu_vs_kaggle():
    points = [
        ("V2", local_val["V2 1-NN"]["BLEU"], kaggle_scores["V2"]),
        ("V3", local_val["V3 Medoid@20"]["BLEU"], kaggle_scores["V3"]),
        ("V3 bis", local_val["V3 bis Medoid@10"]["BLEU"], kaggle_scores["V3 bis (train+val bank)"]),
    ]
    x = [p[1] for p in points]
    y = [p[2] for p in points]
    labels = [p[0] for p in points]

    fig = plt.figure()
    plt.scatter(x, y)
    for xi, yi, lab in zip(x, y, labels):
        plt.annotate(lab, (xi, yi))
    plt.xlabel("Local VAL BLEU (best decoding)")
    plt.ylabel("Kaggle score")
    plt.title("Local VAL BLEU vs Kaggle score (retrieval models)")
    savefig(fig, "bleu_vs_kaggle.png")

# =========================================================
# Main
# =========================================================
def main():
    plot_kaggle_scores()

    plot_local_val_metrics("BLEU")
    plot_local_val_metrics("BERT_F1")

    plot_training_curve("V2", v2_epochs, v2_train_loss, v2_val_loss)
    plot_training_curve("V3", v3_epochs, v3_train_loss, v3_val_loss)
    plot_training_curve("V3 bis", v3b_epochs, v3b_train_loss, v3b_val_loss)

    plot_scale_comparison()

    plot_medoid_curve("v3", "BLEU")
    plot_medoid_curve("v3", "BERT_F1")
    plot_medoid_curve("v3bis", "BLEU")
    plot_medoid_curve("v3bis", "BERT_F1")

    plot_bleu_vs_kaggle()

    print("\nDone. Check the figs/ folder.")

if __name__ == "__main__":
    main()
