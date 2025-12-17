
# Graph-to-Text Generation V1.1 (BART-based)

Graph-to-text generation model for the **Molecular Graph Captioning** challenge.
This model moves beyond simple retrieval and directly **generates** captions from molecular graphs.

V1.1 is a BART-based model that achieved my best Kaggle score among pure generation approaches
(after decoding tuning).

---

## 1. Overview

The model has two main components:

1. **Graph encoder**
   - Encodes the molecular graph (atoms + bonds) into a compact representation.
   - Uses a GNN over node/edge embeddings and then maps the result to a fixed set of **latent tokens**
     that summarize the graph.

2. **BART decoder (sequence-to-sequence)**  
   - Uses the latent graph tokens as encoder outputs for cross-attention.
   - Generates the caption autoregressively, token by token.
   - Trained with standard cross-entropy loss on the ground-truth description.

During inference, BART is used with **beam search**, **length penalty**, and a **no-repeat \(n\)-gram constraint**.
We tuned these decoding hyperparameters to maximize the combined BLEU + BERTScore proxy on the validation set.

---

## 2. Installation

From inside `2. Graph2TextGeneration_model_V1.1/`:

```bash
pip install -r requirements.txt
```

This installs:
- `torch`, `torch_geometric`
- `transformers` (BART + RoBERTa)
- `sacrebleu`, `bert-score`
- `pandas`, `numpy`, `tqdm`, and other utilities.

---

## 3. Data Setup

Place the preprocessed graph files under `data/`:

```text
2. Graph2TextGeneration_model_V1.1/
├── data/
│   ├── train_graphs.pkl
│   ├── validation_graphs.pkl
│   └── test_graphs.pkl
└── ...
```

You can inspect the data with:

```bash
python inspect_graph_data.py
```

This prints basic statistics and confirms that `train` and `validation` splits contain text descriptions,
while `test` does not.

---

## 4. Training the BART Graph-to-Text Model

To train the V1.1 model:

```bash
python train_g2t_bart_v1_1.py
```

What the script does:

- Loads the training and validation graphs from `data/`.
- Builds a graph encoder (GNN over atoms/bonds) that outputs a sequence of latent tokens.
- Instantiates a BART model (encoder–decoder) from Hugging Face.
- Connects the graph encoder to BART so that the latent tokens become the encoder outputs.
- Trains the model with teacher forcing, minimizing the token-level cross-entropy loss on the target captions.
- Performs validation after each epoch.
- Saves the **best** checkpoint (according to validation loss) as:

```text
g2t_v1_1_best.pt
```

Make sure a GPU is available and that the script is configured with `DEVICE=cuda` for efficient training.

---

## 5. Local Evaluation (Validation BLEU / BERTScore)

To evaluate the generation quality on the validation set:

```bash
python local_eval_generation_v1_1_bleu_bertscore.py
```

This script:

- Loads `g2t_v1_1_best.pt`.
- Generates captions for all validation graphs using a default decoding configuration (beam search, etc.).
- Computes:
  - BLEU (using `sacrebleu`)
  - BERTScore F1 (using RoBERTa-base)
- Prints the scores to the console.

These metrics are used to monitor overfitting and to compare different model variants.

---

## 6. Decoding Hyperparameter Tuning

Decoding greatly affects BLEU/BERTScore. To search over decoding settings for V1.1:

```bash
python tune_decoding_v1_1.py
```

This script runs a grid search over:
- `num_beams` (e.g. 5, 8)
- `max_new_tokens` (e.g. 80, 90, 110, 130)
- `length_penalty` (e.g. 0.8, 0.9, 1.0, 1.1)
- `no_repeat_ngram_size` (e.g. 3, 4)

For each configuration, it:

1. Generates validation captions with the loaded `g2t_v1_1_best.pt` model.
2. Computes BLEU and BERTScore.
3. Reports a combined **proxy score** (e.g. average of normalized BLEU and BERTScore) to rank configurations.

In our experiments, the best configuration (on validation) was of the form:
\[
\texttt{num\_beams}=5,\quad
\texttt{max\_new\_tokens}=130,\quad
\texttt{length\_penalty}=1.1,\quad
\texttt{no\_repeat\_ngram\_size}=4.
\]

This configuration was then used for the final Kaggle submission.

---

## 7. Generate Kaggle Submission

Once the best checkpoint and decoding configuration have been identified, generate test captions and build
the submission file:

```bash
python make_submission_generation_v1_1.py
```

This script:

- Loads `g2t_v1_1_best.pt`.
- Uses the chosen decoding parameters (for example from the tuning step).
- Generates captions for all graphs in `data/test_graphs.pkl`.
- Saves a CSV file:

```text
submission_generation_v1_1_after_fine_tune.csv
```

with columns:

- `ID`: the graph/molecule identifier
- `description`: the generated caption

This file can be directly submitted to Kaggle.

---

## 8. Notes and Tips

- BART-based generation is **sensitive** to decoding settings. Always validate with BLEU and BERTScore, not only
  log-likelihood.
- Keep an eye on `max_new_tokens`: too small values truncate captions and hurt BLEU, while too large values may
  lead to repetitive outputs if `no_repeat_ngram_size` is not used.
- If GPU memory allows, you may experiment with more latent tokens or larger batch sizes in `train_g2t_bart_v1_1.py`.
- For deeper analysis and comparison with other generation models (V0, V1, V1.2, V2), use the plotting script in
  the project root (`2. plot_metrics_Graph2Text generation.py`), which reads aggregate metrics from all runs.
