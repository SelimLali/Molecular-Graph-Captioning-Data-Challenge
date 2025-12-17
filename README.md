
# Molecular Graph Captioning – Data Challenge (ALTEGRAD 2025–26)

This repository contains my work for the **ALTEGRAD 2025–26 Kaggle data challenge**, *Molecular Graph Captioning*.
The goal is to learn models that take a **molecular graph** (atoms + bonds) as input and produce a **natural-language
description** of the molecule.

The project is organized around three main model families:

1. **Baseline retrieval model** (provided by the instructors)
2. **Improved retrieval model (V3)** – my best-performing *retrieval* system
3. **Graph-to-Text generation model (V1.1)** – my best-performing *generative* system

On Kaggle’s hidden test set, the best scores achieved were roughly:
- **Baseline retrieval:** ~0.49
- **Retrieval V3:** ~0.60
- **Graph2Text V1.1 (BART, tuned decoding):** ~0.62
(these scores correspond to Kaggle’s private metric combining BLEU-4 and BERTScore).


### Quick Results Summary

The table below summarizes the main models kept in this repository.

| Model                             | Paradigm          | Validation BLEU (VAL) | Validation BERTScore F1 (VAL) | Kaggle Test Score |
|-----------------------------------|-------------------|------------------------|-------------------------------|-------------------|
| Baseline retrieval (teacher code) | Retrieval (1-NN)  | –                      | –                             | **0.489**         |
| Retrieval V3 (GINE + RoBERTa CLIP)| Retrieval (medoid)| ≈37.6 (Medoid@20)      | ≈0.916                        | **0.604**         |
| Graph2Text V1.1 (BART)           | Generative        | 37.29 (tuned decoding) | 0.9188                        | **0.619**         |

The validation BLEU/BERTScore values are measured on the validation set using local evaluation scripts;
the Kaggle score is the challenge’s hidden test metric (a composite of BLEU-4 and BERTScore).


---

## 1. Problem Overview

Each molecule is given as a **graph**:
- nodes = atoms (with 9 categorical features: atomic number, chirality, degree, formal charge, attached hydrogens, radical electrons, hybridization, aromatic flag, ring flag)
- edges = bonds (with 3 categorical features: bond type, stereo, conjugation)

The task is to learn a model that, given only the graph, produces a caption similar to the expert-written description,
evaluated by text-generation metrics (BLEU-4 and BERTScore with RoBERTa-base) on a hidden test set.

There are three splits:
- **train**: ~31k graphs, with descriptions
- **validation**: 1k graphs, with descriptions
- **test**: 1k graphs, **without** descriptions (labels hidden on Kaggle)

The graphs are stored as PyTorch Geometric `Data` objects in pickle files:
- `train_graphs.pkl`
- `validation_graphs.pkl`
- `test_graphs.pkl`

---

## 2. Dataset and Utilities

All models use the same preprocessed data and utilities:

- The `data/` folder for each model contains:
  - `train_graphs.pkl`, `validation_graphs.pkl`, `test_graphs.pkl`
- The helper script **`data_utils.py`** provides:
  - `PreprocessedGraphDataset`: wraps the `.pkl` files as a `torch.utils.data.Dataset`
  - `collate_fn`: builds batched `torch_geometric.data.Batch` objects
  - Feature maps (`x_map`, `e_map`) that define the categorical → index mapping
- The script **`inspect_graph_data.py`** is used to quickly inspect graph statistics (number of nodes, edges,
  feature shapes, presence of descriptions, etc.).

> **Important:** All three model folders assume the same data format and file names inside their local `data/` subfolder.

---

## 3. Baseline Retrieval Model (Teacher Code)

**Folder:** `0. data_baseline`

### 3.1. High-level idea

The baseline is a **joint-embedding retrieval** system:

- A **GCN encoder** maps each molecular graph to a fixed-size vector \( v_{\text{graph}} \).
- A **pre-trained BERT** (bert-base-uncased) encodes each caption into a vector \( v_{\text{text}} \)
  (CLS token).
- During training, the GCN is trained with an **MSE loss** to make \( v_{\text{graph}} \approx v_{\text{text}} \)
  for matching graph–caption pairs.
- At test time, the model does **nearest-neighbor retrieval**: for each test graph, it finds the
  **most similar caption** among the training captions (cosine similarity in the shared space).

This does not generate new text; it only **retrieves** one of the existing training descriptions.

### 3.2. How to run the baseline

Inside `0. data_baseline` (already provided by the instructors), follow the professor’s README:

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data** (graphs). Place your `.pkl` files under `data/`:
   - `data/train_graphs.pkl`
   - `data/validation_graphs.pkl`
   - `data/test_graphs.pkl`

3. **Generate BERT embeddings for text descriptions**

   ```bash
   python generate_description_embeddings.py
   ```

   This produces:
   - `data/train_embeddings.csv`
   - `data/validation_embeddings.csv`

4. **Train the GCN retriever**

   ```bash
   python train_gcn.py
   ```

   This trains a simple GCN (no edge features, a single learnable node embedding) and saves:
   - `model_checkpoint.pt`

5. **Run retrieval on the test set**

   ```bash
   python retrieval_answer.py
   ```

   This computes graph embeddings for all test molecules, retrieves the nearest caption from the train set, and
   produces:
   - `test_retrieved_descriptions.csv` (submission-style file with columns `ID,description`).

---

## 4. Improved Retrieval Model V3

**Folder:** `1. Retrieval_model_V3`

This is my **strongest retrieval-only model**, significantly improving over the baseline.
Key differences compared to the teacher baseline:

- Uses **RoBERTa-base** to encode captions (better match with BERTScore).
- Uses **GINEConv** (edge-aware GNN) and properly embeds:
  - all 9 atom categorical features via `nn.Embedding` + projection
  - all 3 bond categorical features via `nn.Embedding` inside the GNN.
- Uses a **contrastive CLIP-style loss** with in-batch negatives instead of MSE.
- Uses a **learnable temperature** parameter (\( \tau \)) in the contrastive softmax.
- Adds **attention-based graph pooling** instead of pure sum/mean.
- Trains a **small projection head on text embeddings** so both modalities are mapped into
  a shared space tailored to the retrieval task.
- At inference, uses **top-\(k\) medoid selection** on the retrieval bank to choose a caption that is both
  semantically relevant and stylistically “typical” (this improves BLEU).

### 4.1. Running Retrieval V3

From inside `1. Retrieval_model_V3/`:

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**

   Place the same preprocessed graph files into `data/`:
   - `data/train_graphs.pkl`
   - `data/validation_graphs.pkl`
   - `data/test_graphs.pkl`

3. **Generate RoBERTa text embeddings**

   ```bash
   python generate_text_embeddings_roberta.py
   ```

   This will compute mean-pooled RoBERTa-base embeddings for all train/validation captions and save them to
   `.pt` files in `data/` (e.g. `train_textemb_roberta-base_mean.pt`, `validation_textemb_roberta-base_mean.pt`).

4. **Train the retrieval model**

   ```bash
   python train_retrieval_clip_gine.py
   ```

   This trains the GINE-based graph encoder and the small text projection head using a CLIP-style
   contrastive loss with in-batch negatives, saving:
   - `retrieval_v3_best.pt`

5. **Local evaluation (validation set)**

   ```bash
   python local_eval_retrieval_bleu_bertscore.py
   ```

   This:
   - Uses the trained model to retrieve captions for validation graphs from the **training** bank
   - Computes BLEU and BERTScore against the ground truth validation descriptions
   - Optionally tries different `k` values for medoid selection (e.g. \(k = 5, 10, 20\)).

6. **Make Kaggle submission**

   ```bash
   python make_submission_retrieval.py
   ```

   This:
   - Builds a retrieval bank from `train` (+ optionally `validation`) captions
   - Retrieves captions for each test graph (with medoid strategy)
   - Writes `submission_retrieval_v3.csv` in the correct Kaggle format (`ID,description`).

7. **Plot metrics** (optional)

   At the root of the project, the script `1. plot_metrics_Retrieval_models.py` uses logged metrics to generate
   comparative plots (stored in `1. plots_Retrieval_models/`), visualizing training curves and retrieval scores
   across retrieval variants.

---

## 5. Graph-to-Text Generation Model V1.1 (BART-based)

**Folder:** `2. Graph2TextGeneration_model_V1.1`

The third main model family moves beyond retrieval and performs **true generation**. V1.1 is a BART-based
graph-to-text model that achieved my best Kaggle score among generative variants.

### 5.1. High-level architecture

- A **graph encoder** maps the molecular graph to a compact representation (e.g., a set of latent pseudo-tokens).
- A **BART decoder** (sequence-to-sequence transformer) is conditioned on these latent tokens via cross-attention.
- The model is trained with a standard **autoregressive cross-entropy loss** over the caption tokens.

In practice, the architecture follows this pattern:

1. **Graph encoder**
   - Uses a GNN (e.g., GINE/GCN) over atom/bond embeddings, with global pooling or learned latent tokens.
   - Projects the graph representation to a fixed number of **latent tokens**.
2. **BART decoder (conditional)**
   - The latent tokens serve as encoder outputs for BART’s cross-attention.
   - The decoder generates the caption autoregressively.
3. **Training**
   - Teacher forcing: the ground-truth caption is fed to the decoder shifted by one token.
   - Loss is token-level cross-entropy on the BART vocabulary.
4. **Inference**
   - Given a test graph, the encoder produces latent tokens.
   - BART decodes a caption using **beam search**, length penalty, and a no-repeat \(n\)-gram constraint.

### 5.2. Running Graph2Text V1.1

From inside `2. Graph2TextGeneration_model_V1.1/`:

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**

   Copy the graph files into `data/` as before:
   - `data/train_graphs.pkl`
   - `data/validation_graphs.pkl`
   - `data/test_graphs.pkl`

3. **Inspect graphs (optional)**

   ```bash
   python inspect_graph_data.py
   ```

4. **Train the BART graph-to-text model**

   ```bash
   python train_g2t_bart_v1_1.py
   ```

   This script:
   - Loads train/validation graphs
   - Builds the graph encoder + BART decoder
   - Trains for multiple epochs on GPU (if available)
   - Saves the best checkpoint according to validation loss as:
     - `g2t_v1_1_best.pt`

5. **Local evaluation (validation BLEU/BERTScore)**

   ```bash
   python local_eval_generation_v1_1_bleu_bertscore.py
   ```

   This script:
   - Loads `g2t_v1_1_best.pt`
   - Generates captions for the validation set
   - Computes BLEU (sacreBLEU) and BERTScore (RoBERTa-base) against the ground-truth captions.

6. **Decoding hyperparameter tuning**

   ```bash
   python tune_decoding_v1_1.py
   ```

   This script performs a grid search over decoding parameters such as:
   - `num_beams`
   - `max_new_tokens`
   - `length_penalty`
   - `no_repeat_ngram_size`

   It reports validation BLEU/BERTScore and a combined proxy score. The best configuration (for example):
   \[
   \texttt{num\_beams}=5,\quad
   \texttt{max\_new\_tokens}=130,\quad
   \texttt{length\_penalty}=1.1,\quad
   \texttt{no\_repeat\_ngram\_size}=4
   \]
   is then used for the final Kaggle submission.

7. **Make Kaggle submission**

   ```bash
   python make_submission_generation_v1_1.py
   ```

   This:
   - Loads the best BART checkpoint and best decoding config
   - Generates captions for all test graphs
   - Writes `submission_generation_v1_1_after_fine_tune.csv` in the correct Kaggle format.


8. **Plot metrics** (optional)

   At the root of the project, the script `2. plot_metrics_Graph2Text generation.py` produces summary plots in
   `2. plots_generation_models/`, including:
   - Training/validation loss curves
   - Validation BLEU/BERTScore per model
   - Kaggle scores comparison for the different generation variants (V0, V1, V1.1, V1.2, V2).

---

## 6. Old Trials and Experimental Attempts

**Folder:** `3. old_trials_attempts`

This folder collects various older experiments and prototypes (e.g., earlier GPT-2-only runs, alternative
architectures and hyperparameters) that were superseded by the final V3 retrieval and V1.1 generation models.
They are kept mainly for reference and ablation analysis.

---

## 7. Reproducibility Notes

- All models assume **Python 3.9+**, **PyTorch**, **PyTorch Geometric**, and **Hugging Face Transformers**.
- Each model folder ships its own `requirements.txt` capturing the main dependencies.
- To reproduce Kaggle submissions:
  1. Install the dependencies for the corresponding model.
  2. Make sure the `data/` subfolder contains the three `.pkl` files.
  3. Follow the exact sequence of scripts documented above for that model (train → local eval → submission).
- When using GPU (e.g., A100 on Colab), set the `DEVICE` or `--device` parameter to `cuda` in the training scripts.
