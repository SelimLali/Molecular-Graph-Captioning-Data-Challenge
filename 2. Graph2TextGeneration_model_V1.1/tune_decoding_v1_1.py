import pickle
import itertools
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from sacrebleu import corpus_bleu
from bert_score import score as bertscore

from transformers import AutoTokenizer, BartForConditionalGeneration
from train_g2t_bart_v1_1 import Graph2TextBART, DEVICE, VAL_GRAPHS

CKPT_PATH = "g2t_v1_1_best.pt"

# Small grid search to fine-tune the parameters (but keep runtime reasonable)
BEAMS_LIST = [5, 8, 10, 12]
MAX_NEW_LIST = [80, 90, 110, 130]
LENP_LIST = [0.8, 0.9, 1.0, 1.1]
NO_REPEAT_LIST = [3, 4]

BATCH_SIZE = 32  # generation batch

def collate_graphs(batch):
    graphs, texts = zip(*batch)
    return Batch.from_data_list(list(graphs)), list(texts)

@torch.no_grad()
def run_one(model, tokenizer, dl, beams, max_new, lenp, no_rep):
    preds, refs = [], []
    for graphs, texts in dl:
        graphs = graphs.to(DEVICE)
        gen_ids = model.generate(
            graphs,
            num_beams=beams,
            max_new_tokens=max_new,
            length_penalty=lenp,
            no_repeat_ngram_size=no_rep,
            early_stopping=True,
        )
        out = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        preds.extend([t.strip() for t in out])
        refs.extend(texts)

    bleu = corpus_bleu(preds, [refs]).score
    _, _, F1 = bertscore(preds, refs, lang="en", model_type="roberta-base", verbose=False)
    return float(bleu), float(F1.mean())

def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model_name = ckpt["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    bart = BartForConditionalGeneration.from_pretrained(model_name)
    bart.resize_token_embeddings(len(tokenizer))

    model = Graph2TextBART(bart=bart).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    with open(VAL_GRAPHS, "rb") as f:
        val_graphs = pickle.load(f)

    dl = DataLoader(
        [(g, g.description) for g in val_graphs],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=2,
        pin_memory=True,
    )

    best = None
    for beams, max_new, lenp, no_rep in itertools.product(
        BEAMS_LIST, MAX_NEW_LIST, LENP_LIST, NO_REPEAT_LIST
    ):
        bleu, f1 = run_one(model, tokenizer, dl, beams, max_new, lenp, no_rep)
    
        score = 0.5 * (bleu / 100.0) + 0.5 * f1

        print(f"beams={beams:2d} max_new={max_new:3d} lenp={lenp:.1f} nrep={no_rep} | "
              f"BLEU={bleu:.3f} F1={f1:.4f} | proxy={score:.4f}")

        if best is None or score > best[0]:
            best = (score, beams, max_new, lenp, no_rep, bleu, f1)

    print("\nBEST:")
    print(best)

if __name__ == "__main__":
    main()
