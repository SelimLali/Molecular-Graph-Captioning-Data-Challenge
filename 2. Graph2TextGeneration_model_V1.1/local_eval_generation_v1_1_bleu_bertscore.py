import pickle
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from sacrebleu import corpus_bleu
from bert_score import score as bertscore

from transformers import AutoTokenizer, BartForConditionalGeneration
from train_g2t_bart_v1_1 import Graph2TextBART, DEVICE, VAL_GRAPHS

BEAMS = 8
MAX_NEW_TOKENS = 90
LENGTH_PENALTY = 1.0
NO_REPEAT_NGRAM = 3

def collate_graphs(batch):
    graphs, texts = zip(*batch)
    return Batch.from_data_list(list(graphs)), list(texts)

def main():
    ckpt = torch.load("g2t_v1_1_best.pt", map_location=DEVICE)
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
        batch_size=32,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=2,
        pin_memory=True,
    )

    preds, refs = [], []
    for graphs, texts in dl:
        graphs = graphs.to(DEVICE)
        gen_ids = model.generate(
            graphs,
            num_beams=BEAMS,
            max_new_tokens=MAX_NEW_TOKENS,
            length_penalty=LENGTH_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
            early_stopping=True,
        )
        out = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        preds.extend([t.strip() for t in out])
        refs.extend(texts)

    bleu = corpus_bleu(preds, [refs]).score
    _, _, F1 = bertscore(preds, refs, lang="en", model_type="roberta-base", verbose=False)

    print("Local Generative Eval v1.1 on VAL:")
    print(f"  BLEU (sacrebleu) = {bleu:.3f}")
    print(f"  BERTScore F1     = {float(F1.mean()):.4f}")

if __name__ == "__main__":
    main()
