import pickle
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from transformers import AutoTokenizer, BartForConditionalGeneration
from train_g2t_bart_v1_1 import Graph2TextBART, DEVICE

TEST_GRAPHS = "data/test_graphs.pkl"
OUT_CSV = "submission_generation_v1_1.csv"

#fine-tuned parameters
BEAMS = 5
MAX_NEW_TOKENS = 130
LENGTH_PENALTY = 1.1
NO_REPEAT_NGRAM = 4

def collate_graphs_only(batch):
    return Batch.from_data_list(batch)

def main():
    ckpt = torch.load("g2t_v1_1_best.pt", map_location=DEVICE)
    model_name = ckpt["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    bart = BartForConditionalGeneration.from_pretrained(model_name)
    bart.resize_token_embeddings(len(tokenizer))

    model = Graph2TextBART(bart=bart).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    with open(TEST_GRAPHS, "rb") as f:
        test_graphs = pickle.load(f)

    test_ids = [str(g.id) for g in test_graphs]

    dl = DataLoader(
        test_graphs,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_graphs_only,
        num_workers=2,
        pin_memory=True,
    )

    preds = []
    for bg in dl:
        bg = bg.to(DEVICE)
        gen_ids = model.generate(
            bg,
            num_beams=BEAMS,
            max_new_tokens=MAX_NEW_TOKENS,
            length_penalty=LENGTH_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
            early_stopping=True,
        )
        out = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        preds.extend([t.strip() for t in out])

    df = pd.DataFrame({"ID": test_ids, "description": preds})
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")

if __name__ == "__main__":
    main()
