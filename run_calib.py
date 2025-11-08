# Code Based on SmoothQuant
import torch
import torch.nn as nn
import sys

from modeling import LlamaForCausalLM
from transformers import AutoTokenizer
import tqdm
from datasets import load_dataset
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf")

args, _ = parser.parse_known_args()
model_path = args.model_path


class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // 2048
        for i in tqdm.tqdm(range(n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (n_samples * 2048))


tokenizer = AutoTokenizer.from_pretrained(model_path)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
evaluator = Evaluator(dataset, tokenizer, "cuda", n_samples=128)

model = LlamaForCausalLM.from_pretrained(
    model_path, device_map="auto"
)

ppl = evaluator.evaluate(model)
print(f"Perplexity: {ppl}")

save_path = f"./models/{model_path.split('/')[-1]}"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"model saved to {save_path}")