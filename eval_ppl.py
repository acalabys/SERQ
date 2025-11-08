import torch
import torch.nn as nn

from serq_quant import SVDLinear
from modeling import LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM, Gemma3ForCausalLM, PhiForCausalLM, Phi3ForCausalLM
from transformers import AutoTokenizer
import tqdm
import os

from datasets import load_dataset
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--n_samples", type=int, default=None)
parser.add_argument("--quantize", action="store_true")

args, _ = parser.parse_known_args()
model_path = args.model_path
n_samples = args.n_samples


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
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
evaluator = Evaluator(dataset, tokenizer, "cuda", n_samples=n_samples)

if "llama" in model_path.lower() or "deepseek" in model_path.lower():
    model = LlamaForCausalLM.from_pretrained(
        model_path, device_map="auto"
    )
elif "Qwen" in model_path:
    model = Qwen2ForCausalLM.from_pretrained(
        model_path, device_map="auto"
    )
elif "mistral" in model_path:
    model = MistralForCausalLM.from_pretrained(
        model_path, device_map="auto"
    )
elif "gemma" in model_path:
    model = Gemma3ForCausalLM.from_pretrained(
        model_path, device_map="auto"
    )
elif "phi" in model_path:
    model = PhiForCausalLM.from_pretrained(
        model_path, device_map="auto"
    )
elif "Phi-4-mini" in model_path:
    model = Phi3ForCausalLM.from_pretrained(
        model_path, device_map="auto"
    )
else:
    raise ValueError(f"Unsupported model: {model_path}")

if args.quantize:
    for module in model.modules():
        if (isinstance(module, SVDLinear)):
            module.quantA.update_smooth_factor()
            module.order = torch.argsort(module.quantA.smooth_factor.squeeze(dim=(0,1)), descending=True)
            module.serq_quant()   

ppl = evaluator.evaluate(model)
print(f"Perplexity: {ppl}")