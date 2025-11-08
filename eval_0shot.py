import argparse
import json
import os
from typing import Any, Dict

from modeling import LlamaForCausalLM
from transformers import AutoTokenizer

from lm_eval.evaluator import simple_evaluate
from lm_eval.utils import handle_non_serializable

import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Run lm-evaluation-harness simple_evaluate (text-only)")
    parser.add_argument(
        "--model",
        type=str,
        default="hf",
        help="Model key (e.g., hf-causal, hf-auto, vllm, etc.)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help=(
            "Model argument string. Example: "
            "'pretrained=/path/to/model,device_map=auto,dtype=float16' or "
            "'pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16'"
        ),
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="piqa",
        help="Comma-separated task list or group name. Example: 'wikitext,hellaswag'",
    )
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default="1")
    parser.add_argument("--max_batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda:0, cuda, cpu, etc.")
    parser.add_argument("--limit", type=float, default=None, help="Sample limit per task (for debugging)")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Result save path (file or directory). Saves as JSON when specified",
    )
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--log_samples", action="store_true", help="Log sample-level results")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--numpy_random_seed", type=int, default=1234)
    parser.add_argument("--torch_random_seed", type=int, default=1234)
    parser.add_argument("--fewshot_random_seed", type=int, default=1234)
    return parser.parse_known_args()


def main():
    args, _ = parse_args()

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto"
    )

    if args.quantize:
        for module in model.modules():
            if (type(module).__name__ == "SVDLinear"):
                module.quantA.update_smooth_factor()
                module.order = torch.argsort(module.quantA.smooth_factor.squeeze(dim=(0,1)), descending=True)
                module.serq_quant()  

    model.eval()
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        trust_remote_code=True
    )

    from lm_eval.models.huggingface import HFLM
    hf_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        trust_remote_code=True
    )   

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    outputs = simple_evaluate(
        model=hf_model,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )

    results = outputs.get("results", {})

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True) if os.path.dirname(args.output_path) else None
        with open(args.output_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=_json_default)
        print(f"Saved results to {args.output_path}")
    else:
        metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
        print(json.dumps(metric_vals, ensure_ascii=False, indent=2, default=handle_non_serializable))


if __name__ == "__main__":
    main()


