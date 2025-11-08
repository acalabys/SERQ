import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from modeling import LlamaForCausalLM

from utils.gptq_utils import gptq_fwrd, rtn_fwrd
import utils.quant_utils
from utils.data_utils import get_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Custom GPTQ execution script")
    parser.add_argument(
        "--model_path", type=str, default="/home/yspark/2025_winter/llm_SVD/models/Llama2_7B_1_v2",
        help="HuggingFace checkpoint name or local path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="/home/yspark/2025_winter/llm_SVD/models/Llama2_7B_gptq_v2",
        help="Directory to save quantized model"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device for quantization"
    )
    parser.add_argument(
        "--nsamples", type=int, default=128,
        help="Number of samples for calibration"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Calibration batch size"
    )
    parser.add_argument(
        "--w_bits", type=int, default=4, choices=[2, 3, 4, 8, 16],
        help="Number of bits for layer weight quantization (LM head is automatically 16-bit)"
    )
    parser.add_argument(
        "--w_groupsize", type=int, default=-1,
        help="Group size for quantization (-1 for RTN only)"
    )
    parser.add_argument(
        "--percdamp", type=float, default=0.01,
        help="Dampening ratio for Hessian diagonal approximation"
    )
    parser.add_argument(
        "--w_asym", action="store_true",
        help="Use asymmetric quantization (default: symmetric)"
    )
    parser.add_argument(
        "--w_clip", type=float, default=0.0,
        help="Clipping threshold for MSE-based scale optimization (0 for disabled)"
    )
    parser.add_argument(
        "--int8_down_proj", action="store_true",
        help="Quantize only FFN down_proj to 8-bit and use w_bits for others"
    )
    parser.add_argument(
        "--act_order", action="store_true",
        help="Whether to use Hessian diagonal ordering (actorder)"
    )
    parser.add_argument(
        "--use_rtn", action="store_true",
        help="Use RTN (Round-to-Nearest) quantization instead of Hessian"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for data splitting"
    )
    return parser.parse_known_args()


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from modeling.modeling_llama import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


def get_qwen(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from modeling.modeling_qwen import Qwen2ForCausalLM
    model = Qwen2ForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


def main():
    args, _ = parse_args()

    # 1) Load model & tokenizer
    logger.info("1) Loading model: %s" % args.model_path)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if "qwen" in args.model_path:
        model = get_qwen(args.model_path)
    else:
        model = get_llama(args.model_path)

    # Keep entire model on CPU, move quantization target layers to GPU only when needed
    model.to("cpu")
    model.eval()

    # Iterate through all modules and update Quantizer instance parameters
    for module in model.modules():
        if (type(module).__name__ == "SVDLinear"):
            module.quantA.update_smooth_factor()
            module.order = torch.argsort(module.quantA.smooth_factor.squeeze(dim=(0,1)), descending=True)
            module.init_lora()
            
    def scale_input_pre_hook(module, inputs):
        x = inputs[0]
        x = x / module.quantA.smooth_factor
        x = x[:,:,module.order]
        return (x,) + inputs[1:]
    for module in model.modules():
        if (type(module).__name__ == "SVDLinear"):
            module.register_forward_pre_hook(scale_input_pre_hook)
    
    # 2) Prepare calibration data (e.g., WikiText2 validation)
    logger.info("2) Loading calibration data (WikiText2)")
    dataloader, testloader = get_loaders(
        "wikitext2", nsamples=args.nsamples, seed=args.seed, model=args.model_path, seqlen=model.seqlen
    )

    # 3) Set quantization parameters
    logger.info("3) Quantization parameters: w_bits=%d, groupsize=%d, percdamp=%.4f" %
                (args.w_bits, args.w_groupsize, args.percdamp))

    # 4) Perform actual quantization
    if args.use_rtn:
        logger.info("Starting RTN quantization")
        quantizers = rtn_fwrd(model, device, args)
    else:
        logger.info("Starting GPTQ quantization")
        quantizers = gptq_fwrd(model, dataloader, device, args)
    # quantizers: { "model.layers.0.self_attn.q_proj.module": quantizer, ... } dictionary

    # 5) Save quantized model
    logger.info("5) Saving quantized model: %s" % args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    # Model weights are already quantized, so just save_pretrained
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("All tasks completed")


if __name__ == "__main__":
    main()
