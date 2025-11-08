# SERQ : Saliency-aware Low-rank Error Reconstruction for LLM Quantization
---
## 📌 Abstract
---
hello.



Minimal, reproducible pipeline to (1) calibrate, (2) apply GPTQ, and (3) evaluate PPL & zero-shot.
All quantization behavior is controlled via serq_quant/int_cfg.py.

0) Environment
# (optional) clean env
conda create -n serq python=3.11
conda activate serq

# core deps
pip3 install torch
pip install transformers==4.53.0
pip install tensorboard
pip install datasets
pip install accelerate
pip install lm-eval

Repository Layout (expected)
.
├─ modeling/
│  └─ modeling_llama.py
├─ serq_quant/
│  ├─ int_cfg.py         # qphase, mxfp, and all quant settings
│  ├─ int_quant.py
│  └─ observers.py
├─ utils/
│  ├─ data_utils.py
│  ├─ gptq_utils.py
│  └─ quant_utils.py
├─ models/               # stores calibration outputs / GPTQ checkpoints
├─ calibration.py
├─ run_gptq.py
├─ eval_ppl.py
└─ eval_0shot.py

Key Configs — serq_quant/int_cfg.py

qphase (execution mode):

0: baseline sim

1: calibration

777: GPTQ

3: evaluation (PPL / zero-shot)

mxfp (MXFP4 path toggle):

1: enable MXFP4 (on-the-fly quantization)

0: disable MXFP4 (e.g., use a pre-quantized GPTQ model)

All quantization behavior is governed by these arguments.

1) Calibration

Set qphase = 1 in serq_quant/int_cfg.py, then run:

python calibration.py


Artifacts (stats/caches) are written under models/ (as defined in the script).

2) Apply GPTQ

Set qphase = 777 in serq_quant/int_cfg.py, then run:

python run_gptq.py \
  --model_path ./models/Llama-2-7b-hf \
  --output_dir ./models/Llama-2-7b_gptq \
  --w_groupsize 128


Key args

--model_path: the calibrated HF model path

--output_dir: destination for the GPTQ checkpoint

--w_groupsize: GPTQ group size (e.g., 32/64/128)

3) Evaluate Perplexity (PPL)

Set qphase = 3 in serq_quant/int_cfg.py.

(A) MXFP4 path (on-the-fly)

Set mxfp default to 1

Use the calibrated (non-GPTQ) model

python eval_ppl.py \
  --model_path ./models/Llama-2-7b-hf \
  --quantize

(B) GPTQ checkpoint

Set mxfp default to 0

Use the GPTQ model

python eval_ppl.py \
  --model_path ./models/Llama-2-7b_gptq

4) Zero-Shot Evaluation

Set qphase = 3 in serq_quant/int_cfg.py.

(A) MXFP4 version

mxfp default = 1

Use the calibrated (non-GPTQ) model

python eval_0shot.py \
  --model_args ./models/Llama-2-7b-hf \
  --tasks piqa \
  --quantize

(B) GPTQ version

mxfp default = 0

Use the GPTQ model

python eval_0shot.py \
  --model_args ./models/Llama-2-7b_gptq \

  --tasks piqa
