import torch
import json
import os
import time
import statistics
import argparse
import re
from datetime import datetime
from typing import Dict, Any, List

# =============================================================================
# All supported models — pass via --model flag
# =============================================================================

MODELS = {
    # --- Base Models ---
    "ministral-3b":   "mistralai/Ministral-3-3B-Instruct-2512",
    "ministral-8b":   "mistralai/Ministral-3-8B-Instruct-2512",
    "ministral-14b":  "mistralai/Ministral-3-14B-Instruct-2512",
    "qwen3-4b":       "Qwen/Qwen3-4B-Instruct-2507",
    "granite-8b":     "ibm-granite/granite-3.1-8b-instruct",
    "llama3-8b":      "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral-7b":     "mistralai/Mistral-7B-Instruct-v0.3",
    "phi-4":          "microsoft/phi-4",

    # --- Fine-tuned LoRA adapters ---
    "ministral-14b-lora": "output/ministral14b-bf16-lora_new/final_lora",
    "ministral-8b-lora":  "output/ministral8b-bf16-lora/final_lora",
    "ministral-3b-lora":  "output/ministral3b-bf16-lora/final_lora",
    "qwen3-4b-lora":      "output/qwen3-4b-instruct-bf16-lora/final_lora",
    "granite-8b-lora":    "output/granite-3.1-8b-json-extractor-bf16-fixed/final_lora",
    "llama3-8b-lora":     "output/llama3-8b-bf16-lora/final_lora",
    "mistral-7b-lora":    "output/mistral7b-bf16-lora/final_lora",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

if DEVICE == "cpu":
    print("WARNING: Running on CPU – will be very slow")
else:
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")

try:
    import flash_attn
    HAS_FLASH = torch.cuda.is_available()
except ImportError:
    HAS_FLASH = False

IGNORE_KEYS = {'case', 'case_number', 'skadekommune'}


# =============================================================================
# Detect model type from config
# =============================================================================

def detect_model_type(model_id: str) -> str:
    """Returns 'mistral3_vlm', 'mistral3_vlm_fp8', or 'causal_lm'."""
    config_path = os.path.join(model_id, "config.json") if os.path.isdir(model_id) else None

    model_type = None

    # For LoRA adapters, check the base model's config
    adapter_config_path = os.path.join(model_id, "adapter_config.json") if os.path.isdir(model_id) else None
    if adapter_config_path and os.path.exists(adapter_config_path):
        with open(adapter_config_path, 'r') as f:
            base_model_path = json.load(f).get('base_model_name_or_path', '')
        # Try to read the base model's config
        base_config_path = os.path.join(base_model_path, "config.json")
        if os.path.exists(base_config_path):
            with open(base_config_path, 'r') as f:
                model_type = json.load(f).get('model_type', '')
        else:
            try:
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
                model_type = getattr(cfg, 'model_type', '')
            except Exception:
                pass
        if model_type == 'mistral3':
            if 'bf16' in base_model_path.lower() or 'bf16' in model_id.lower():
                return 'mistral3_vlm'
            else:
                return 'mistral3_vlm_fp8'

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            model_type = json.load(f).get('model_type', '')
    else:
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            model_type = getattr(cfg, 'model_type', '')
        except Exception:
            pass

    if model_type == 'mistral3':
        if 'bf16' in model_id.lower():
            return 'mistral3_vlm'
        else:
            return 'mistral3_vlm_fp8'

    return 'causal_lm'


# =============================================================================
# Model + tokenizer loading
# =============================================================================

def load_model_and_tokenizer(model_id: str):
    print(f"\nLoading {model_id} ...")
    attn_impl = "flash_attention_2" if HAS_FLASH else "eager"
    print(f"Attention: {attn_impl}")

    is_lora = os.path.isdir(model_id) and os.path.exists(
        os.path.join(model_id, "adapter_config.json")
    )

    mtype = detect_model_type(model_id)
    print(f"Detected type: {mtype} | LoRA: {is_lora}")

    if mtype.startswith('mistral3_vlm'):
        # ---------------------------------------------------------------
        # Mistral3 VLM checkpoint (BF16 or FP8)
        # ---------------------------------------------------------------
        from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

        load_kwargs = dict(
            device_map={"": DEVICE},
            dtype=DTYPE,
            attn_implementation=attn_impl,
            trust_remote_code=False,
        )

        if mtype == 'mistral3_vlm_fp8':
            from transformers import FineGrainedFP8Config
            load_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)

        if is_lora:
            from peft import PeftModel
            # Read base model path from adapter config
            adapter_cfg_path = os.path.join(model_id, "adapter_config.json")
            with open(adapter_cfg_path, 'r') as f:
                base_model_path = json.load(f).get('base_model_name_or_path', '')
            print(f"LoRA base model: {base_model_path}")
            # Load the base model first with Mistral3ForConditionalGeneration
            base_model = Mistral3ForConditionalGeneration.from_pretrained(
                base_model_path, **load_kwargs
            )
            # Apply the LoRA adapter on top (no tie_weights for LoRA)
            model = PeftModel.from_pretrained(base_model, model_id)
            tokenizer = MistralCommonBackend.from_pretrained(base_model_path)
        else:
            model = Mistral3ForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
            model.tie_weights()
            tokenizer = MistralCommonBackend.from_pretrained(model_id)

        print(f"Loaded as Mistral3 VLM ({'FP8→BF16' if 'fp8' in mtype else 'BF16'})")

    else:
        # ---------------------------------------------------------------
        # Standard CausalLM (Qwen, Llama, Granite, Mistral-7B, Phi, etc.)
        # ---------------------------------------------------------------
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if is_lora:
            from peft import AutoPeftModelForCausalLM
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_id,
                device_map={"": DEVICE},
                dtype=DTYPE,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            )
            base_id = (
                model.peft_config["default"].base_model_name_or_path
                if hasattr(model, "peft_config") else model_id
            )
            tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map={"": DEVICE},
                dtype=DTYPE,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        print("Loaded as standard CausalLM")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if DEVICE == "cuda":
        vram_alloc = torch.cuda.memory_allocated() / 1e9
        vram_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"VRAM: {vram_alloc:.1f} GB allocated, {vram_reserved:.1f} GB reserved\n")
    return model, tokenizer


# =============================================================================
# Inference
# =============================================================================

def generate_response(model, tokenizer, prompt_json: Dict, case_json: Dict) -> Any:
    combined = (
        f"Please follow the instructions in the following JSON object:\n"
        f"{json.dumps(prompt_json, indent=2, ensure_ascii=False)}\n\n"
        f"Apply to this data:\n"
        f"{json.dumps(case_json, indent=2, ensure_ascii=False)}\n\n"
        f"Answer with ONLY raw JSON (object or array), no markdown, no explanation."
    )
    messages = [{"role": "user", "content": combined}]

    # Qwen3 supports enable_thinking parameter in apply_chat_template
    chat_kwargs = dict(
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    # Disable thinking mode for Qwen3 (avoids burning tokens on <think>...</think>)
    try:
        inputs = tokenizer.apply_chat_template(messages, enable_thinking=False, **chat_kwargs).to(DEVICE)
    except (TypeError, ValueError):
        inputs = tokenizer.apply_chat_template(messages, **chat_kwargs).to(DEVICE)

    # Keep only keys the model accepts
    inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}

    # Build generate kwargs
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=2048,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\nModel output:\n{response}\n")
    return extract_json_from_response(response)


# =============================================================================
# JSON extraction
# =============================================================================

def extract_json_from_response(response: str) -> Any:
    try:
        if match := re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE):
            json_str = match.group(1)
        else:
            start  = response.find('{')
            start2 = response.find('[')
            candidates = [i for i in [start, start2] if i != -1]
            if not candidates:
                raise ValueError("No JSON found")
            start = min(candidates)
            end   = max(response.rfind('}'), response.rfind(']'))
            if end == -1:
                raise ValueError("No JSON found")
            json_str = response[start:end + 1]
        json_str = json_str.replace("\u00a0", " ")
        return json.loads(json_str)
    except Exception as e:
        print(f"JSON extraction failed: {e}\nRaw output:\n{response[:1000]}")
        return {"error": "Invalid JSON", "raw_output": response[:1000]}


# =============================================================================
# I/O helpers
# =============================================================================

def create_prompt() -> Dict:
    path = 'data/prompts/few_shot.json'
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_output(filename: str, data: Any):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved → {filename}")

def save_txt_output(filename: str, content: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Saved → {filename}")


# =============================================================================
# Metrics
# =============================================================================

def calculate_field_metrics(output_cases, gold_cases):
    fields = {k for c in gold_cases for k in c.keys() if k not in IGNORE_KEYS}
    metrics = {}
    for f in fields:
        correct = sum(1 for o, g in zip(output_cases, gold_cases) if o.get(f) == g.get(f))
        metrics[f] = {"accuracy": correct / len(gold_cases)}
    avg = sum(m["accuracy"] for m in metrics.values()) / len(metrics) if metrics else 0.0
    return metrics, avg

def calculate_overall_accuracy(output_cases, gold_cases):
    exact = sum(
        all(o.get(k) == g.get(k) for k in g.keys() if k not in IGNORE_KEYS)
        for o, g in zip(output_cases, gold_cases)
    )
    return exact / len(gold_cases) if gold_cases else 0.0

def generate_detailed_report(model_id: str, out_file: str, gold_file: str, report_path: str, perf: dict):
    with open(out_file, 'r', encoding='utf-8') as f:
        out = json.load(f)
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold = json.load(f)

    out_cases  = out.get("cases", [])
    gold_cases = gold.get("cases", [])

    mismatches = []
    for i, (o, g) in enumerate(zip(out_cases, gold_cases)):
        cid = o.get("case", f"Case {i+1}")
        for k in g.keys():
            if k in IGNORE_KEYS:
                continue
            if o.get(k) != g.get(k):
                mismatches.append(
                    f"Case: {cid} | {k}\n Out: {o.get(k)}\n Gold: {g.get(k)}\n{'-'*30}"
                )

    overall = calculate_overall_accuracy(out_cases, gold_cases)
    field_acc, avg_field = calculate_field_metrics(out_cases, gold_cases)

    lines = ["="*60, "ACCURACY REPORT", "="*60, f"Model: {model_id}"]
    if perf:
        lines.append("--- Performance (sec/case) ---")
        for k, v in perf.items():
            if k == 'N (Count)':
                lines.append(f" {k:<20} {int(v)}")
            else:
                lines.append(f" {k:<20} {v:.4f}")
        lines.append("")

    lines += [
        f"Overall exact-match: {overall:.4f}",
        f"Avg field accuracy : {avg_field:.4f}",
        "--- Per-field ---",
    ]
    for f, s in sorted(field_acc.items()):
        lines.append(f" {f:<25} {s['accuracy']:.4f}")

    lines.append("\n--- MISMATCHES ---")
    if mismatches:
        lines.append("\n".join(mismatches))
    else:
        lines.append("No mismatches found!")

    save_txt_output(report_path, "\n".join(lines))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inference script — run any supported model.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model key or full HuggingFace ID / local path.\n\n"
             "Available keys:\n" + "\n".join(f"  {k:25s} → {v}" for k, v in MODELS.items()),
    )
    args = parser.parse_args()

    model_id = MODELS.get(args.model, args.model)
    model_label = model_id.replace("/", "_").replace("output/", "")
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{ts}_{args.model}"

    base_dir = f"output/{model_label}"
    out_path = f"{base_dir}/output/{run_name}_output.json"
    rep_path = f"{base_dir}/report/{run_name}_report.txt"

    model, tokenizer = load_model_and_tokenizer(model_id)
    prompt = create_prompt()

    case_file = "data/test_data/case.json"
    gold_file = "data/test_data/gold_standard.json"

    with open(case_file, 'r', encoding='utf-8') as f:
        cases = json.load(f).get("cases", [])

    print(f"Loaded {len(cases)} cases → starting inference...\n")

    results = []
    times   = []

    for idx, case in enumerate(cases, start=1):
        cid = case.get("case", f"Case {idx}")
        print(f"Processing {cid} ({idx}/{len(cases)})")
        t0      = time.time()
        out     = generate_response(model, tokenizer, prompt, case)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"Time: {elapsed:.2f}s\n")

        if isinstance(out, dict):
            out["case"] = cid
            results.append(out)
        elif isinstance(out, list) and out:
            if isinstance(out[0], dict):
                out[0]["case"] = cid
            results.append(out[0] if len(out) == 1 else out)
        else:
            results.append({"case": cid, "raw_output": out})

    save_json_output(out_path, {"cases": results})

    if times:
        quantiles = statistics.quantiles(times, n=4) if len(times) > 3 else [min(times)] * 3
        iqr = quantiles[2] - quantiles[0] if len(quantiles) >= 3 else 0
        perf = {
            "N (Count)":          len(times),
            "Mean":               statistics.mean(times),
            "Median":             statistics.median(times),
            "Standard Deviation": statistics.stdev(times) if len(times) > 1 else 0,
            "Min":                min(times),
            "Max":                max(times),
            "IQR":                iqr,
        }
        generate_detailed_report(model_id, out_path, gold_file, rep_path, perf)

    print(f"\nDONE: {model_id}")
    print(f"Output → {out_path}")
    print(f"Report → {rep_path}")


if __name__ == "__main__":
    main()