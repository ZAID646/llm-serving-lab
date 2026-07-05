import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from huggingface_hub import HfApi

MODEL_ID = os.environ.get("SOURCE_MODEL", "google/gemma-2-9b")
OUTPUT_REPO = os.environ.get("OUTPUT_REPO", "zaid/gemma-2-9b-awq")
CALIBRATION_SAMPLES = int(os.environ.get("CALIBRATION_SAMPLES", "64"))
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "512"))


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable is required")
        sys.exit(1)

    print(f"Model: {MODEL_ID}")
    print(f"Output: {OUTPUT_REPO}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        free, total = torch.cuda.mem_get_info()
        print(f"Free VRAM: {free / 1e9:.1f} GB / {total / 1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model with aggressive offloading (GPU capped at 11GB)...")
    from accelerate import infer_auto_device_map
    max_memory = {0: "11GiB", "cpu": "40GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=token,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    free_cuda, total_cuda = torch.cuda.mem_get_info()
    print(f"VRAM after load: {(total_cuda - free_cuda) / 1e9:.1f} GB used / {total_cuda / 1e9:.1f} GB total")

    print("Loading calibration data (wikitext-2)...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds = ds.select(range(min(CALIBRATION_SAMPLES, len(ds))))
    print(f"Calibration samples: {len(ds)}")

    recipe = [
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(
            ignore=["lm_head"],
            scheme="W4A16_ASYM",
            targets=["Linear"],
        ),
    ]

    print("Starting AWQ quantization (sequential, GPU offload)...")
    print(f"  max_seq_length={MAX_SEQ_LENGTH}")
    print(f"  batch_size=1")
    oneshot(
        model=model,
        dataset=ds,
        tokenizer=tokenizer,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LENGTH,
        num_calibration_samples=CALIBRATION_SAMPLES,
        batch_size=1,
    )

    save_dir = "/tmp/quantized"
    print(f"Saving model to {save_dir}...")
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)

    print(f"Pushing to HF Hub: {OUTPUT_REPO}")
    api = HfApi(token=token)
    api.create_repo(repo_id=OUTPUT_REPO, exist_ok=True, private=False)
    api.upload_folder(
        folder_path=save_dir,
        repo_id=OUTPUT_REPO,
        repo_type="model",
    )

    print(f"Done: https://huggingface.co/{OUTPUT_REPO}")


if __name__ == "__main__":
    main()
