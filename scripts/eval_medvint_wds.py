#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import difflib
import glob
import json
import random
import re
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
from PIL import Image
from torchvision import transforms

try:
    import webdataset as wds
except ImportError as e:
    raise ImportError("webdataset is required. Install with: pip install webdataset") from e


class ModelManager:
    _patch_applied = False

    def __init__(self, repo_root: Path, checkpoints_dir: Path, device: str):
        self.repo_root = repo_root
        self.checkpoints_dir = checkpoints_dir
        self.device = device
        self._model = None
        self._tokenizer = None

    @classmethod
    def _apply_compat_patches(cls, checkpoints_dir: Path, device: str) -> None:
        if cls._patch_applied:
            return

        orig_from_pretrained = transformers.LlamaForCausalLM.from_pretrained

        def patched_from_pretrained(*args, **kwargs):
            kwargs.setdefault("torch_dtype", torch.float16 if device != "cpu" else torch.float32)
            kwargs.setdefault("low_cpu_mem_usage", True)
            # If a hub id is provided, downloaded weights are cached under checkpoints_dir.
            if args and not Path(str(args[0])).exists():
                kwargs.setdefault("cache_dir", str(checkpoints_dir / "hf_cache"))
                kwargs.setdefault("local_files_only", True)
                try:
                    return orig_from_pretrained(*args, **kwargs)
                except OSError:
                    kwargs["local_files_only"] = False
                    print(f"Local cache miss for {args[0]}; downloading into {checkpoints_dir / 'hf_cache'}")
                    return orig_from_pretrained(*args, **kwargs)
            return orig_from_pretrained(*args, **kwargs)

        transformers.LlamaForCausalLM.from_pretrained = patched_from_pretrained

        orig_torch_load = torch.load

        def patched_torch_load(*args, **kwargs):
            # Required for legacy checkpoints on torch>=2.6.
            kwargs.setdefault("weights_only", False)
            return orig_torch_load(*args, **kwargs)

        torch.load = patched_torch_load
        cls._patch_applied = True

    def _resolve_path(self, value: str, default_rel: str) -> Path:
        candidate = Path(value) if value else self.checkpoints_dir / default_rel
        if candidate.is_absolute() and candidate.exists():
            return candidate
        alt = (self.checkpoints_dir / value) if value else (self.checkpoints_dir / default_rel)
        if alt.exists():
            return alt
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Could not resolve checkpoint path: {value or default_rel}")

    def _resolve_local_or_hub(self, value: str, default_rel: str) -> str:
        candidate = Path(value) if value else self.checkpoints_dir / default_rel
        if candidate.is_absolute() and candidate.exists():
            return str(candidate)
        alt = (self.checkpoints_dir / value) if value else (self.checkpoints_dir / default_rel)
        if alt.exists():
            return str(alt)
        if candidate.exists():
            return str(candidate)
        # Fall back to hub id only when user explicitly passed a non-path token.
        if value and "/" in value and not value.endswith((".bin", ".pt", ".json", ".model")):
            return value
        raise FileNotFoundError(f"Could not resolve model/tokenizer path locally: {value or default_rel}")

    def load(self, model_path: str | None, tokenizer_path: str | None, vision_model_path: str | None, checkpoint: str | None):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        self._apply_compat_patches(self.checkpoints_dir, self.device)

        import sys

        medvint_src = self.repo_root / "upstream_pmcvqa/src/MedVInT_TD"
        if str(medvint_src) not in sys.path:
            sys.path.insert(0, str(medvint_src))

        from models.QA_model import QA_model

        resolved_model_path = self._resolve_local_or_hub(model_path or "", "PMC_LLAMA_7B")
        resolved_tokenizer_path = self._resolve_local_or_hub(tokenizer_path or "", "PMC_LLAMA_7B")
        resolved_vision_path = self._resolve_path(vision_model_path or "", "medvint/pmc_clip/checkpoint.pt")
        resolved_checkpoint = self._resolve_path(
            checkpoint or "",
            "medvint/VQA_lora_PMC_LLaMA_PMCCLIP/choice/checkpoint-4000/pytorch_model.bin",
        )

        model_args = SimpleNamespace(
            model_path=str(resolved_model_path),
            ckp="",
            checkpointing=False,
            N=12,
            H=8,
            img_token_num=32,
            voc_size=32000,
            hidden_dim=4096,
            Vision_module="PMC-CLIP",
            visual_model_path=str(resolved_vision_path),
            is_lora=True,
            peft_mode="lora",
            lora_rank=8,
        )

        model = QA_model(model_args)
        state = torch.load(str(resolved_checkpoint), map_location="cpu")
        target_keys = set(model.state_dict().keys())
        state = remap_state_dict_keys(state, target_keys=target_keys)
        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        if unexpected_keys:
            print(f"Warning: ignored unexpected checkpoint keys: {len(unexpected_keys)}")
        if missing_keys:
            # Keep this explicit so we fail fast if compatibility mapping regresses.
            raise RuntimeError(f"Missing keys after remap/load: {len(missing_keys)}")
        model = model.to(self.device)
        model.eval()

        tokenizer_kwargs = {}
        if not Path(str(resolved_tokenizer_path)).exists():
            tokenizer_kwargs["cache_dir"] = str(self.checkpoints_dir / "hf_cache")
            tokenizer_kwargs["local_files_only"] = True
            try:
                tokenizer = transformers.LlamaTokenizer.from_pretrained(
                    str(resolved_tokenizer_path), legacy=True, **tokenizer_kwargs
                )
            except OSError:
                tokenizer_kwargs["local_files_only"] = False
                print(f"Local tokenizer cache miss for {resolved_tokenizer_path}; downloading into {self.checkpoints_dir / 'hf_cache'}")
                tokenizer = transformers.LlamaTokenizer.from_pretrained(
                    str(resolved_tokenizer_path), legacy=True, **tokenizer_kwargs
                )
        else:
            tokenizer = transformers.LlamaTokenizer.from_pretrained(str(resolved_tokenizer_path), legacy=True)

        self._model = model
        self._tokenizer = tokenizer
        return self._model, self._tokenizer


def build_parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Minimal MedVInT-TD evaluation on WebDataset shards")
    p.add_argument("--checkpoints-dir", type=str, required=True)
    p.add_argument("--datashards-dir", type=str, default=str(repo / "data/webdataset_v2/test_clean"))
    p.add_argument("--shards", type=str, default="test_clean-*.tar", help="Glob relative to --datashards-dir or absolute")

    p.add_argument("--model-path", type=str, default="PMC_LLAMA_7B")
    p.add_argument("--tokenizer-path", type=str, default="PMC_LLAMA_7B")
    p.add_argument("--vision-model-path", type=str, default="medvint/pmc_clip/checkpoint.pt")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="medvint/VQA_lora_PMC_LLaMA_PMCCLIP/choice/checkpoint-4000/pytorch_model.bin",
    )

    p.add_argument("--max-samples", type=int, default=3)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--output-csv", type=str, default=str(repo / "eval_runs/minimal_eval_results.csv"))
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--predictor", type=str, default="model", choices=["model", "oracle", "random"])
    return p


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def remap_state_dict_keys(
    state_dict: dict[str, torch.Tensor], target_keys: set[str] | None = None
) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    target_keys = target_keys or set()

    for key, value in state_dict.items():
        candidates = [key]
        if "lora_A.weight" in key:
            candidates.append(key.replace("lora_A.weight", "lora_A.default.weight"))
        if "lora_B.weight" in key:
            candidates.append(key.replace("lora_B.weight", "lora_B.default.weight"))
        if "self_attn.q_proj.weight" in key and "base_layer" not in key:
            candidates.append(key.replace("self_attn.q_proj.weight", "self_attn.q_proj.base_layer.weight"))
        if "self_attn.v_proj.weight" in key and "base_layer" not in key:
            candidates.append(key.replace("self_attn.v_proj.weight", "self_attn.v_proj.base_layer.weight"))

        selected = None
        if target_keys:
            for c in candidates:
                if c in target_keys:
                    selected = c
                    break
            if selected is None:
                continue
        else:
            selected = candidates[-1]

        remapped[selected] = value

    return remapped


def resolve_shard_glob(datashards_dir: Path, shards: str) -> str:
    pattern = Path(shards)
    if pattern.is_absolute():
        return str(pattern)
    return str(datashards_dir / shards)


class WebDatasetQALoader:
    def __init__(self, shard_pattern: str):
        self.shard_paths = [Path(p) for p in sorted(glob.glob(shard_pattern))]
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards found for pattern: {shard_pattern}")

        self.dataset = (
            wds.WebDataset([str(p) for p in self.shard_paths], shardshuffle=False)
            .decode("pil")
            .to_tuple("__key__", "__url__", "jpg;jpeg;png", "json")
        )

    def __iter__(self):
        for key, url, image, payload in self.dataset:
            if isinstance(payload, (bytes, bytearray)):
                payload = json.loads(payload.decode("utf-8"))
            elif isinstance(payload, str):
                payload = json.loads(payload)
            shard_name = Path(url).name
            if image.mode != "RGB":
                image = image.convert("RGB")
            yield shard_name, key, payload, image


def build_prompt(payload: dict) -> tuple[str, list[str], str]:
    choices = [payload.get("choice_a", ""), payload.get("choice_b", ""), payload.get("choice_c", ""), payload.get("choice_d", "")]
    question = payload.get("question", "")
    prompt = f"Question: {question}Choices:{choices[0]}{choices[1]}{choices[2]}{choices[3]}The Answer is:"
    gold_label = str(payload.get("answer_label", "")).strip().upper()[:1]
    return prompt, choices, gold_label


def label_from_text(text: str, choices: list[str]) -> str:
    matches = re.findall(r"\b([ABCD])\b", text.upper())
    if matches:
        return matches[-1]
    labels = ["A", "B", "C", "D"]
    best_idx = 0
    best_score = -1.0
    text_norm = text.strip().lower()
    for i, choice in enumerate(choices):
        score = similarity(text_norm, str(choice).strip().lower())
        if score > best_score:
            best_score = score
            best_idx = i
    return labels[best_idx]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    checkpoints_dir = Path(args.checkpoints_dir).expanduser().resolve()
    datashards_dir = Path(args.datashards_dir).expanduser().resolve()
    if not checkpoints_dir.exists() or not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"--checkpoints-dir must exist and be a directory: {checkpoints_dir}")
    if not datashards_dir.exists() or not datashards_dir.is_dir():
        raise FileNotFoundError(f"--datashards-dir must exist and be a directory: {datashards_dir}")
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(f"Checkpoints dir: {checkpoints_dir}")
    print(f"Datashards dir: {datashards_dir}")

    model = None
    tokenizer = None
    if args.predictor == "model":
        manager = ModelManager(repo_root=repo_root, checkpoints_dir=checkpoints_dir, device=device)
        model, tokenizer = manager.load(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            vision_model_path=args.vision_model_path,
            checkpoint=args.checkpoint,
        )

    image_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    shard_pattern = resolve_shard_glob(datashards_dir, args.shards)
    qa_loader = WebDatasetQALoader(shard_pattern)

    correct = 0
    total = 0

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "shard", "gold_label", "pred_label", "is_correct", "prediction_text"],
        )
        writer.writeheader()

        for shard_name, sample_id, payload, image in qa_loader:
            prompt, choices, gold_label = build_prompt(payload)
            continuation = ""
            print(f"starting sample={sample_id}", flush=True)
            if args.predictor == "model":
                encoded = tokenizer(prompt, return_tensors="pt")
                input_ids = encoded["input_ids"].to(device)
                pixel_values = image_transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model.generate(input_ids, pixel_values)
                    generated = logits.argmax(-1)

                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
                continuation = decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()
                if not continuation:
                    continuation = decoded[-32:].strip()
                pred_label = label_from_text(continuation, choices)
            elif args.predictor == "oracle":
                pred_label = gold_label
                continuation = f"oracle:{gold_label}"
            else:
                pred_label = random.choice(["A", "B", "C", "D"])
                continuation = f"random:{pred_label}"

            is_correct = int(pred_label == gold_label)
            correct += is_correct
            total += 1

            writer.writerow(
                {
                    "sample_id": sample_id,
                    "shard": shard_name,
                    "gold_label": gold_label,
                    "pred_label": pred_label,
                    "is_correct": is_correct,
                    "prediction_text": continuation,
                }
            )

            print(
                f"step={total} sample={sample_id} gold={gold_label} pred={pred_label} "
                f"correct={is_correct} text={continuation[:80]!r}"
            )

            if total >= args.max_samples:
                break

    acc = correct / total if total else 0.0
    print(f"Finished {total} samples. accuracy={acc:.4f}")
    print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    main()
