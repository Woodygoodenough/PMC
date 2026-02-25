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
from runtime_utils import (
    choose_device,
    extract_choice_label,
    make_image_transform,
    resolve_hf_file,
    resolve_local_or_hub_ref,
    resolve_shard_glob,
)

try:
    import webdataset as wds
except ImportError as e:
    raise ImportError("webdataset is required. Install with: pip install webdataset") from e


class ModelManager:
    _patch_applied = False

    def __init__(self, repo_root: Path, checkpoints_dir: Path, device: str, verbose: bool = True):
        self.repo_root = repo_root
        self.checkpoints_dir = checkpoints_dir
        self.device = device
        self.verbose = verbose
        self._model = None
        self._tokenizer = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    @classmethod
    def _apply_compat_patches(cls, checkpoints_dir: Path, device: str, llm_torch_dtype: torch.dtype) -> None:
        if cls._patch_applied:
            return

        orig_from_pretrained = transformers.LlamaForCausalLM.from_pretrained

        def patched_from_pretrained(*args, **kwargs):
            kwargs.setdefault("torch_dtype", llm_torch_dtype)
            kwargs.setdefault("low_cpu_mem_usage", True)
            if args and not Path(str(args[0])).exists():
                kwargs.setdefault("cache_dir", str(checkpoints_dir / "hf_cache"))
                kwargs.setdefault("local_files_only", True)
                try:
                    return orig_from_pretrained(*args, **kwargs)
                except OSError:
                    kwargs["local_files_only"] = False
                    print(
                        f"[cache] miss model={args[0]}; downloading into {checkpoints_dir / 'hf_cache'}",
                        flush=True,
                    )
                    return orig_from_pretrained(*args, **kwargs)
            return orig_from_pretrained(*args, **kwargs)

        transformers.LlamaForCausalLM.from_pretrained = patched_from_pretrained

        orig_torch_load = torch.load

        def patched_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return orig_torch_load(*args, **kwargs)

        torch.load = patched_torch_load
        cls._patch_applied = True

    def load(
        self,
        llm_path: str | None,
        tokenizer_path: str | None,
        vision_model_path: str | None,
        checkpoint_path: str | None,
        llm_torch_dtype: torch.dtype,
        llm_repo_id: str | None = None,
        tokenizer_repo_id: str | None = None,
        vision_repo_id: str | None = None,
        vision_filename: str | None = None,
        checkpoint_repo_id: str | None = None,
        checkpoint_filename: str | None = None,
    ):
        if self._model is not None and self._tokenizer is not None:
            self._log("[model] using in-memory cache")
            return self._model, self._tokenizer

        self._log("[model] applying compatibility patches")
        self._apply_compat_patches(self.checkpoints_dir, self.device, llm_torch_dtype)

        import sys

        medvint_src = self.repo_root / "upstream_pmcvqa/src/MedVInT_TD"
        if str(medvint_src) not in sys.path:
            sys.path.insert(0, str(medvint_src))

        from models.QA_model import QA_model

        resolved_model_path = resolve_local_or_hub_ref(
            llm_path or "", checkpoints_dir=self.checkpoints_dir, default_repo_id=llm_repo_id
        )
        resolved_tokenizer_path = resolve_local_or_hub_ref(
            tokenizer_path or "", checkpoints_dir=self.checkpoints_dir, default_repo_id=tokenizer_repo_id
        )
        resolved_vision_path = resolve_hf_file(
            checkpoints_dir=self.checkpoints_dir,
            local_rel_path=(vision_model_path or "medvint/pmc_clip/checkpoint.pt"),
            repo_id=(vision_repo_id or "xmcmic/MedVInT-TE"),
            filename=(vision_filename or "pmc_clip/checkpoint.pt"),
            verbose=self.verbose,
            log_fn=self._log,
        )
        resolved_checkpoint = resolve_hf_file(
            checkpoints_dir=self.checkpoints_dir,
            local_rel_path=(checkpoint_path or "medvint/VQA_lora_PMC_LLaMA_PMCCLIP/choice/checkpoint-4000/pytorch_model.bin"),
            repo_id=(checkpoint_repo_id or "xmcmic/MedVInT-TD"),
            filename=(checkpoint_filename or "VQA_lora_PMC_LLaMA_PMCCLIP/choice/checkpoint-4000/pytorch_model.bin"),
            verbose=self.verbose,
            log_fn=self._log,
        )

        self._log(f"[model] llm={resolved_model_path}")
        self._log(f"[model] tokenizer={resolved_tokenizer_path}")
        self._log(f"[model] vision={resolved_vision_path}")
        self._log(f"[model] checkpoint={resolved_checkpoint}")

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
        self._log("[model] loading checkpoint into model")
        state = torch.load(str(resolved_checkpoint), map_location="cpu")
        target_keys = set(model.state_dict().keys())
        state = remap_state_dict_keys(state, target_keys=target_keys)
        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        if unexpected_keys:
            self._log(f"[warn] ignored unexpected checkpoint keys={len(unexpected_keys)}")
        if missing_keys:
            raise RuntimeError(f"Missing keys after remap/load: {len(missing_keys)}")

        model = model.to(self.device)
        model.eval()
        self._log(f"[model] ready on device={self.device}")
        self._log(
            f"[dtype] llama_embed={self.llm_dtype(model)} "
            f"vision_conv1={self.vision_dtype(model)}"
        )

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
                self._log(
                    f"[cache] miss tokenizer={resolved_tokenizer_path}; downloading into {self.checkpoints_dir / 'hf_cache'}"
                )
                tokenizer = transformers.LlamaTokenizer.from_pretrained(
                    str(resolved_tokenizer_path), legacy=True, **tokenizer_kwargs
                )
        else:
            tokenizer = transformers.LlamaTokenizer.from_pretrained(str(resolved_tokenizer_path), legacy=True)

        self._log("[model] tokenizer loaded")
        self._model = model
        self._tokenizer = tokenizer
        return self._model, self._tokenizer

    @staticmethod
    def llm_dtype(model) -> str:
        return str(model.llamacasual.get_input_embeddings().weight.dtype)

    @staticmethod
    def vision_dtype(model) -> str:
        try:
            return str(next(model.vision_model.parameters()).dtype)
        except StopIteration:
            return "unknown"


def build_parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Minimal MedVInT-TD evaluation on WebDataset shards")
    p.add_argument("--checkpoints-dir", type=str, required=True)
    p.add_argument("--datashards-dir", type=str, default=str(repo / "data/webdataset_v2/test_clean"))
    p.add_argument("--shards", type=str, default="test_clean-*.tar", help="Glob relative to --datashards-dir or absolute")

    p.add_argument("--llm-path", "--model-path", dest="llm_path", type=str, default="PMC_LLAMA_7B")
    p.add_argument("--tokenizer-path", type=str, default="PMC_LLAMA_7B")
    p.add_argument("--llm-repo-id", "--model-repo-id", dest="llm_repo_id", type=str, default="chaoyi-wu/PMC_LLAMA_7B")
    p.add_argument("--tokenizer-repo-id", type=str, default="chaoyi-wu/PMC_LLAMA_7B")
    p.add_argument("--vision-repo-id", type=str, default="xmcmic/MedVInT-TE")
    p.add_argument("--vision-filename", type=str, default="pmc_clip/checkpoint.pt")
    p.add_argument("--checkpoint-repo-id", type=str, default="xmcmic/MedVInT-TD")
    p.add_argument(
        "--checkpoint-filename",
        type=str,
        default="VQA_lora_PMC_LLaMA_PMCCLIP/choice/checkpoint-4000/pytorch_model.bin",
    )
    p.add_argument("--vision-model-path", type=str, default="medvint/pmc_clip/checkpoint.pt")
    p.add_argument(
        "--checkpoint-path", "--checkpoint",
        dest="checkpoint_path",
        type=str,
        default="medvint/VQA_lora_PMC_LLaMA_PMCCLIP/choice/checkpoint-4000/pytorch_model.bin",
    )

    p.add_argument("--max-samples", type=int, default=3)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--llm-dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--output-csv", type=str, default=str(repo / "eval_runs/minimal_eval_results.csv"))
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--predictor", type=str, default="model", choices=["model", "oracle", "random"])
    p.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return p


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
    choices = [
        payload.get("choice_a", ""),
        payload.get("choice_b", ""),
        payload.get("choice_c", ""),
        payload.get("choice_d", ""),
    ]
    question = payload.get("question", "")
    prompt = f"Question: {question}Choices:{choices[0]}{choices[1]}{choices[2]}{choices[3]}The Answer is:"
    gold_label = extract_choice_label(payload)
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

    def log(msg: str) -> None:
        if args.verbose:
            print(msg, flush=True)

    llm_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    llm_torch_dtype = llm_dtype_map[args.llm_dtype]

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
    log(
        f"[run] predictor={args.predictor} device={device} max_samples={args.max_samples} "
        f"llm_dtype={args.llm_dtype}"
    )
    log(f"[run] checkpoints_dir={checkpoints_dir}")
    log(f"[run] datashards_dir={datashards_dir}")

    model = None
    tokenizer = None
    if args.predictor == "model":
        manager = ModelManager(repo_root=repo_root, checkpoints_dir=checkpoints_dir, device=device, verbose=args.verbose)
        model, tokenizer = manager.load(
            llm_path=args.llm_path,
            tokenizer_path=args.tokenizer_path,
            vision_model_path=args.vision_model_path,
            checkpoint_path=args.checkpoint_path,
            llm_torch_dtype=llm_torch_dtype,
            llm_repo_id=args.llm_repo_id,
            tokenizer_repo_id=args.tokenizer_repo_id,
            vision_repo_id=args.vision_repo_id,
            vision_filename=args.vision_filename,
            checkpoint_repo_id=args.checkpoint_repo_id,
            checkpoint_filename=args.checkpoint_filename,
        )

    image_transform = make_image_transform(args.image_size)

    shard_pattern = resolve_shard_glob(datashards_dir, args.shards)
    log(f"[data] shard_pattern={shard_pattern}")
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
            log(f"[eval] step={total+1}/{args.max_samples} sample={sample_id} shard={shard_name}")

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

            log(
                f"[eval] done step={total}/{args.max_samples} sample={sample_id} "
                f"gold={gold_label} pred={pred_label} correct={is_correct} text={continuation[:80]!r}"
            )

            if total >= args.max_samples:
                break

    acc = correct / total if total else 0.0
    log(f"[result] finished samples={total} accuracy={acc:.4f}")
    log(f"[result] wrote_csv={output_csv}")


if __name__ == "__main__":
    main()
