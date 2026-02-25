#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from huggingface_hub.errors import LocalEntryNotFoundError
from PIL import Image
from torch.optim import AdamW
from torchvision import transforms
from transformers import LlamaForCausalLM, LlamaTokenizer

try:
    import webdataset as wds
except ImportError as e:
    raise ImportError("webdataset is required. Install with: pip install webdataset") from e

REPO_ROOT = Path(__file__).resolve().parents[1]
MEDVINT_SRC = REPO_ROOT / "upstream_pmcvqa/src/MedVInT_TD"
if str(MEDVINT_SRC) not in sys.path:
    sys.path.insert(0, str(MEDVINT_SRC))

from models.blocks import ModifiedResNet, PMC_CLIP_cfg  # noqa: E402


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_shard_glob(datashards_dir: Path, shards: str) -> str:
    pattern = Path(shards)
    if pattern.is_absolute():
        return str(pattern)
    return str(datashards_dir / shards)


def build_prompt_and_answer(payload: dict) -> tuple[str, str]:
    question = str(payload.get("question", "")).strip()
    choices = [
        str(payload.get("choice_a", "")).strip(),
        str(payload.get("choice_b", "")).strip(),
        str(payload.get("choice_c", "")).strip(),
        str(payload.get("choice_d", "")).strip(),
    ]
    gold = str(payload.get("answer_label", "")).strip().upper()[:1]
    prompt = (
        f"Question: {question}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        "Answer:"
    )
    return prompt, gold


def _is_readable_file(path: Path) -> bool:
    try:
        if not path.is_file():
            return False
        with path.open("rb"):
            return True
    except OSError:
        return False


def _materialize(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if _is_readable_file(target):
        return target
    shutil.copy2(source, target)
    if not _is_readable_file(target):
        raise FileNotFoundError(f"Materialized file is not readable: {target}")
    return target


def resolve_vision_checkpoint(
    checkpoints_dir: Path,
    local_rel_path: str,
    repo_id: str,
    filename: str,
    verbose: bool = True,
) -> Path:
    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    stable_target = (checkpoints_dir / local_rel_path).resolve()
    if _is_readable_file(stable_target):
        log(f"[cache] hit {stable_target}")
        return stable_target

    cache_dir = checkpoints_dir / "hf_cache"
    remote_name = filename

    cached = try_to_load_from_cache(repo_id=repo_id, filename=remote_name, cache_dir=str(cache_dir))
    if isinstance(cached, str):
        cached_path = Path(cached)
        if _is_readable_file(cached_path):
            out = _materialize(cached_path, stable_target)
            log(f"[cache] hit hf://{repo_id}/{remote_name} -> {cached_path}")
            if out != cached_path:
                log(f"[cache] materialized {out}")
            return out
        log(f"[cache] stale hf entry ignored hf://{repo_id}/{remote_name} -> {cached_path}")

    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=remote_name,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
        source = Path(local)
        out = _materialize(source, stable_target)
        log(f"[cache] hit hf://{repo_id}/{remote_name} -> {source}")
        if out != source:
            log(f"[cache] materialized {out}")
        return out
    except LocalEntryNotFoundError:
        pass

    log(f"[cache] miss {local_rel_path}; downloading {remote_name} from {repo_id}")
    downloaded = Path(hf_hub_download(repo_id=repo_id, filename=remote_name, cache_dir=str(cache_dir)))
    out = _materialize(downloaded, stable_target)
    log(f"[cache] stored {downloaded}")
    if out != downloaded:
        log(f"[cache] materialized {out}")
    return out


def resolve_hf_model_ref(ref: str, default_repo: str, checkpoints_dir: Path) -> str:
    candidate = Path(ref).expanduser()
    if ref and candidate.exists():
        return str(candidate.resolve())
    default_local = checkpoints_dir / ref
    if ref and default_local.exists():
        return str(default_local.resolve())
    if ref and "/" in ref:
        return ref
    if not ref:
        return default_repo
    raise FileNotFoundError(f"Model/tokenizer ref not found locally and is not a HF repo id: {ref}")


class WebDatasetQATrainLoader:
    def __init__(self, shard_pattern: str):
        self.shard_paths = [Path(p) for p in sorted(glob.glob(shard_pattern))]
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards found for pattern: {shard_pattern}")
        dataset = wds.WebDataset([str(p) for p in self.shard_paths], shardshuffle=1000).decode("pil")
        self.dataset = dataset.to_tuple("__key__", "jpg;jpeg;png", "json")

    def __iter__(self):
        for key, image, payload in self.dataset:
            if isinstance(payload, (bytes, bytearray)):
                payload = json.loads(payload.decode("utf-8"))
            elif isinstance(payload, str):
                payload = json.loads(payload)
            if image.mode != "RGB":
                image = image.convert("RGB")
            yield key, payload, image


class TestModel(nn.Module):
    def __init__(
        self,
        llm_name_or_path: str,
        vision_checkpoint: str,
        llm_torch_dtype: torch.dtype,
        hf_cache_dir: str,
        num_query_tokens: int = 32,
        qformer_hidden_dim: int = 1024,
        qformer_layers: int = 6,
        qformer_heads: int = 8,
        freeze_llm: bool = True,
        freeze_vision: bool = True,
    ) -> None:
        super().__init__()

        vision_cfg = PMC_CLIP_cfg()
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        vision_model = ModifiedResNet(
            layers=vision_cfg.layers,
            heads=vision_heads,
            output_dim=768,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
        ckpt = torch.load(vision_checkpoint, map_location="cpu", weights_only=False)
        if "state_dict" not in ckpt:
            raise KeyError("Vision checkpoint must contain key 'state_dict'")
        state_dict = {k.replace("module.visual.", ""): v for k, v in ckpt["state_dict"].items() if ".visual" in k}
        vision_model.load_state_dict(state_dict, strict=True)
        self.vision_model = nn.Sequential(*list(vision_model.children())[:-2])

        self.vision_to_qformer = nn.Linear(1024, qformer_hidden_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, qformer_hidden_dim) * 0.02)
        q_layer = nn.TransformerDecoderLayer(
            d_model=qformer_hidden_dim,
            nhead=qformer_heads,
            dim_feedforward=qformer_hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.qformer = nn.TransformerDecoder(q_layer, num_layers=qformer_layers, norm=nn.LayerNorm(qformer_hidden_dim))

        self.llm = LlamaForCausalLM.from_pretrained(
            llm_name_or_path,
            torch_dtype=llm_torch_dtype,
            low_cpu_mem_usage=True,
            cache_dir=hf_cache_dir,
        )
        llm_embed_dim = self.llm.get_input_embeddings().weight.shape[1]
        self.q_to_llm = nn.Linear(qformer_hidden_dim, llm_embed_dim)

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
        if freeze_vision:
            for p in self.vision_model.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: torch.Tensor,
    ):
        bsz = input_ids.shape[0]

        vis = self.vision_model(pixel_values).flatten(2).transpose(1, 2)
        vis = self.vision_to_qformer(vis)
        queries = self.query_tokens.expand(bsz, -1, -1)
        aligned_queries = self.qformer(tgt=queries, memory=vis)

        llm_dtype = self.llm.get_input_embeddings().weight.dtype
        visual_prefix = self.q_to_llm(aligned_queries).to(llm_dtype)
        text_embeds = self.llm.get_input_embeddings()(input_ids).to(llm_dtype)
        inputs_embeds = torch.cat([visual_prefix, text_embeds], dim=1)

        prefix_len = visual_prefix.shape[1]
        prefix_mask = torch.ones((bsz, prefix_len), dtype=attention_mask.dtype, device=attention_mask.device)
        full_attention = torch.cat([prefix_mask, attention_mask], dim=1)
        ignore_prefix = torch.full((bsz, prefix_len), -100, dtype=labels.dtype, device=labels.device)
        aligned_labels = torch.cat([ignore_prefix, labels], dim=1)

        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=full_attention, labels=aligned_labels)
        return out, aligned_labels, prefix_len


def make_batch(
    records: list[tuple[str, dict, Image.Image]],
    tokenizer: LlamaTokenizer,
    image_transform: transforms.Compose,
    max_text_len: int,
    device: str,
) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    attention_masks: list[list[int]] = []
    pixels: list[torch.Tensor] = []
    first_answer_positions: list[int] = []
    first_answer_token_ids: list[int] = []

    for _, payload, image in records:
        prompt, answer = build_prompt_and_answer(payload)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = tokenizer.encode(f" {answer}", add_special_tokens=False)
        if not answer_ids:
            answer_ids = [tokenizer.eos_token_id]

        seq = prompt_ids + answer_ids + [tokenizer.eos_token_id]
        lab = ([-100] * len(prompt_ids)) + answer_ids + [tokenizer.eos_token_id]

        first_answer_pos = len(prompt_ids)
        first_answer_token = answer_ids[0]

        seq = seq[:max_text_len]
        lab = lab[:max_text_len]
        attn = [1] * len(seq)

        if first_answer_pos >= max_text_len:
            first_answer_pos = -1

        pad_len = max_text_len - len(seq)
        if pad_len > 0:
            seq = seq + [pad_id] * pad_len
            lab = lab + [-100] * pad_len
            attn = attn + [0] * pad_len

        input_ids.append(seq)
        labels.append(lab)
        attention_masks.append(attn)
        pixels.append(image_transform(image))
        first_answer_positions.append(first_answer_pos)
        first_answer_token_ids.append(first_answer_token)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long, device=device),
        "pixel_values": torch.stack(pixels, dim=0).to(device=device, dtype=torch.float32),
        "first_answer_pos": torch.tensor(first_answer_positions, dtype=torch.long, device=device),
        "first_answer_token_id": torch.tensor(first_answer_token_ids, dtype=torch.long, device=device),
    }


def compute_metrics(
    logits: torch.Tensor,
    aligned_labels: torch.Tensor,
    prefix_len: int,
    first_answer_pos: torch.Tensor,
    first_answer_token_id: torch.Tensor,
) -> tuple[float, float]:
    pred_ids = logits.argmax(dim=-1)

    supervised = aligned_labels != -100
    if supervised.any():
        token_acc = (pred_ids[supervised] == aligned_labels[supervised]).float().mean().item()
    else:
        token_acc = float("nan")

    bsz = logits.shape[0]
    correct = 0
    valid = 0
    for i in range(bsz):
        pos = int(first_answer_pos[i].item())
        if pos < 0:
            continue
        idx = prefix_len + pos
        if idx >= logits.shape[1]:
            continue
        valid += 1
        pred = int(pred_ids[i, idx].item())
        gold = int(first_answer_token_id[i].item())
        if pred == gold:
            correct += 1
    answer_acc = (correct / valid) if valid > 0 else float("nan")

    return token_acc, answer_acc


def init_metrics_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "loss",
                "token_acc",
                "answer_acc",
                "lr",
                "elapsed_sec",
                "run_start_time",
                "step_end_time",
            ]
        )


def append_metrics_csv(path: Path, row: dict) -> None:
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["step"],
            f"{row['loss']:.8f}",
            f"{row['token_acc']:.8f}",
            f"{row['answer_acc']:.8f}",
            f"{row['lr']:.10f}",
            f"{row['elapsed_sec']:.4f}",
            row["run_start_time"],
            row["step_end_time"],
        ])


def save_training_curve(rows: list[dict], curve_path: Path) -> None:
    if not rows:
        return
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    steps = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]
    token_accs = [r["token_acc"] for r in rows]
    answer_accs = [r["answer_acc"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(steps, losses, color="#d62728", label="loss", linewidth=2)
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(steps, token_accs, color="#1f77b4", label="token_acc", linewidth=1.8)
    ax2.plot(steps, answer_accs, color="#2ca02c", label="answer_acc", linewidth=1.8)
    ax2.set_ylabel("accuracy", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(0.0, 1.0)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TestModel (PMC vision + Q-Former + PMC LLaMA)")
    p.add_argument("--checkpoints-dir", type=str, required=True)
    p.add_argument("--datashards-dir", type=str, required=True)
    p.add_argument("--shards", type=str, default="train_2-*.tar")

    p.add_argument("--llm-path", type=str, default="chaoyi-wu/PMC_LLAMA_7B")
    p.add_argument("--tokenizer-path", type=str, default="chaoyi-wu/PMC_LLAMA_7B")
    p.add_argument("--llm-dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    p.add_argument("--vision-model-path", type=str, default="medvint/pmc_clip/checkpoint.pt")
    p.add_argument("--vision-repo-id", type=str, default="xmcmic/MedVInT-TE")
    p.add_argument("--vision-filename", type=str, default="pmc_clip/checkpoint.pt")

    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=1)
    p.add_argument("--max-text-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    p.add_argument("--freeze-llm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--freeze-vision", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--qformer-layers", type=int, default=6)
    p.add_argument("--qformer-hidden-dim", type=int, default=1024)
    p.add_argument("--qformer-heads", type=int, default=8)
    p.add_argument("--num-query-tokens", type=int, default=32)

    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--metrics-csv", type=str, default="logs/train_runs/testmodel_metrics.csv")
    p.add_argument("--curve-path", type=str, default="logs/train_runs/testmodel_curve.png")
    p.add_argument("--save-bridge-path", type=str, default="")
    p.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    def log(msg: str) -> None:
        if args.verbose:
            print(msg, flush=True)

    checkpoints_dir = Path(args.checkpoints_dir).expanduser().resolve()
    datashards_dir = Path(args.datashards_dir).expanduser().resolve()
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"--checkpoints-dir does not exist: {checkpoints_dir}")
    if not datashards_dir.exists():
        raise FileNotFoundError(f"--datashards-dir does not exist: {datashards_dir}")

    device = choose_device(args.device)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    llm_torch_dtype = dtype_map[args.llm_dtype]
    hf_cache_dir = str(checkpoints_dir / "hf_cache")

    vision_ckpt = resolve_vision_checkpoint(
        checkpoints_dir=checkpoints_dir,
        local_rel_path=args.vision_model_path,
        repo_id=args.vision_repo_id,
        filename=args.vision_filename,
        verbose=args.verbose,
    )

    resolved_llm_path = resolve_hf_model_ref(args.llm_path, "chaoyi-wu/PMC_LLAMA_7B", checkpoints_dir)
    resolved_tokenizer_path = resolve_hf_model_ref(args.tokenizer_path, "chaoyi-wu/PMC_LLAMA_7B", checkpoints_dir)

    tokenizer = LlamaTokenizer.from_pretrained(resolved_tokenizer_path, legacy=True, cache_dir=hf_cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"[run] device={device} steps={args.max_steps} batch={args.batch_size} log_every={args.log_every}")
    log(f"[model] llm={resolved_llm_path}")
    log(f"[model] tokenizer={resolved_tokenizer_path}")
    log(f"[model] vision={vision_ckpt}")

    model = TestModel(
        llm_name_or_path=resolved_llm_path,
        vision_checkpoint=str(vision_ckpt),
        llm_torch_dtype=llm_torch_dtype,
        hf_cache_dir=hf_cache_dir,
        num_query_tokens=args.num_query_tokens,
        qformer_hidden_dim=args.qformer_hidden_dim,
        qformer_layers=args.qformer_layers,
        qformer_heads=args.qformer_heads,
        freeze_llm=args.freeze_llm,
        freeze_vision=args.freeze_vision,
    ).to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters. Disable --freeze-llm and/or --freeze-vision if needed.")

    trainable = sum(p.numel() for p in params)
    total = sum(p.numel() for p in model.parameters())
    log(f"[model] trainable={trainable:,} / total={total:,} ({100.0 * trainable / max(total, 1):.4f}%)")

    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    image_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    shard_pattern = resolve_shard_glob(datashards_dir, args.shards)
    loader = WebDatasetQATrainLoader(shard_pattern=shard_pattern)
    stream = iter(loader)

    metrics_csv = Path(args.metrics_csv).expanduser().resolve()
    curve_path = Path(args.curve_path).expanduser().resolve()
    init_metrics_csv(metrics_csv)
    rows: list[dict] = []

    run_start_ts = time.time()
    run_start_time = datetime.now().isoformat(timespec="seconds")
    for step in range(1, args.max_steps + 1):
        records = []
        for _ in range(args.batch_size):
            try:
                records.append(next(stream))
            except StopIteration:
                stream = iter(loader)
                records.append(next(stream))

        batch = make_batch(
            records=records,
            tokenizer=tokenizer,
            image_transform=image_transform,
            max_text_len=args.max_text_len,
            device=device,
        )

        out, aligned_labels, prefix_len = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
        )
        loss = out.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        token_acc, answer_acc = compute_metrics(
            logits=out.logits,
            aligned_labels=aligned_labels,
            prefix_len=prefix_len,
            first_answer_pos=batch["first_answer_pos"],
            first_answer_token_id=batch["first_answer_token_id"],
        )

        row = {
            "step": step,
            "loss": float(loss.item()),
            "token_acc": float(token_acc),
            "answer_acc": float(answer_acc),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_sec": float(time.time() - run_start_ts),
            "run_start_time": run_start_time,
            "step_end_time": datetime.now().isoformat(timespec="seconds"),
        }
        append_metrics_csv(metrics_csv, row)
        rows.append(row)

        if step % args.log_every == 0 or step == 1 or step == args.max_steps:
            ppl = math.exp(min(20.0, row["loss"]))
            log(
                "[train] "
                f"step={step}/{args.max_steps} "
                f"loss={row['loss']:.6f} ppl={ppl:.4f} "
                f"token_acc={row['token_acc']:.4f} answer_acc={row['answer_acc']:.4f} "
                f"grad_norm={float(grad_norm):.4f}"
            )
            save_training_curve(rows, curve_path)

    if args.save_bridge_path:
        out_path = Path(args.save_bridge_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "qformer": model.qformer.state_dict(),
                "query_tokens": model.query_tokens.detach().cpu(),
                "vision_to_qformer": model.vision_to_qformer.state_dict(),
                "q_to_llm": model.q_to_llm.state_dict(),
            },
            out_path,
        )
        log(f"[save] {out_path}")

    log(f"[metrics] csv={metrics_csv}")
    log(f"[metrics] curve={curve_path}")
    log("[done] training finished")


if __name__ == "__main__":
    main()
