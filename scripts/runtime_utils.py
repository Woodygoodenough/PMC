from __future__ import annotations

import shutil
import re
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
from PIL import Image
from torchvision import transforms


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


def make_image_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def is_readable_file(path: Path) -> bool:
    try:
        if not path.is_file():
            return False
        with path.open("rb"):
            return True
    except OSError:
        return False


def materialize_file(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if is_readable_file(target):
        return target
    shutil.copy2(source, target)
    if not is_readable_file(target):
        raise FileNotFoundError(f"Materialized file is not readable: {target}")
    return target


def resolve_local_or_hub_ref(ref: str, checkpoints_dir: Path, default_repo_id: str | None = None) -> str:
    candidate = Path(ref).expanduser() if ref else None
    if candidate and candidate.exists():
        return str(candidate.resolve())

    local = (checkpoints_dir / ref) if ref else None
    if local and local.exists():
        return str(local.resolve())

    if ref and "/" in ref and not ref.endswith((".pt", ".bin", ".json", ".model")):
        return ref

    if default_repo_id:
        return default_repo_id

    raise FileNotFoundError(f"Could not resolve local/HF reference: {ref}")


def resolve_hf_file(
    checkpoints_dir: Path,
    local_rel_path: str,
    repo_id: str,
    filename: str,
    *,
    verbose: bool = True,
    log_fn=None,
) -> Path:
    def log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)
        elif verbose:
            print(msg, flush=True)

    stable_target = (checkpoints_dir / local_rel_path).resolve()
    if is_readable_file(stable_target):
        log(f"[cache] hit {stable_target}")
        return stable_target

    cache_dir = checkpoints_dir / "hf_cache"
    cached = try_to_load_from_cache(repo_id=repo_id, filename=filename, cache_dir=str(cache_dir))
    if isinstance(cached, str):
        cached_path = Path(cached)
        if is_readable_file(cached_path):
            out = materialize_file(cached_path, stable_target)
            log(f"[cache] hit hf://{repo_id}/{filename} -> {cached_path}")
            if out != cached_path:
                log(f"[cache] materialized {out}")
            return out
        log(f"[cache] stale hf entry ignored hf://{repo_id}/{filename} -> {cached_path}")

    repo_cache_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    snapshot_hits = sorted(repo_cache_dir.glob(f"snapshots/*/{filename}"))
    for picked in reversed(snapshot_hits):
        if is_readable_file(picked):
            out = materialize_file(picked, stable_target)
            log(f"[cache] hit snapshot://{repo_id}/{filename} -> {picked}")
            if out != picked:
                log(f"[cache] materialized {out}")
            return out
        log(f"[cache] stale snapshot ignored snapshot://{repo_id}/{filename} -> {picked}")

    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
        source = Path(local)
        out = materialize_file(source, stable_target)
        log(f"[cache] hit hf://{repo_id}/{filename} -> {source}")
        if out != source:
            log(f"[cache] materialized {out}")
        return out
    except LocalEntryNotFoundError:
        pass

    log(f"[cache] miss {local_rel_path}; downloading {filename} from {repo_id}")
    try:
        downloaded = Path(hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(cache_dir)))
    except EntryNotFoundError as e:
        raise FileNotFoundError(f"Remote file not found: repo={repo_id}, filename={filename}") from e
    out = materialize_file(downloaded, stable_target)
    log(f"[cache] stored {downloaded}")
    if out != downloaded:
        log(f"[cache] materialized {out}")
    return out


def extract_choice_label(payload: dict) -> str:
    def normalize(value) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text or text.lower() == "none":
            return ""
        up = text.upper()
        if up in {"A", "B", "C", "D"}:
            return up
        if up.isdigit():
            n = int(up)
            if n in {0, 1, 2, 3}:
                return "ABCD"[n]
            if n in {1, 2, 3, 4}:
                return "ABCD"[n - 1]
        m = re.search(r"\b([ABCD])\b", up)
        if m:
            return m.group(1)
        return ""

    for key in ("answer_label", "answer", "label", "gt", "correct_answer"):
        label = normalize(payload.get(key))
        if label:
            return label
    return ""
