# PMC-VQA WebDataset (Rematched)

## Summary
Matching was redone for **all** `test_clean.csv` rows using both image archives.

Result:
- `test_clean.csv` rows matched in `images.zip` (exact basename): **2000/2000**
- `test_clean.csv` rows matched in `images_2.zip` (exact basename): partial only
- Final rematched build uses:
  - `train_2` + `test_2` from `images_2.zip`
  - `test_clean` from `images.zip`

## Output Location
- WebDataset root: `data/webdataset_v2/`
- Build stats: `data/webdataset_v2/dataset_stats.json`
- Split shards:
  - `data/webdataset_v2/train_2/train_2-*.tar`
  - `data/webdataset_v2/test_2/test_2-*.tar`
  - `data/webdataset_v2/test_clean/test_clean-*.tar`
- Skip logs:
  - `data/webdataset_v2/train_2/train_2_skipped_rows.csv`
  - `data/webdataset_v2/test_2/test_2_skipped_rows.csv`
  - `data/webdataset_v2/test_clean/test_clean_skipped_rows.csv`

## Environment
- Conda env: `pmc_vqa_wds`
- Python 3.11
- Packages: `pandas`, `numpy`, `pillow`, `tqdm`, `webdataset`

## Build Command
```bash
conda run -n pmc_vqa_wds python scripts/build_pmc_webdataset.py \
  --output-dir data/webdataset_v2 \
  --maxcount 2000
```

## Matching Policy (Rematched)
- Source order is split-specific:
  - `train_2`: `images_2.zip` then `images.zip`
  - `test_2`: `images_2.zip` then `images.zip`
  - `test_clean`: `images.zip` then `images_2.zip`
- Current run used exact basename matching only (no canonical fallback needed).

## WebDataset Schema
Each sample stores:
- `<image_ext>`: image bytes (`jpg` here)
- `json`: record with fields
  - `split`
  - `row_idx`
  - `figure_path_csv`
  - `figure_path_resolved`
  - `image_source_zip` (new; `images.zip` or `images_2.zip`)
  - `resolve_mode`
  - `question`
  - `answer`
  - `choice_a`
  - `choice_b`
  - `choice_c`
  - `choice_d`
  - `answer_label`
  - `caption`
  - `source_index`
  - `source_split`

## Split Statistics
### train_2
- Rows total: 152,603
- Samples written: 152,603
- Rows skipped: 0
- Unique images written: 135,339
- Image source: `images_2.zip` only (152,603)
- Duplicate-image QA behavior:
  - images with 1 QA: 118,115
  - images with >1 QA: 17,224
  - max QAs per image: 4
  - mean QAs/image: 1.1276

### test_2
- Rows total: 33,430
- Samples written: 33,430
- Rows skipped: 0
- Unique images written: 29,021
- Image source: `images_2.zip` only (33,430)
- Duplicate-image QA behavior:
  - images with 1 QA: 24,616
  - images with >1 QA: 4,405
  - max QAs per image: 4
  - mean QAs/image: 1.1519

### test_clean
- Rows total: 2,000
- Samples written: 2,000
- Rows skipped: 0
- Unique images written: 1,440
- Image source: `images.zip` only (2,000)
- Duplicate-image QA behavior:
  - images with 1 QA: 1,009
  - images with >1 QA: 431
  - max QAs per image: 5
  - mean QAs/image: 1.3889

## Duplication Findings
Yes, duplicate image usage exists (multiple QA rows per image) in all splits.
- `train_2`: 17,224 images reused across multiple QA rows
- `test_2`: 4,405 images reused across multiple QA rows
- `test_clean`: 431 images reused across multiple QA rows

This is preserved intentionally in the shards: each QA row is emitted as one sample, and repeated images appear in multiple samples.

## Archive-Level Stats
- `images_2.zip`:
  - files: 164,360
  - canonical-collision keys: 40,254
- `images.zip`:
  - files: 149,075
  - canonical-collision keys: 17

## Notes
- Previous partial `test_clean` coverage came from forcing `test_clean` to map against `images_2.zip`.
- After redoing matching against `images.zip` for all entries, `test_clean` is fully paired with zero skips.
