#!/usr/bin/env python3
"""
Full StereoLunar Dataset Downloader & Fine-Tuning Manifest Generator
===================================================================

Downloads the official StereoLunar dataset from Zenodo (Record 17514376),
extracts it sequentially to maintain strict disk safety on constrained storage,
and organizes it into properly partitioned scenes with full train/val/test manifests
ready for EfficientLoFTR fine-tuning.

Zenodo Record: https://zenodo.org/records/17514376
- dynamic.zip: 1.66 GB (~1.88 GB uncompressed, 12,696 images)
- nadir.zip:   3.48 GB (~4.06 GB uncompressed, 27,066 images)
- oblique.zip: 3.90 GB (~3.75 GB uncompressed, 25,040 images)
Total compressed: ~9.05 GB | Total uncompressed: ~9.7 GB

Usage:
  # Download all 3 trajectories (default):
  python scripts/download_stereo_lunar.py

  # Download specific trajectory (e.g. dynamic only):
  python scripts/download_stereo_lunar.py --trajectories dynamic

  # Dry-run check (disk space & URL reachability):
  python scripts/download_stereo_lunar.py --dry_run
"""

import os
import sys
import shutil
import subprocess
import argparse
import json
from pathlib import Path
import numpy as np
from loguru import logger

# Zenodo URLs and metadata for StereoLunar Record 17514376
ZENODO_RECORD_URL = "https://zenodo.org/api/records/17514376"
ARCHIVES = {
    "dynamic": {
        "filename": "dynamic.zip",
        "url": "https://zenodo.org/api/records/17514376/files/dynamic.zip/content",
        "size_bytes": 1786104801,
        "size_gb": 1.66,
        "est_uncomp_gb": 1.88,
        "num_images": 12696,
    },
    "nadir": {
        "filename": "nadir.zip",
        "url": "https://zenodo.org/api/records/17514376/files/nadir.zip/content",
        "size_bytes": 3732503007,
        "size_gb": 3.48,
        "est_uncomp_gb": 4.06,
        "num_images": 27066,
    },
    "oblique": {
        "filename": "oblique.zip",
        "url": "https://zenodo.org/api/records/17514376/files/oblique.zip/content",
        "size_bytes": 4192186273,
        "size_gb": 3.90,
        "est_uncomp_gb": 3.75,
        "num_images": 25040,
    }
}


def check_disk_space(target_dir: Path, requested_trajectories: list):
    """
    Evaluates available filesystem space and verifies whether the sequential
    download + extraction pipeline will fit comfortably without disk exhaustion.
    """
    stat = shutil.disk_usage(target_dir.anchor if target_dir.exists() else "/")
    free_gb = stat.free / (1024 ** 3)
    
    total_uncomp_gb = sum(ARCHIVES[t]["est_uncomp_gb"] for t in requested_trajectories)
    max_zip_gb = max(ARCHIVES[t]["size_gb"] for t in requested_trajectories)
    peak_required_gb = total_uncomp_gb + max_zip_gb + 1.0  # 1 GB safety margin

    logger.info("=" * 70)
    logger.info("DISK SPACE PRE-FLIGHT CHECK")
    logger.info("=" * 70)
    logger.info(f"Target filesystem:       {target_dir.anchor}")
    logger.info(f"Available free space:    {free_gb:.2f} GB")
    logger.info(f"Selected trajectories:   {requested_trajectories}")
    logger.info(f"Total uncompressed size: ~{total_uncomp_gb:.2f} GB")
    logger.info(f"Peak required space:     ~{peak_required_gb:.2f} GB (with sequential extract & delete)")

    if free_gb < peak_required_gb:
        logger.warning("-" * 70)
        logger.warning(f"CAUTION: Available space ({free_gb:.2f} GB) is close to or below peak requirement ({peak_required_gb:.2f} GB)!")
        logger.warning("Recommended actions to free up disk space immediately:")
        logger.warning("  1. Purge pip cache:          pip cache purge")
        logger.warning("  2. Clean conda package cache: conda clean -p -y")
        logger.warning("  3. Download trajectories individually, e.g.:")
        logger.warning("     python scripts/download_stereo_lunar.py --trajectories dynamic")
        logger.warning("-" * 70)
        return False
    else:
        logger.info(f"[✓] Sufficient disk space verified ({free_gb:.2f} GB free >= {peak_required_gb:.2f} GB peak).")
        return True


def is_valid_zip(zip_path: Path) -> bool:
    """Checks if a zip file exists and all internal files pass CRC-32 verification."""
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        return False
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zf:
            return zf.testzip() is None
    except Exception:
        return False


def download_file(url: str, dest_path: Path, expected_bytes: int):
    """
    Downloads a large file with resumption support using aria2c (fastest, multi-connection)
    or curl/requests fallback. Verifies CRC integrity before returning.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # 0. Check if archive is already valid on disk
    if is_valid_zip(dest_path):
        logger.info(f"[✓] {dest_path.name} is already completely downloaded and CRC-verified. Skipping download.")
        return True

    # 1. Try aria2c with automatic retry on 504 / connection timeouts
    aria2_bin = shutil.which("aria2c")
    if aria2_bin:
        cmd = [
            aria2_bin,
            "--continue=true",
            "--max-connection-per-server=8",
            "--split=8",
            "--min-split-size=1M",
            "--max-tries=50",
            "--retry-wait=5",
            "--timeout=60",
            "--connect-timeout=30",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            "--dir", str(dest_path.parent),
            "--out", dest_path.name,
            url
        ]
        logger.info(f"Downloading with aria2c (8 connections, auto-retry): {url}")
        res = subprocess.run(cmd)
        if is_valid_zip(dest_path):
            logger.info(f"[✓] Download and CRC verification passed for {dest_path.name}.")
            return True
        logger.warning("aria2c finished but zip CRC verification did not pass. Trying curl fallback...")

    # 2. Try curl with resume
    curl_bin = shutil.which("curl")
    if curl_bin:
        cmd = [
            curl_bin,
            "-L",
            "-C", "-",
            "--retry", "10",
            "--retry-delay", "5",
            "-o", str(dest_path),
            url
        ]
        logger.info(f"Downloading with curl (resumable): {url}")
        subprocess.run(cmd)
        if is_valid_zip(dest_path):
            logger.info(f"[✓] Download and CRC verification passed for {dest_path.name}.")
            return True
        logger.warning("curl finished but zip CRC verification did not pass. Falling back to python requests...")

    # 3. Python requests fallback with Range resumption
    import requests
    from tqdm import tqdm

    headers = {}
    downloaded = 0
    if dest_path.exists():
        downloaded = dest_path.stat().st_size
        if downloaded < expected_bytes:
            headers['Range'] = f'bytes={downloaded}-'

    mode = 'ab' if downloaded > 0 and downloaded < expected_bytes else 'wb'
    if downloaded >= expected_bytes and not is_valid_zip(dest_path):
        # File has full size but corrupt contents (e.g. sparse aria2 allocation), restart
        mode = 'wb'
        downloaded = 0
        headers.pop('Range', None)

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0)) + downloaded
        with open(dest_path, mode) as f, tqdm(
            desc=dest_path.name,
            initial=downloaded,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                size = f.write(chunk)
                bar.update(size)

    if not is_valid_zip(dest_path):
        raise RuntimeError(f"Zip archive {dest_path.name} failed CRC validation after download.")

    logger.info(f"[✓] Download and CRC verification passed for {dest_path.name}.")
    return True


def extract_and_organize_archive(zip_path: Path, traj_name: str, scenes_root: Path, blocks_per_scene: int = 20):
    """
    Extracts archive sequentially and organizes frames into distinct physical scene batches.
    Skips any files that have already been extracted.
    Immediately deletes the zip archive upon successful extraction to free up disk space.
    
    Structure created:
      scenes/{traj_name}_scene_{scene_idx:04d}/im_{idx:05d}.{jpg,exr,npz}
    """
    import zipfile

    logger.info(f"Extracting {zip_path.name} into scene directories (blocks_per_scene={blocks_per_scene})...")
    scenes_root.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    skipped_count = 0
    with zipfile.ZipFile(zip_path, 'r') as zf:
        infolist = zf.infolist()
        for member in infolist:
            filename = member.filename
            if member.is_dir():
                continue
            
            # File name pattern: <traj>/im_XXXXX.ext
            parts = filename.strip('/').split('/')
            basename = parts[-1]
            if not basename.startswith("im_"):
                continue

            stem = basename.split('.')[0]
            try:
                frame_idx = int(stem.replace("im_", ""))
            except ValueError:
                continue

            # Group every 6 frames into a stereo-illumination station block
            block_idx = frame_idx // 6
            scene_idx = block_idx // blocks_per_scene
            scene_name = f"{traj_name}_scene_{scene_idx:04d}"

            dest_dir = scenes_root / scene_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / basename

            # Skip if already extracted with non-zero size
            if dest_file.exists() and dest_file.stat().st_size > 0:
                skipped_count += 1
                continue

            with zf.open(member) as src, open(dest_file, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_count += 1

    logger.info(f"[✓] Extracted {extracted_count} new file(s) ({skipped_count} already existed) for trajectory '{traj_name}'.")

    # Immediate deletion of the zip file to free space before downloading the next archive
    logger.info(f"Removing archive {zip_path.name} to reclaim disk space...")
    zip_path.unlink(missing_ok=True)
    aria2_ctrl = Path(str(zip_path) + ".aria2")
    if aria2_ctrl.exists():
        aria2_ctrl.unlink(missing_ok=True)
    logger.info(f"[✓] Archive and temporary files removed. Reclaimed space.")


def build_split_manifest(scenes_root: Path, out_dir: Path, val_ratio: float = 0.10, test_ratio: float = 0.10, seed: int = 42):
    """
    Scans all scene directories, identifies stereo pairs and illumination variation pairs,
    and generates the official split.npz manifest with strict terrain-disjoint train/val/test splits.
    """
    logger.info("=" * 70)
    logger.info("BUILDING SPLIT MANIFEST (split.npz)")
    logger.info("=" * 70)

    # 1. Discover scenes
    scene_dirs = sorted([d for d in scenes_root.iterdir() if d.is_dir() and "_scene_" in d.name])
    if not scene_dirs:
        raise RuntimeError(f"No scene directories matching '*_scene_*' found in {scenes_root}")

    scenes_list = [d.name for d in scene_dirs]
    scene_name_to_idx = {name: idx for idx, name in enumerate(scenes_list)}

    all_images_set = set()
    all_pairs = []

    for s_idx, scene_dir in enumerate(scene_dirs):
        # Find all image stems with valid triplets
        jpgs = sorted(scene_dir.glob("im_*.jpg"))
        for jpg in jpgs:
            stem = jpg.stem
            if (scene_dir / f"{stem}.exr").exists() and (scene_dir / f"{stem}.npz").exists():
                all_images_set.add(stem)

    images_list = sorted(list(all_images_set))
    image_name_to_idx = {name: idx for idx, name in enumerate(images_list)}

    # Build pairs within each scene using camera-extrinsic geometry:
    # 1. Genuine stereo pairs: consecutive (2k, 2k+1) -> score = 1.0
    # 2. Genuine zero-baseline pairs: same station (||pos_i - pos_j|| < 1.0m) -> score = 0.9 / 0.8
    for s_idx, scene_dir in enumerate(scene_dirs):
        scene_stems = sorted([
            f.stem for f in scene_dir.glob("im_*.jpg")
            if (scene_dir / f"{f.stem}.exr").exists() and (scene_dir / f"{f.stem}.npz").exists()
        ])
        if len(scene_stems) < 2:
            continue

        c2ws = {s: np.load(scene_dir / f"{s}.npz")["cam2world"][:3, 3] for s in scene_stems}

        # 1. Genuine stereo pairs (consecutive even-odd)
        for i in range(0, len(scene_stems) - 1, 2):
            s1, s2 = scene_stems[i], scene_stems[i+1]
            idx1, idx2 = int(s1.replace("im_", "")), int(s2.replace("im_", ""))
            if idx2 == idx1 + 1:
                all_pairs.append([s_idx, image_name_to_idx[s1], image_name_to_idx[s2], 1.0])

        # 2. Genuine zero-baseline pairs (identical camera position under different illumination)
        # Search within a local station window of 6 frames
        for i in range(len(scene_stems)):
            for j in range(i + 1, min(len(scene_stems), i + 6)):
                s1, s2 = scene_stems[i], scene_stems[j]
                dist = np.linalg.norm(c2ws[s1] - c2ws[s2])
                if dist < 1.0:
                    idx_diff = abs(int(s2.replace("im_", "")) - int(s1.replace("im_", "")))
                    score = 0.9 if idx_diff <= 2 else 0.8
                    all_pairs.append([s_idx, image_name_to_idx[s1], image_name_to_idx[s2], score])

    all_pairs = np.array(all_pairs, dtype=np.float32)
    logger.info(f"Total discovered scenes: {len(scenes_list)}")
    logger.info(f"Total valid images:      {len(images_list)}")
    logger.info(f"Total generated pairs:   {len(all_pairs)}")

    # 2. Strict Scene-Disjoint Partitioning
    # Group scenes by trajectory type to guarantee balanced representation in train/val/test
    rng = np.random.RandomState(seed)
    traj_types = set(s.split('_scene_')[0] for s in scenes_list)
    train_sids = set()
    val_sids = set()
    test_sids = set()

    for traj in sorted(list(traj_types)):
        traj_sids = [idx for idx, name in enumerate(scenes_list) if name.startswith(traj)]
        perm = rng.permutation(traj_sids)
        n_traj = len(perm)
        n_val = max(1, int(n_traj * val_ratio))
        n_test = max(1, int(n_traj * test_ratio))
        n_train = n_traj - n_val - n_test

        v_set = set(perm[:n_val])
        te_set = set(perm[n_val:n_val + n_test])
        tr_set = set(perm[n_val + n_test:])
        if len(tr_set) == 0:
            tr_set = v_set

        val_sids.update(v_set)
        test_sids.update(te_set)
        train_sids.update(tr_set)

    splits = []
    for p in all_pairs:
        sid = int(p[0])
        if sid in train_sids:
            splits.append('train')
        elif sid in val_sids:
            splits.append('val')
        else:
            splits.append('test')
    splits = np.array(splits)

    # Validate disjointness
    train_scenes = set(scenes_list[sid] for sid in train_sids)
    val_scenes = set(scenes_list[sid] for sid in val_sids)
    test_scenes = set(scenes_list[sid] for sid in test_sids)
    assert train_scenes.isdisjoint(val_scenes), f"Data leakage detected: {train_scenes & val_scenes}"
    assert train_scenes.isdisjoint(test_scenes), f"Data leakage detected: {train_scenes & test_scenes}"
    assert val_scenes.isdisjoint(test_scenes), f"Data leakage detected: {val_scenes & test_scenes}"

    # 3. Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = out_dir / "split.npz"
    np.savez(
        str(split_path),
        scenes=np.array(scenes_list),
        images=np.array(images_list),
        pairs=all_pairs,
        split=splits
    )
    logger.info(f"[✓] Saved master manifest: {split_path}")

    # Text list files for MultiSceneDataModule (points to split.npz master manifest)
    for sp, s_set in [('train', train_scenes), ('val', val_scenes), ('test', test_scenes)]:
        list_file = out_dir / f"{sp}_list.txt"
        with open(list_file, 'w') as f:
            f.write("split.npz\n")
        
        # Also save detailed scene list for inspection
        scenes_file = out_dir / f"{sp}_scenes.txt"
        with open(scenes_file, 'w') as f:
            for sname in sorted(list(s_set)):
                f.write(f"{sname}\n")
        logger.info(f"[✓] Wrote {list_file} and {scenes_file} ({len(s_set)} scenes)")

    # Report JSON
    report = {
        'total_scenes': len(scenes_list),
        'total_images': len(images_list),
        'total_pairs': len(all_pairs),
        'train_scenes_count': len(train_scenes),
        'train_pairs_count': int(np.sum(splits == 'train')),
        'val_scenes_count': len(val_scenes),
        'val_pairs_count': int(np.sum(splits == 'val')),
        'test_scenes_count': len(test_scenes),
        'test_pairs_count': int(np.sum(splits == 'test')),
        'trajectories_included': sorted(list(traj_types)),
        'scene_leakage_detected': False
    }
    report_file = out_dir / "split_manifest_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"[✓] Saved manifest report: {report_file}")

    return report


def verify_dataset_loader(out_dir: Path):
    """
    Executes a verification pass by loading dataset samples with StereoLunarDataset.
    """
    logger.info("=" * 70)
    logger.info("VERIFYING DATASET COMPATIBILITY")
    logger.info("=" * 70)

    try:
        from EfficientLoFTR.src.datasets.stereo_lunar import StereoLunarDataset
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "EfficientLoFTR"))
        from src.datasets.stereo_lunar import StereoLunarDataset

    scenes_dir = out_dir / "scenes"
    split_file = out_dir / "split.npz"

    for sp in ['train', 'val', 'test']:
        ds = StereoLunarDataset(str(scenes_dir), str(split_file), split=sp, img_resize=512)
        logger.info(f"Loaded {sp} split: {len(ds)} pairs across {len(ds.assigned_scenes)} scenes.")
        if len(ds) > 0:
            sample = ds[0]
            assert 'image0' in sample and sample['image0'].shape == (1, 512, 512)
            assert 'depth0' in sample and sample['depth0'].shape == (512, 512)
            assert 'T_0to1' in sample and sample['T_0to1'].shape == (4, 4)
            logger.info(f"  Sample 0 verified: T_0to1 norm = {np.linalg.norm(sample['T_0to1'][:3, 3]):.2f}m")

    logger.info("[✓] Dataset verification PASSED across all splits!")


def main():
    parser = argparse.ArgumentParser(description="StereoLunar Full Dataset Downloader & Preparer")
    parser.add_argument(
        "--trajectories", nargs="+", default=["dynamic", "nadir", "oblique"],
        choices=["dynamic", "nadir", "oblique"],
        help="Trajectory subsets to download (default: all three: dynamic nadir oblique)"
    )
    parser.add_argument(
        "--out_dir", type=str,
        default=str(Path(__file__).resolve().parent.parent / "EfficientLoFTR" / "data" / "stereo_lunar"),
        help="Target directory for processed dataset"
    )
    parser.add_argument(
        "--tmp_download_dir", type=str, default=None,
        help="Temporary download directory for zip archives (default: out_dir/downloads)"
    )
    parser.add_argument(
        "--blocks_per_scene", type=int, default=20,
        help="Number of 6-frame station blocks per scene batch (default: 20 blocks = 120 images)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Perform pre-flight checks and URL reachability verification without downloading"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    download_dir = Path(args.tmp_download_dir).resolve() if args.tmp_download_dir else out_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir = out_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pre-flight disk space check
    check_disk_space(out_dir, args.trajectories)

    if args.dry_run:
        logger.info("[Dry Run] Checking URL reachability for requested trajectories...")
        import requests
        for traj in args.trajectories:
            meta = ARCHIVES[traj]
            r = requests.head(meta["url"], timeout=10)
            logger.info(f"  {traj:7s}: Status {r.status_code} | Size: {meta['size_gb']:.2f} GB ({meta['num_images']} frames)")
        logger.info("[Dry Run] Pre-flight checks complete. Exiting without downloading.")
        return

    # 2. Sequential download, extraction, and cleanup
    for traj in args.trajectories:
        meta = ARCHIVES[traj]
        zip_path = download_dir / meta["filename"]

        # Check if already fully extracted
        existing_scenes = sorted(list(scenes_dir.glob(f"{traj}_scene_*")))
        total_existing_imgs = sum(len(list(d.glob("im_*.jpg"))) for d in existing_scenes)
        if total_existing_imgs >= meta["num_images"]:
            logger.info(f"[✓] Trajectory '{traj}' is ALREADY fully extracted ({total_existing_imgs}/{meta['num_images']} images). Skipping.")
            continue
        elif total_existing_imgs > 0:
            logger.info(f"Trajectory '{traj}' partially extracted ({total_existing_imgs}/{meta['num_images']} images). Resuming extraction...")

        logger.info("-" * 70)
        logger.info(f"PROCESSING TRAJECTORY: {traj.upper()} ({meta['size_gb']:.2f} GB)")
        logger.info("-" * 70)

        # Step A: Download
        download_file(meta["url"], zip_path, meta["size_bytes"])

        # Step B: Extract and delete zip immediately
        extract_and_organize_archive(zip_path, traj, scenes_dir, blocks_per_scene=args.blocks_per_scene)

    # Clean up empty download directory
    if download_dir.exists() and not list(download_dir.iterdir()):
        download_dir.rmdir()

    # 3. Build master manifest (split.npz)
    build_split_manifest(scenes_dir, out_dir)

    # 4. Verify loader compatibility
    verify_dataset_loader(out_dir)

    logger.info("=" * 70)
    logger.info("DATASET PREPARATION COMPLETE!")
    logger.info(f"Data root: {scenes_dir}")
    logger.info(f"Manifest:  {out_dir / 'split.npz'}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
