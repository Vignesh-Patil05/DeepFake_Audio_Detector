import argparse
import csv
import os
import random
import subprocess
from collections import defaultdict


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def iter_meta_rows(meta_csv: str):
    """
    meta_data.csv rows look like:
    source,target1,target2,method,category,type,race,gender,filename,path,
    """
    with open(meta_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            # Some rows may have a trailing empty cell due to final comma.
            if row and row[-1] == "":
                row = row[:-1]
            if len(row) < 10:
                continue
            yield {
                "source": row[0],
                "target1": row[1],
                "target2": row[2],
                "method": row[3],
                "category": row[4],
                "type": row[5],
                "race": row[6],
                "gender": row[7],
                "filename": row[8],
                "path": row[9],
            }


def build_mp4_path(dataset_root: str, r: dict) -> str:
    # Observed on disk: <root>/<type>/<race>/<gender>/<source>/<filename>
    return os.path.join(dataset_root, r["type"], r["race"], r["gender"], r["source"], r["filename"])


def is_fake_audio(r: dict) -> bool:
    # Fake audio categories are those whose "type" contains "FakeAudio"
    return "FakeAudio" in r["type"]


def run_ffmpeg(ffmpeg_exe: str, inp: str, out_wav: str, sr: int = 16000) -> None:
    ensure_dir(os.path.dirname(out_wav))
    # -vn: disable video, mono, resample
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        inp,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "wav",
        out_wav,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def split_paths(items, seed: int, train_ratio: float, dev_ratio: float):
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    train = items[:n_train]
    dev = items[n_train : n_train + n_dev]
    test = items[n_train + n_dev :]
    return train, dev, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, help="Path to FakeAVCeleb_v1.2 folder")
    ap.add_argument("--out_root", required=True, help="Where to write extracted wavs")
    ap.add_argument("--out_lists_dir", required=True, help="Where to write train/dev/test .list files")
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    ap.add_argument("--max_files", type=int, default=0, help="0 = no limit, else process only first N rows")
    args = ap.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    meta_csv = os.path.join(dataset_root, "meta_data.csv")
    if not os.path.isfile(meta_csv):
        raise FileNotFoundError(f"meta_data.csv not found at {meta_csv}")

    try:
        import imageio_ffmpeg  # type: ignore
    except Exception as e:
        raise RuntimeError("imageio-ffmpeg is required. Install with: pip install imageio-ffmpeg") from e

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Map each fake-audio method to a stable integer label (1..K). Real is 0.
    method_to_label = {}
    next_label = 1

    extracted = []  # list of (wav_path_abs, label_int, lblname)
    seen_missing = 0
    processed = 0

    for r in iter_meta_rows(meta_csv):
        if args.max_files and processed >= args.max_files:
            break
        processed += 1

        mp4_path = build_mp4_path(dataset_root, r)
        if not os.path.isfile(mp4_path):
            seen_missing += 1
            continue

        fake = is_fake_audio(r)
        if fake:
            method = r["method"] or "fake"
            if method not in method_to_label:
                method_to_label[method] = next_label
                next_label += 1
            label = method_to_label[method]
            lblname = method
        else:
            label = 0
            lblname = "real"

        # Output wav path mirrors the dataset structure but ends with .wav
        rel_dir = os.path.relpath(os.path.dirname(mp4_path), dataset_root)
        base = os.path.splitext(os.path.basename(mp4_path))[0] + ".wav"
        wav_path = os.path.join(os.path.abspath(args.out_root), rel_dir, base)

        if not os.path.isfile(wav_path):
            run_ffmpeg(ffmpeg_exe, mp4_path, wav_path, sr=args.sr)

        extracted.append((os.path.abspath(wav_path), label, lblname))

    if not extracted:
        raise RuntimeError("No audio extracted. Check dataset_root path and meta_data.csv consistency.")

    # Stratified split by (label)
    by_label = defaultdict(list)
    for item in extracted:
        by_label[item[1]].append(item)

    train, dev, test = [], [], []
    for label, items in by_label.items():
        tr, dv, te = split_paths(items, args.seed + int(label), args.train_ratio, args.dev_ratio)
        train.extend(tr)
        dev.extend(dv)
        test.extend(te)

    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)

    ensure_dir(args.out_lists_dir)
    out_train = os.path.join(args.out_lists_dir, "fakeavceleb_train.list")
    out_dev = os.path.join(args.out_lists_dir, "fakeavceleb_dev.list")
    out_test = os.path.join(args.out_lists_dir, "fakeavceleb_test.list")

    def write_list(path, items):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for wav_path, label, lblname in items:
                f.write(f"{wav_path},{label},{lblname}\n")

    write_list(out_train, train)
    write_list(out_dev, dev)
    write_list(out_test, test)

    print("Prepared FakeAVCeleb audio.")
    print("dataset_root:", dataset_root)
    print("out_root:", os.path.abspath(args.out_root))
    print("lists:", os.path.abspath(args.out_lists_dir))
    print("counts:", {"train": len(train), "dev": len(dev), "test": len(test)})
    print("fake_method_labels:", method_to_label)
    if seen_missing:
        print("missing_mp4_files:", seen_missing)


if __name__ == "__main__":
    main()

