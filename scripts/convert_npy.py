#!/usr/bin/env python3
"""
Batch-convert *.pkl.zip → *.npy containing x = diff(rpeaks)/fs.

Usage examples
--------------
# run in the directory that has many *.pkl.zip
python make_x_npys.py

# or specify folders and sampling rate
python make_x_npys.py --in-dir /path/to/zips --out-dir /path/to/out --fs 700

# overwrite existing npy files if needed
python make_x_npys.py --overwrite
"""
import argparse
import sys
from pathlib import Path
import zipfile
import pickle
import numpy as np

def pick_pkl_member(z: zipfile.ZipFile, base: str) -> str:
    """Choose the .pkl member inside the zip.
    Prefer '<base>.pkl'; otherwise take the first .pkl we find.
    """
    candidates = [n for n in z.namelist() if n.lower().endswith(".pkl")]
    if not candidates:
        raise FileNotFoundError("no .pkl found inside zip")
    exact = f"{base}.pkl"
    for n in candidates:
        # normalize path separators just in case
        if n.replace("\\", "/").endswith(exact):
            return n
    return candidates[0]

def load_rpeaks_from_zip(zip_path: Path, base: str) -> np.ndarray:
    """Open <zip_path>, load the pickle dict, and return rpeaks as np.ndarray."""
    with zipfile.ZipFile(zip_path, "r") as z:
        member = pick_pkl_member(z, base)
        with z.open(member, "r") as f:
            # many .pkl from older code need latin1 to load in py3
            data = pickle.load(f, encoding="latin1")  # type: ignore[arg-type]
    # Common key names; default to 'rpeaks'
    for key in ("rpeaks", "rpeak_idx", "rpeak_indices", "Rpeaks"):
        if key in data:
            rpeaks = data[key]
            break
    else:
        raise KeyError("Could not find 'rpeaks' in pickle (tried rpeaks/rpeak_idx/...).")
    return np.asarray(rpeaks)

def base_from_zipname(p: Path) -> str:
    """Return base name without the .pkl.zip suffix."""
    name = p.name
    if name.lower().endswith(".pkl.zip"):
        return name[:-8]  # remove '.pkl.zip'
    return p.stem

def process_one(zip_path: Path, out_dir: Path, fs: float, overwrite: bool) -> str:
    base = base_from_zipname(zip_path)
    out_path = out_dir / f"{base}.npy"
    if out_path.exists() and not overwrite:
        return f"SKIP  {zip_path.name}  (exists)"
    rpeaks = load_rpeaks_from_zip(zip_path, base)
    if rpeaks.ndim != 1 or rpeaks.size < 2:
        raise ValueError(f"{zip_path.name}: rpeaks must be 1D with >=2 points, got shape {rpeaks.shape}")
    x = np.diff(rpeaks).astype(np.float64) / float(fs)
    # Save compactly as float32 (change to float64 if you prefer)
    np.save(out_path, x.astype(np.float32))
    return f"DONE  {zip_path.name}  →  {out_path.name}  (len={x.size})"

def main():
    ap = argparse.ArgumentParser(description="Convert *.pkl.zip to *.npy of x = diff(rpeaks)/fs")
    ap.add_argument("--in-dir", type=Path, default=Path("."), help="Folder with .pkl.zip files")
    ap.add_argument("--out-dir", type=Path, default=Path("."), help="Where to write .npy files")
    ap.add_argument("--fs", type=float, default=700.0, help="Sampling rate used for rpeaks (default 700)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy files")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(args.in_dir.glob("*.pkl.zip"))
    if not zips:
        print(f"No .pkl.zip files found in {args.in_dir}", file=sys.stderr)
        sys.exit(1)

    ok = 0
    for zp in zips:
        try:
            msg = process_one(zp, args.out_dir, args.fs, args.overwrite)
            ok += msg.startswith("DONE")
            print(msg)
        except Exception as e:
            print(f"FAIL  {zp.name}: {e}", file=sys.stderr)
    print(f"\nFinished. Converted {ok}/{len(zips)} files. Output dir: {args.out_dir}")

if __name__ == "__main__":
    main()
