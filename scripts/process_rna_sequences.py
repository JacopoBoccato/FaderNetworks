#!/usr/bin/env python3
import sys
import json
from pathlib import Path
import argparse
import torch

# -------------------------------
# Constants / RNA alphabet
# -------------------------------
RNA_ALPHABET = ["A", "C", "G", "U"]
RNA_TO_IDX = {b: i for i, b in enumerate(RNA_ALPHABET)}

# -------------------------------
# One-hot encoder
# -------------------------------
def onehot_encode_rna(seqs, seq_len=None):
    """
    One-hot encode RNA sequences (A,C,G,U). Unknowns/gaps become all-zeros.
    Returns tensor of shape (N, 4, L)
    """
    if not seqs:
        raise ValueError("No sequences provided to onehot_encode_rna().")
    max_len = seq_len or max(len(s) for s in seqs)
    out = torch.zeros((len(seqs), 4, max_len), dtype=torch.float32)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s[:max_len]):
            idx = RNA_TO_IDX.get(ch, None)
            if idx is not None:
                out[i, idx, j] = 1.0
    return out

# -------------------------------
# Robust line parser
# -------------------------------
def parse_line(line: str):
    """
    Accepts lines like:
        ID<TAB>SEQ<TAB>TAXONOMY
    Falls back to splitting on ANY whitespace if tabs are missing.
    Returns (id, seq, taxonomy_string) or None if malformed.
    """
    line = line.strip()
    if not line:
        return None
    parts = line.split("\t")
    if len(parts) < 3:
        # fall back to generic split with max 3 fields
        parts = line.split(None, 2)  # split on any whitespace
        if len(parts) < 3:
            return None
    return parts[0].strip(), parts[1].strip(), parts[2].strip()

# -------------------------------
# Main dataset creator
# -------------------------------
def create_rna_dataset(file_path: str, outdir: str, seq_len: int = None):
    """
    Build RNA dataset from a tab-separated file:
      ID \t RNA_SEQUENCE \t TAXONOMY

    - Converts any T->U (safe; leaves U as U)
    - Drops entries with missing taxonomy or missing phylum
    - Labels (phylum):
        Bacillota -> [0, 1]
        Other     -> [1, 0]
    - One-hot encodes to (N, 4, L)
    - Saves:
        rna_sequences_<L>.pth
        rna_labels_<L>.pth
        rna_taxonomy.json
    """
    src = Path(file_path)
    dst = Path(outdir)
    dst.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    kept_ids, kept_seqs, kept_tax = [], [], []
    n_total = n_malformed = n_no_tax = n_no_phylum = 0

    with src.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            n_total += 1
            parsed = parse_line(line)
            if parsed is None:
                n_malformed += 1
                continue
            rec_id, rec_seq, rec_tax = parsed

            # Normalize sequence: uppercase, DNA->RNA
            rec_seq = rec_seq.upper().replace("T", "U")

            # Allow standard gap/unknown chars without crashing; they encode to zeros
            # Remove spaces just in case
            rec_seq = rec_seq.replace(" ", "")

            # Parse taxonomy
            if not rec_tax:
                n_no_tax += 1
                continue

            tax_levels = [t.strip() for t in rec_tax.split(";") if t.strip()]
            if not tax_levels:
                n_no_tax += 1
                continue

            keys = ["kingdom", "phylum", "class", "order", "family", "genus"]
            taxonomy = {}
            for i, level in enumerate(tax_levels[:len(keys)]):
                taxonomy[keys[i]] = level

            if not taxonomy.get("phylum", ""):
                n_no_phylum += 1
                continue

            kept_ids.append(rec_id)
            kept_seqs.append(rec_seq)
            kept_tax.append(taxonomy)

    print(f"[INFO] Read:           {n_total}")
    print(f"[INFO] Malformed:      {n_malformed}")
    print(f"[INFO] No taxonomy:    {n_no_tax}")
    print(f"[INFO] No phylum:      {n_no_phylum}")
    print(f"[INFO] Kept valid:     {len(kept_seqs)}")

    if not kept_seqs:
        raise RuntimeError("No valid sequences with phylum were found after filtering.")

    # One-hot encode
    seq_tensor = onehot_encode_rna(kept_seqs, seq_len=seq_len)
    L = seq_tensor.shape[-1]
    print(f"[INFO] One-hot shape:  {tuple(seq_tensor.shape)}  (N, 4, {L})")

    # Labels by phylum
    labels = []
    for tax in kept_tax:
        if tax.get("phylum") == "Bacillota":
            labels.append([0, 1])
        else:
            labels.append([1, 0])
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    print(f"[INFO] Label shape:    {tuple(label_tensor.shape)}  (N, 2)")

    # Save outputs
    seq_out = dst / f"rna_sequences_{L}.pth"
    lab_out = dst / f"rna_labels_{L}.pth"
    meta_out = dst / "rna_taxonomy.json"

    torch.save(seq_tensor, seq_out)
    torch.save(label_tensor, lab_out)
    with meta_out.open("w", encoding="utf-8") as f:
        json.dump([{"id": i, "taxonomy": t} for i, t in zip(kept_ids, kept_tax)], f, indent=2, ensure_ascii=False)

    print("[OK] Saved to:")
    print(f"  - {seq_out}  {tuple(seq_tensor.shape)}")
    print(f"  - {lab_out}  {tuple(label_tensor.shape)}")
    print(f"  - {meta_out}  ({len(kept_ids)} entries)")

    # Quick distribution check (unique phyla)
    phyla = {}
    for t in kept_tax:
        ph = t.get("phylum", "UNKNOWN")
        phyla[ph] = phyla.get(ph, 0) + 1
    top = sorted(phyla.items(), key=lambda x: x[1], reverse=True)[:10]
    print("[INFO] Top phyla (up to 10):")
    for ph, cnt in top:
        print(f"    {ph}: {cnt}")

    return seq_tensor, label_tensor

# -------------------------------
# CLI entrypoint
# -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Process RNA sequences with taxonomy into tensors + labels.")
    ap.add_argument("--file", required=True, help="Path to input file (ID<TAB>SEQ<TAB>TAXONOMY)")
    ap.add_argument("--outdir", required=True, help="Output directory (will be created)")
    ap.add_argument("--seq_len", type=int, default=None, help="Fixed length (pad/truncate). If None, uses max length.")
    args = ap.parse_args()

    print(f"[RUN] file={args.file}")
    print(f"[RUN] outdir={args.outdir}")
    print(f"[RUN] seq_len={args.seq_len}")

    create_rna_dataset(args.file, args.outdir, args.seq_len)

if __name__ == "__main__":
    main()

