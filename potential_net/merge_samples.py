"""
Merge multiple potential-net samples pickle files into one, deduplicating
exact duplicates.

Usage:
    python -m potential_net.merge_samples output/potential_samples.pkl output/potential_samples_low.pkl --output output/potential_samples.pkl
"""

import argparse
import os
import pickle


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge potential samples pickle files")
    parser.add_argument("inputs", nargs="+", help="Input .pkl files to merge")
    parser.add_argument("--output", type=str, default="output/potential_samples.pkl",
                        help="Output path")
    args = parser.parse_args()

    merged = []
    for p in args.inputs:
        with open(p, "rb") as f:
            batch = pickle.load(f)
        merged.extend(batch)
        print(f"  {p}: {len(batch):,} samples")

    seen = set()
    unique = []
    for s in merged:
        key = (s[0], s[1], round(s[2], 9), round(s[3], 9))
        if key not in seen:
            seen.add(key)
            unique.append(s)

    print(f"  merged: {len(unique):,} unique (removed {len(merged) - len(unique):,} dups)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(unique, f)
    print(f"  saved: {args.output}")


if __name__ == "__main__":
    main()
