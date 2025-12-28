import sys
import random
from pathlib import Path

def split_data(file_path, train_pct=0.8, dev_pct=0.1, test_pct=0.1):
    # 1. Validation
    if not (0.999 < (train_pct + dev_pct + test_pct) < 1.001):
        print(f"Error: Percentages sum to {train_pct + dev_pct + test_pct}, must be 1.0")
        return

    # 2. Load and Shuffle
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File {file_path} not found.")
        return

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    random.shuffle(lines)
    total = len(lines)

    # 3. Calculate split indices
    train_end = int(total * train_pct)
    dev_end = train_end + int(total * dev_pct)

    splits = {
        "_train": lines[:train_end],
        "_dev": lines[train_end:dev_end],
        "_test": lines[dev_end:]
    }

    # 4. Save files
    for suffix, data in splits.items():
        # keeps original extension, e.g., names.txt -> names_train.txt
        out_path = path.parent / f"{path.stem}{suffix}{path.suffix}"
        with open(out_path, 'w', encoding='utf-8') as f:
            f.writelines(data)
        print(f"Created {out_path} ({len(data)} lines)")

if __name__ == "__main__":
    # Command line argument handling
    if len(sys.argv) < 2:
        print("Usage: python3 split_data.py <filename> [train_pct] [dev_pct] [test_pct]")
        sys.exit(1)

    fname = sys.argv[1]
    # Default to 80 10 10 if not provided
    t = float(sys.argv[2]) / 100 if len(sys.argv) > 2 else 0.8
    d = float(sys.argv[3]) / 100 if len(sys.argv) > 3 else 0.1
    s = float(sys.argv[4]) / 100 if len(sys.argv) > 4 else 0.1

    split_data(fname, t, d, s)