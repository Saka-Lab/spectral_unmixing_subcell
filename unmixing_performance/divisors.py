import pandas as pd
from pathlib import Path
import re
from collections import defaultdict

def extract_divisors_from_dir(matrix_dir):
    """
    Extracts divisors from all CSV matrices in a directory.
    """

    matrix_dir = Path(matrix_dir)

    if not matrix_dir.is_dir():
        raise ValueError(f"Not a directory: {matrix_dir}")

    divisors_by_reference = defaultdict(dict)
    divisors_by_group = defaultdict(dict)

    # Parse <left>_vs_<right> from filename
    pattern = re.compile(r'_(.*?)_vs_(.*?)_')

    for csv_path in matrix_dir.glob("*.csv"):
        match = pattern.search(csv_path.name)
        if not match:
            continue

        left, right = match.groups()

        df = pd.read_csv(csv_path, index_col=0)
        divisor = float(df.iloc[0, 0])

        # Reference-based (raw/group/full vs GrX)
        if left in {"raw", "group", "full"} and right.startswith("Gr"):
            divisors_by_reference[left][right] = divisor

        # Group-based (GrX vs GrY)
        elif left.startswith("Gr") and right.startswith("Gr"):
            divisors_by_group[left][right] = divisor

    return dict(divisors_by_reference), dict(divisors_by_group)