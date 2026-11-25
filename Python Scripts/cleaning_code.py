"""
Prepare the ENSDF half-life dataset used for regression and EDA.

Input:  Extraction/extracted_data.csv
Output: EDA/cleaned_data.csv

Missing predictor values are intentionally retained. The downstream
model-training code is responsible for removing incomplete rows.
"""

from pathlib import Path

import numpy as np
import pandas as pd


EDA_DIR = Path.cwd()
EXTRACTION_DIR = EDA_DIR.parent / "Extraction"

INPUT_FILE = EXTRACTION_DIR / "extracted_data.csv"
OUTPUT_FILE = EDA_DIR / "cleaned_data.csv"

# Select columns to retain in the cleaned dataset.
OUTPUT_COLUMNS = [
    "element",
    "Z",
    "A",
    "N",
    "N/Z",
    "Q_beta_keV",
    "Q_alpha_keV",
    "S_n_keV",
    "S_p_keV",
    "log_half_life",
]


def clean_half_life_data(source: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the original cleaning transformations without dropping NaNs."""

    # Exclude stable, missing, and unparsed half-lives.
    finite = source.loc[source["half_life_status"] == "finite"].copy()

    # Preserve the original transformations.
    finite["log_half_life"] = np.log10(finite["half_life_seconds"])
    finite["N/Z"] = finite["N"] / finite["Z"]

    return finite.loc[:, OUTPUT_COLUMNS].copy()


def main() -> None:
    source = pd.read_csv(INPUT_FILE, keep_default_na=False, na_values=[""])
    cleaned = clean_half_life_data(source)

    cleaned.to_csv(OUTPUT_FILE, index=False)

    print(
        f"Saved {len(cleaned):,} records and "
        f"{len(cleaned.columns)} columns to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()



