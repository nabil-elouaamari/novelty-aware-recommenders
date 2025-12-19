import pandas as pd
import zipfile
from pathlib import Path


def save_submission(recs_df: pd.DataFrame, filename: str) -> None:
    """
    Save recommendations in the Codabench submission format.

    This helper writes two files:
      - submissions/submissions/<filename>.csv
      - submissions/submissions/<filename>.csv.zip

    Parameters
    ----------
    recs_df : pd.DataFrame
        DataFrame with at least:
          - "user_id"
          - "item_id"
        Any additional columns are ignored.

    filename : str
        Base filename without extension. For example:
        "ease_lambda_300" will produce:
          - ease_lambda_300.csv
          - ease_lambda_300.csv.zip
    """
    # ensure only required columns
    if not {"user_id", "item_id"}.issubset(recs_df.columns):
        raise ValueError("DataFrame must contain columns 'user_id' and 'item_id'.")

    recs_df = recs_df[["user_id", "item_id"]]

    # define output directory at project root: submissions/submissions
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{filename}.csv"

    # write CSV directly into ZIP
    zip_path = output_dir / f"{filename}.csv.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        csv_bytes = recs_df.to_csv(index=False).encode("utf-8")
        zf.writestr(csv_path.name, csv_bytes)

    print(f"Saved ZIP for Codabench: {zip_path}")