import pandas as pd
import zipfile
from pathlib import Path


def save_submission(recs_df: pd.DataFrame, filename: str):
    """
    Saves recommendations in Codabench submission format.

    Parameters
    ----------
    recs_df : pd.DataFrame
        Must contain columns ["user_id", "item_id"].

    filename : str
        Base filename (without .csv)
        Example: "ease_baseline" → saves ease_baseline.csv and ease_baseline.csv.zip
    """

    # Ensure only required columns
    if not {"user_id", "item_id"}.issubset(recs_df.columns):
        raise ValueError("DataFrame must contain columns 'user_id' and 'item_id'.")

    recs_df = recs_df[["user_id", "item_id"]]

    # Define output directory at project root: `results/submissions`
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "results" / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw CSV
    csv_path = output_dir / f"{filename}.csv"
    recs_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Save zipped CSV for Codabench
    zip_path = Path(str(csv_path) + ".zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(str(csv_path), arcname=csv_path.name)

    print(f"Saved ZIP for Codabench: {zip_path}")