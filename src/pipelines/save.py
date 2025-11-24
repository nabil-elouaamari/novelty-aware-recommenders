import pandas as pd
import zipfile
import os


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

    # Sort by user_id for reproducibility (not required but recommended)
    recs_df = recs_df[["user_id", "item_id"]].sort_values(["user_id"])

    # Save raw CSV
    csv_path = f"{filename}.csv"
    recs_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Save zipped CSV for Codabench
    zip_path = f"{csv_path}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=os.path.basename(csv_path))

    print(f"Saved ZIP for Codabench: {zip_path}")
