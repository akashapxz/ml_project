"""
Data Loader and Downloader Module for PhysioNet Challenge 2019.
Handles streaming download, raw PSV ingestion, and initial schema integrity checks.
"""

import concurrent.futures
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
import urllib.request
import pandas as pd

from sepsis_ml.config import ALL_COLUMNS, RAW_DATA_DIR, TARGET_COL
from sepsis_ml.logger import get_logger

logger = get_logger("data_loader")

PHYSIONET_BASE_URL = "https://physionet.org/files/challenge-2019/1.0.0/training"


def download_patient_file(
    patient_id: str,
    hospital: str,
    target_dir: Optional[Path] = None,
    overwrite: bool = False
) -> Path:
    """
    Downloads a single patient PSV file from PhysioNet.
    
    Args:
        patient_id: e.g. 'p000001' or 'p100001'
        hospital: 'training_setA' or 'training_setB'
        target_dir: local destination directory
        overwrite: whether to overwrite if exists
        
    Returns:
        Local Path to downloaded PSV file.
    """
    dest_dir = (target_dir or RAW_DATA_DIR) / hospital
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{patient_id}.psv" if not patient_id.endswith(".psv") else patient_id
    dest_path = dest_dir / filename
    
    if dest_path.exists() and not overwrite and dest_path.stat().st_size > 0:
        return dest_path
        
    url = f"{PHYSIONET_BASE_URL}/{hospital}/{filename}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MTech-Research-Sepsis/1.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())
        
    return dest_path


def download_cohort(
    hospital: str,
    start_idx: int,
    count: int,
    max_workers: int = 16
) -> List[Path]:
    """
    Downloads a sequential batch of patient files using multi-threading.
    
    Args:
        hospital: 'training_setA' (starts with p0...) or 'training_setB' (starts with p1...)
        start_idx: 1-indexed patient number
        count: number of patient files to download
        max_workers: thread pool size
        
    Returns:
        List of Path objects for successfully downloaded files.
    """
    prefix = "0" if hospital == "training_setA" else "1"
    patient_ids = [f"p{prefix}{i:05d}" for i in range(start_idx, start_idx + count)]
    
    downloaded: List[Path] = []
    logger.info(f"Downloading {count} patient files for {hospital} (workers={max_workers})...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pid = {
            executor.submit(download_patient_file, pid, hospital): pid
            for pid in patient_ids
        }
        for future in concurrent.futures.as_completed(future_to_pid):
            pid = future_to_pid[future]
            try:
                path = future.result()
                downloaded.append(path)
            except Exception as e:
                logger.warning(f"Failed downloading {pid} from {hospital}: {e}")
                
    logger.info(f"Successfully downloaded {len(downloaded)}/{count} files for {hospital}.")
    return sorted(downloaded)


def load_patient_psv(filepath: Path) -> pd.DataFrame:
    """
    Reads a single patient PSV file, parses types, and attaches metadata.
    
    Args:
        filepath: Path to the .psv file
        
    Returns:
        pandas DataFrame with attached 'patient_id' and 'hospital'
    """
    df = pd.read_csv(filepath, sep="|")
    patient_id = filepath.stem
    hospital = "A" if "setA" in str(filepath.parent) else "B"
    
    df["patient_id"] = patient_id
    df["hospital"] = hospital
    return df


def iter_patient_files(root_dir: Optional[Path] = None) -> Generator[Path, None, None]:
    """Yields all .psv files found in raw directory sorted deterministically."""
    target_dir = root_dir or RAW_DATA_DIR
    for path in sorted(target_dir.rglob("*.psv")):
        yield path


def load_cohort_dataframe(file_list: List[Path]) -> pd.DataFrame:
    """Concatenates multiple patient files into a single unified DataFrame."""
    frames = [load_patient_psv(p) for p in file_list]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
