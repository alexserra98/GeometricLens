from pathlib import Path
from src.metrics.utils import TensorStorageManager
from src.common.global_vars import _OUTPUT_DIR_TRANSPOSED
from enum import Enum
from src.common.utils import read_config
from src.metrics.overlap import PointOverlap

import argparse
import importlib
import logging
from src.logging_utils.logging_config import setup_logging
import datetime
import os
import numpy as np
import tqdm

now = datetime.datetime.now()


def set_log_name():
    job_name = os.getenv("SLURM_JOB_ID")
    if job_name:
        return f"metrics_{job_name}"
    else:
        return f"metrics_{now.hour}{now.minute}{now.second}"


# save result in a temporary directory to avoid overwriting
_TMP_RESULT_DIR = Path(_OUTPUT_DIR_TRANSPOSED) / set_log_name()
_TMP_RESULT_DIR.mkdir(parents=True, exist_ok=True)

logger = setup_logging(_TMP_RESULT_DIR / set_log_name())


def main():
    tsm = TensorStorageManager(tensor_storage_location="npy")
    mask = np.load("assets/dev+validation_mask_20.npy")
    point_overlap = PointOverlap(queries={},
                                df = "",
                                tensor_storage = "",
                                variations = {"point_overlap": "None"},
                                parallel = True)
    for model_name in ["llama-3-8b", "mistral-1-7b", "llama-3-70b"]:
        logger.info(f"Computing metric for model {model_name}")
        try:
            path =  f"/orfeo/cephfs/scratch/area/ddoimo/" \
                f"open/geometric_lens/repo/results/" \
                f"finetuned_dev_val_balanced_20samples/" \
                f"evaluated_dev+validation/{model_name}/" \
                f"4epochs/epoch_0"
            path = Path(path)
            
            base_tensors, _, _ = tsm.retrieve_from_storage_npy_path("llama-3-8b", 0, path)
            base_tensors = base_tensors[mask]
            
            overlap_result = []
            
            pbar = tqdm.tqdm([1, 2, 3, 6, 12, 22, 42, 77, 144, 268])
            for step_number in pbar:
                pbar.set_description(f"Computing metric for model {model_name}")              
                path = f"/orfeo/cephfs/scratch/area/ddoimo/" \
                       f"open/geometric_lens/repo/results/" \
                       f"finetuned_dev_val_balanced_20samples/" \
                       f"evaluated_dev+validation/{model_name}/" \
                       f"4epochs/10ckpts/step_{step_number}"
                path = Path(path)
                chkp_point_tensor, _, _ = tsm.retrieve_from_storage_npy_path("llama-3-8b", 0, path)
                chkp_point_tensor = chkp_point_tensor[mask]
                output = point_overlap.parallel_compute(base_tensors,
                                                        chkp_point_tensor,
                                                        k=30)
                overlap_result.append(output)
                    
            overlap_result = np.stack(overlap_result)
            np.save(_TMP_RESULT_DIR / f"point_overlap_{model_name}.npy", overlap_result)
            logging.info(f"Metric  computed successfully")
        except Exception as e:
            logger.error(f"Error computing metric: {e}")


if __name__ == "__main__":
    main()
