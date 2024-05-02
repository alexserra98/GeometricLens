import pandas as pd
from pathlib import Path
from metrics.intrinisic_dimension import IntrinsicDimension
from metrics.utils import TensorStorageManager, DataFrameQuery
from common.globals_vars import _OUTPUT_DIR_TRANSPOSED
from enum import Enum
import argparse
import importlib
import logging
from logging.logging_config import setup_logging

logger = setup_logging()
class MetricName(Enum):
    INTRINSIC_DIMENSION = ("metrics.intrinisic_dimension", "IntrinsicDimension")
    LABEL_OVERLAP = ("metrics.overlap", "LabelOverlap")
    POINT_OVERLAP = ("metrics.overlap", "PointOverlap")
    PROBE = ("metrics.probe", "LinearProbe")

def metric_function(name):
    try:
        metric_info = MetricName[name.upper()].value
        module = importlib.import_module(metric_info[0])
        return getattr(module, metric_info[1])
    except KeyError:
        raise ValueError(f"Unknown metric function {name}")
    except AttributeError:
        raise ImportError(f"Could not import the specified class from the module")

def create_queries(models):
    queries = []
    for model in models:
        for shot in [0, 2, 5]:
            if "70" in model and shot == 5 and "chat" not in model:
                shot = 4
            elif "70" in model and "chat" in model and shot == 5:
                continue
            queries.append({"method": "last", "model_name": model, "train_instances": shot})
    return queries

def main():
    #Getting commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', help='Ground truth to use', required=False)
    args, remaining_argv = parser.parse_known_args()
    label = args.label

    logger.info("Metrics computer started")
    
    models = [
        "meta-llama-Llama-2-7b-hf",
        "meta-llama-Llama-2-7b-chat-hf",
        "meta-llama-Llama-2-13b-hf",
        "meta-llama-Llama-2-13b-chat-hf",
        "meta-llama-Llama-2-70b-hf",
        "meta-llama-Llama-2-70b-chat-hf",
        "meta-llama-Llama-3-8b-hf",
        "meta-llama-Llama-3-8b-chat-hf",
        "meta-llama-Llama-3-70b-hf",
        "meta-llama-Llama-3-70b-chat-hf",
        "meta-llama-Llama-3-8b-ft-hf",
    ]

    # List of variations to apply
    variations = {"intrinsic_dimension": "None", "point_overlap": "cosine", "label_overlap": "balanced_letter", "cka": "rbf"}
    
    # List of metrics to compute
    metrics = ["intrinsic_dimension", "label_overlap", "point_overlap", "probe"]
    for metric in metrics:
        metric_class = metric_function(metric)

        if metric == "probe" or metric == "label_overlap" and label is None:
            raise ValueError("Probe and LabelOverlap metrics require a label to be specified")
        
        # Create queries for hidden states
        queries = create_queries(models) 
        metric_instance = metric_class(queries=queries, 
                                       tensor_storage=None, 
                                       variations=variations, 
                                       storage_logic="npy"
        )
        
        try:
            if label:
                labels = list(label)
                for label in labels:
                    result = metric_instance.main(label)
                    result_path = _OUTPUT_DIR_TRANSPOSED / "result" / f"{metric}_{label}.pkl"
                    result.to_pickle(result_path)
            else:
                result = metric_instance.main()
                result_path = _OUTPUT_DIR_TRANSPOSED / "result" / f"{metric}.pkl"
                result.to_pickle(result_path)
            
            logging.info(f"Metric {metric} computed successfully")
        except Exception as e:
            logger.error(f"Error computing metric {metric}: {e}")
            

if __name__ == "__main__":
    main()
