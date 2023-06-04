from datetime import datetime
import logging
import os
import random

import torch


def setup_logging(save_dir):
    os.makedirs(os.path.join(save_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'ckpts'), exist_ok=True)
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=os.path.join(save_dir, "logs", f"log_{dt}.txt"),
        filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    return logging.getLogger("").handlers[0].baseFilename


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        try:
            import transformers

            transformers.set_seed(seed)
        except ImportError:
            pass
        logging.info(f"Set random seed to {seed}")
