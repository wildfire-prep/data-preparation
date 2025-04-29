#!/usr/bin/env python3
# Run MOSAIKS model on basemap imagery
import os
import sys
import time
import torch
import pickle  # noqa: F401  (left in case you still need it elsewhere)
import numpy as np
import pandas as pd
import geopandas as gpd  # noqa: F401
import logging
import argparse
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt  # noqa: F401
from tqdm import tqdm  # noqa: F401
from torch.utils.data import DataLoader
from torchgeo.models import RCF

sys.path.append("/capstone/wildfire_prep/leilanie/data-preparation/code/utils")
import config  # noqa: E402
import data_utils  # noqa: E402


# ----------------------------------------------------------------------
# Logging utilities
# ----------------------------------------------------------------------
def setup_logging(
    log_dir: str, logger_name: str = "mosaiks_processor"
) -> logging.Logger:
    """Configure root logging and return a named logger."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"mosaiks_processing_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,  # overwrite any previous basicConfig calls
    )

    logger = logging.getLogger(logger_name)
    logger.info("Logging initialised → %s", log_file.resolve())
    return logger


# ----------------------------------------------------------------------
# Core processing
# ----------------------------------------------------------------------
def process_imagery(
    root_dir: str,
    output_path: str,
    batch_size: int = 100,
    num_workers: int = 20,
    save_frequency: int = 500,  # kept for future use
    logger: logging.Logger | None = None,
) -> None:
    """Run RCF (MOSAIKS) feature extraction on imagery."""
    logger = logger or logging.getLogger(__name__)

    logger.info("Loading dataset from %s", root_dir)
    dataset = data_utils.VisualBasemapDataset(
        root_dir=root_dir,
        transform=None,
        resize=None,
        specific_dir=None,
        clipped=True,
        verbosity=0,
    )

    logger.info("Creating dataloader (batch=%d, workers=%d)", batch_size, num_workers)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=num_workers,
    )

    logger.info("Initialising MOSAIKS (RCF) model")
    mosaiks = RCF(
        in_channels=3,
        features=4000,
        kernel_size=3,
        bias=-1,
        seed=42,
        mode="gaussian",
        dataset=None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    mosaiks.eval().to(device)

    # Pre-allocate output
    n_samples = len(dataset)
    x_all = np.zeros((n_samples, 4000), dtype=float)
    unique_ids, basemap_ids = [], []

    tic = time.time()
    logger.info("Beginning processing of %,d images", n_samples)

    i = 0
    for batch in dataloader:
        for image in batch:
            if i % 100 == 0:
                logger.info(
                    "%d/%d  (%.2f%%)  elapsed %.1fs",
                    i,
                    n_samples,
                    100 * i / n_samples,
                    time.time() - tic,
                )
                tic = time.time()

            with torch.inference_mode():
                feats = mosaiks(image["image"].to(device)).cpu().numpy()
            x_all[i] = feats
            unique_ids.append(image["unique_id"])
            basemap_ids.append(image["basemap_id"])
            i += 1

    logger.info("Processing complete, building DataFrame")
    features_df = (
        pd.DataFrame(x_all, index=unique_ids)
        .add_prefix("X_")
        .reset_index(names="unique_id")
        .assign(basemap_id=basemap_ids)
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving features → %s", output_path)
    features_df.to_feather(output_path)

    logger.info("All done ✔")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MOSAIKS features from imagery"
    )
    parser.add_argument(
        "--root_dir", default=config.clipped_dir, help="Imagery root directory"
    )
    parser.add_argument(
        "--output_path",
        default=os.path.join(
            config.data_dir, "features", "features_mosaiks_4000.feather"
        ),
        help="Output .feather file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="DataLoader batch size"
    )
    parser.add_argument(
        "--num_workers", type=int, default=20, help="DataLoader worker count"
    )
    parser.add_argument(
        "--log_dir",
        default="/capstone/wildfire_prep/leilanie/data-preparation/logs",
        help="Directory for log files",
    )
    args = parser.parse_args()

    # configure logging once
    logger = setup_logging(args.log_dir)
    logger.info("CLI args: %s", vars(args))

    try:
        process_imagery(
            root_dir=args.root_dir,
            output_path=args.output_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            logger=logger,
        )
    except Exception:
        logger.exception("Fatal error during processing")
        sys.exit(1)


if __name__ == "__main__":
    main()

# cd /capstone/wildfire_prep/leilanie/data-preparation
# python code/04_model_development/01_mosaiks.py --batch_size 50 --num_workers 20










# # Run MOSAIKS model on basemaps imagery
# import os
# import sys
# import time
# import torch
# import pickle
# import numpy as np
# import pandas as pd
# import geopandas as gpd

# import matplotlib.pyplot as plt

# from tqdm import tqdm
# from datetime import datetime
# from torchgeo.models import RCF
# from torch.utils.data import DataLoader

# sys.path.append("../utils")

# import config
# import data_utils


# dataset = data_utils.VisualBasemapDataset(
#     root_dir=config.clipped_dir_2019,
#     transform=None,
#     resize=None,
#     specific_dir=None,
#     clipped=True,
#     verbosity=0,
# )


# dataloader = DataLoader(
#     dataset=dataset,
#     batch_size=100,
#     shuffle=False,
#     collate_fn=lambda x: x,
#     num_workers=20,
# )

# mosaiks = RCF(
#     in_channels=3,
#     features=4000,
#     kernel_size=3,
#     bias=-1,
#     seed=42,
#     mode="gaussian",
#     dataset=None,
# )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mosaiks.eval().to(device)

# # Initialize data structures
# unique_ids = []
# basemap_ids = []
# x_all = np.zeros((len(dataset), 4000), dtype=float)

# i = 0
# tic = time.time()

# for batch in dataloader:
#     for image in batch:
#         unique_id = image["unique_id"]

#         if i % 100 == 0:
#             print(
#                 f"{i:,}/{len(dataset):,} -- {i / len(dataset) * 100:0.2f}% -- {time.time() - tic:0.2f} seconds"
#             )
#             tic = time.time()

#         with torch.inference_mode():
#             image_tensor = image["image"].to(device)
#             basemap_id = image["basemap_id"]

#             feats = mosaiks(image_tensor).cpu().numpy()

#             x_all[i] = feats
#             unique_ids.append(unique_id)
#             basemap_ids.append(basemap_id)
#             i += 1

# features_df = pd.DataFrame(x_all, index=unique_ids)
# features_df = features_df.add_prefix("X_").reset_index()
# features_df.rename(columns={"index": "unique_id"}, inplace=True)
# features_df["basemap_id"] = basemap_ids
# filename = os.path.join(config.data_dwir, "features", "features_mosaiks_4000.feather")
# features_df.to_feather(filename)
