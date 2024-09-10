import argparse
import os
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch_ecg.utils.misc import get_record_list_recursive3, str2bool
from tqdm.auto import tqdm

from const import MODEL_CACHE_DIR
from models import MultiHead_MBAS2024
from pipeline import run_pipeline

try:
    TEST_FLAG = os.environ.get("MBAS2024_REVENGER_TEST", False)
    TEST_FLAG = str2bool(TEST_FLAG)
except Exception:
    TEST_FLAG = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference.",
        dest="device",
    )
    parser.add_argument("--input_dir", type=str, default="/input", help="path to input")
    parser.add_argument("--output_dir", type=str, default="/output", help="path to input")
    parser.add_argument(
        "--model_pth",
        type=str,
        default=MODEL_CACHE_DIR,
        help="Directory of the trained models.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Generate new output: ", output_dir)

    if "cuda" in args.device and not torch.cuda.is_available():
        args.device = "cpu"
        warnings.warn("CUDA is not available. Using CPU for inference.")
    device = torch.device(args.device)

    # model_pth = Path(args.model_pth).expanduser().resolve()
    # Since we are using the same model for validation and testing, we will treat the cli argument `model_pth` a dummy argument,
    # and use our pre-defined path (see the environment variable `MODEL_CACHE_DIR` of the `Dockerfile`) to load the models.
    model_pth = Path(MODEL_CACHE_DIR).expanduser().resolve()
    stage0_model = MultiHead_MBAS2024.from_checkpoint(model_pth / "stage0-model.pth.tar", device=device)[0]
    stage1_model = MultiHead_MBAS2024.from_checkpoint(model_pth / "stage1-model.pth.tar", device=device)[0]
    stage0_model = stage0_model.to(device).eval()
    stage1_model = stage1_model.to(device).eval()

    # predict the results, here is just an example. Pls build your own logic here
    image_files = get_record_list_recursive3(
        Path(args.input_dir).expanduser().resolve(),
        rec_patterns="^(?!\\._).*_gt\\.nii\\.gz",  # ignore the hidden backup files (e.g. in __MACOSX)
        relative=False,
        with_suffix=True,
    )
    print(f"Total {len(image_files)} records found.")

    if TEST_FLAG:
        image_files = np.random.choice(image_files, 10, replace=False)
        print(f"Test mode: {len(image_files)} records selected.")

    for rec in tqdm(image_files, desc="Inference", unit="record", dynamic_ncols=True, mininterval=1.0):
        img = nib.load(rec).get_fdata()
        pred_mask = run_pipeline(img, stage0_model, stage1_model, parallel=False)
        nib.save(
            nib.Nifti1Image(pred_mask, affine=np.eye(4)),
            str(output_dir / Path(rec).name.replace("_gt.nii.gz", "_label.nii.gz")),
        )

    print("Generate finished!")
