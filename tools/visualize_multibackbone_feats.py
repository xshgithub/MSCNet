import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.slconfig import SLConfig
import models.dqdetr  # noqa: F401 - registers the dqdetr builder
from models.registry import MODULE_BUILD_FUNCS
from util.misc import clean_state_dict
import numpy as np
from util.misc import NestedTensor
import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def feature_to_map(feature):
    """Convert a CxHxW or BxCxHxW feature tensor to one HxW map."""


    if not torch.is_tensor(feature):
        raise TypeError(f"feature must be a torch.Tensor, got {type(feature)}")
    feature = feature.detach().float().cpu()
    if feature.ndim == 4:
        feature = feature[0]
    if feature.ndim == 3:
        return feature.mean(dim=0)
    if feature.ndim == 2:
        return feature
    raise ValueError(f"Expected a 2D, 3D, or 4D feature tensor, got shape {tuple(feature.shape)}")


def normalize_to_uint8(feature_map):
    """Normalize a feature map to uint8 for image saving."""
    array = feature_map.detach().float().cpu().numpy()
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value - min_value < 1e-12:
        return np.zeros(array.shape, dtype=np.uint8)
    array = (array - min_value) / (max_value - min_value)
    return (array * 255.0).clip(0, 255).astype(np.uint8)


def colorize_heatmap(gray: np.ndarray) -> np.ndarray:
    """Apply a compact blue-cyan-yellow-red heatmap without matplotlib."""
    value = gray.astype(np.float32) / 255.0
    red = np.clip(1.5 * value - 0.25, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(2.0 * value - 1.0) * 1.5, 0.0, 1.0)
    blue = np.clip(1.25 - 1.5 * value, 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def sanitize_name(name: str) -> str:
    keep = []
    for char in name:
        keep.append(char if char.isalnum() or char in ("-", "_") else "_")
    return "".join(keep).strip("_") or "feature"


def load_config_args(config_file: str, device: str, modalities: Sequence[str] = None):

    cfg = SLConfig.fromfile(config_file)
    args = SimpleNamespace(**cfg._cfg_dict.to_dict())
    args.device = device
    if modalities is not None:
        args.modalities = list(modalities)
    if not hasattr(args, "debug"):
        args.debug = False
    if not hasattr(args, "use_ema"):
        args.use_ema = False
    return args


def build_model(args):
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, _, _ = build_func(args)
    return model


def load_checkpoint(model, checkpoint_path: str, strict: bool = False):

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(clean_state_dict(state_dict), strict=strict)
    return missing, unexpected


def pil_to_normalized_tensor(path: str, size_hw=None):

    image = Image.open(path).convert("RGB")
    if size_hw is not None:
        height, width = size_hw
        image = image.resize((width, height), Image.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def get_image_size_hw(path: str):

    with Image.open(path) as image:
        width, height = image.size
    return height, width


def make_multimodal_samples(image_paths: Sequence[str], device: str, resize_hw=None):

    tensors = [pil_to_normalized_tensor(path, resize_hw).to(device) for path in image_paths]
    samples = []
    for tensor in tensors:
        _, height, width = tensor.shape
        mask = torch.zeros((1, height, width), dtype=torch.bool, device=device)
        samples.append(NestedTensor(tensor.unsqueeze(0), mask))
    return samples


def find_multibackbone(model):
    for module in model.modules():
        if module.__class__.__name__ == "MultiBackbone" and hasattr(module, "dwt") and hasattr(module, "Fusion"):
            return module
    raise RuntimeError("Could not find a MultiBackbone module with dwt and Fusion lists.")


def get_detector_model(model):
    return getattr(model, "detr", model)


def clone_feature_list(features: Iterable):
    return [feature.detach().cpu() for feature in features]


def register_feature_hooks(multibackbone):
    captured: Dict[int, Dict[str, object]] = {
        idx: {"before": None, "after": None, "fused": None}
        for idx in range(len(multibackbone.dwt))
    }
    handles = []

    for idx, dwt_module in enumerate(multibackbone.dwt):
        def dwt_hook(module, inputs, output, stage_idx=idx):
            captured[stage_idx]["before"] = clone_feature_list(inputs[0])
            captured[stage_idx]["after"] = clone_feature_list(output)

        handles.append(dwt_module.register_forward_hook(dwt_hook))

    for idx, fusion_module in enumerate(multibackbone.Fusion):
        def fusion_hook(module, inputs, output, stage_idx=idx):
            captured[stage_idx]["fused"] = output.detach().cpu()

        handles.append(fusion_module.register_forward_hook(fusion_hook))

    return captured, handles


def save_feature_image(feature, path: Path):

    gray = normalize_to_uint8(feature_to_map(feature))
    color = colorize_heatmap(gray)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color).save(path)


def save_captured_features(captured, modalities: Sequence[str], output_dir: str, save_npy: bool = False):
    output_root = Path(output_dir)
    saved_paths = []
    for stage_idx, stage_data in captured.items():
        stage_dir = output_root / f"stage{stage_idx}"
        before = stage_data["before"]
        after = stage_data["after"]
        fused = stage_data["fused"]
        if before is None or after is None or fused is None:
            raise RuntimeError(f"Missing captured features for stage {stage_idx}.")

        for modality, feature in zip(modalities, before):
            path = stage_dir / f"before_{sanitize_name(modality)}.png"
            save_feature_image(feature, path)
            saved_paths.append(path)
            if save_npy:
                np.save(stage_dir / f"before_{sanitize_name(modality)}.npy", feature.numpy())

        for modality, feature in zip(modalities, after):
            path = stage_dir / f"after_{sanitize_name(modality)}.png"
            save_feature_image(feature, path)
            saved_paths.append(path)
            if save_npy:
                np.save(stage_dir / f"after_{sanitize_name(modality)}.npy", feature.numpy())

        fused_path = stage_dir / "fused.png"
        save_feature_image(fused, fused_path)
        saved_paths.append(fused_path)
        if save_npy:
            np.save(stage_dir / "fused.npy", fused.numpy())

    return saved_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize DWT-before, DWT-after, and fused features from the project's MultiBackbone."
    )
    parser.add_argument("--config", default="config/multi_sea.py", help="Path to the config .py file.")
    parser.add_argument("--checkpoint", default=r"D:\pycharm\DQ-DETR-main\output\sla_ssh_sst_chl\50.5_85.6.pth", help="Optional checkpoint path. Uses random weights if omitted.")
    parser.add_argument("--images", nargs="+", required=True, help="One image path per modality, in modality order.")
    parser.add_argument("--modalities", nargs="+", default=None, help="Modality names matching --images.")
    parser.add_argument("--out-dir", default="output/feature_vis", help="Directory for feature images.")
    parser.add_argument("--device", default="cuda:0", help="cuda, cuda:0, or cpu. Defaults to cuda if available.")
    parser.add_argument("--resize", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=None)
    parser.add_argument("--strict-load", action="store_true", help="Use strict=True when loading checkpoint.")
    parser.add_argument("--save-npy", action="store_true", help="Also save raw captured feature tensors as .npy.")
    return parser.parse_args()


def main():
    cli_args = parse_args()

    device = cli_args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args = load_config_args(cli_args.config, device=device, modalities=cli_args.modalities)
    modalities = list(getattr(args, "modalities", []))
    if len(cli_args.images) != len(modalities):
        raise ValueError(
            f"--images count ({len(cli_args.images)}) must match modalities count "
            f"({len(modalities)}): {modalities}"
        )

    resize_hw = tuple(cli_args.resize) if cli_args.resize else get_image_size_hw(cli_args.images[0])
    for image_path in cli_args.images:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)

    model = build_model(args).to(device)
    if cli_args.checkpoint:
        missing, unexpected = load_checkpoint(model, cli_args.checkpoint, strict=cli_args.strict_load)
        if missing:
            print(f"Missing checkpoint keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected checkpoint keys: {len(unexpected)}")
    model.eval()

    detector = get_detector_model(model)
    multibackbone = find_multibackbone(detector)
    captured, handles = register_feature_hooks(multibackbone)
    samples = make_multimodal_samples(cli_args.images, device=device, resize_hw=resize_hw)

    try:
        with torch.no_grad():
            detector.backbone(samples)
    finally:
        for handle in handles:
            handle.remove()

    saved_paths = save_captured_features(captured, modalities, cli_args.out_dir, save_npy=cli_args.save_npy)
    print(f"Saved {len(saved_paths)} feature images to {Path(cli_args.out_dir).resolve()}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()

# Example:
# python tools\visualize_multibackbone_feats.py `
#   --images D:\pycharm\datasets\sea\sla\test\25973.png D:\pycharm\datasets\sea\ssh\test\25973.png D:\pycharm\datasets\sea\sst\test\25973.png D:\pycharm\datasets\sea\chl\test\25973.png
