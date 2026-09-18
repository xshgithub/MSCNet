import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
WAVELET_COMPONENT_KEYS = ("H_f0", "H_f1", "H_f2")


def reduce_feature_to_map(feature):
    """Convert HxW, CxHxW, or BxCxHxW features to one HxW activation map."""
    if not hasattr(feature, "detach"):
        raise TypeError(f"feature must be a torch.Tensor-like object, got {type(feature)}")

    feature = feature.detach().float().cpu()
    if feature.ndim == 4:
        feature = feature[0]
    if feature.ndim == 3:
        return feature.abs().mean(dim=0)
    if feature.ndim == 2:
        return feature.abs()
    raise ValueError(f"Expected a 2D, 3D, or 4D feature tensor, got shape {tuple(feature.shape)}")


def normalize_to_uint8(feature_map) -> np.ndarray:
    """Normalize a tensor-like feature map to uint8 for heatmap rendering."""
    if hasattr(feature_map, "detach"):
        array = feature_map.detach().float().cpu().numpy()
    else:
        array = np.asarray(feature_map, dtype=np.float32)

    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value - min_value < 1e-12:
        return np.zeros(array.shape, dtype=np.uint8)
    array = (array - min_value) / (max_value - min_value)
    return (array * 255.0).clip(0, 255).astype(np.uint8)


def colorize_heatmap(gray: np.ndarray) -> np.ndarray:
    """Apply a compact RGB heatmap without requiring matplotlib or OpenCV."""
    value = gray.astype(np.float32) / 255.0
    red = np.clip(1.5 * value - 0.25, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(2.0 * value - 1.0) * 1.5, 0.0, 1.0)
    blue = np.clip(1.25 - 1.5 * value, 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def feature_to_heatmap(feature, image_size: Optional[int] = None) -> np.ndarray:
    gray = normalize_to_uint8(reduce_feature_to_map(feature))
    color = colorize_heatmap(gray)
    if image_size is None:
        return color
    return np.asarray(Image.fromarray(color).resize((image_size, image_size), Image.BILINEAR))


def sanitize_name(name: str) -> str:
    keep = []
    for char in name:
        keep.append(char if char.isalnum() or char in ("-", "_") else "_")
    return "".join(keep).strip("_") or "feature"


def parse_stages(stage_args: Sequence[str], stage_count: int) -> Optional[List[int]]:
    if len(stage_args) == 1 and stage_args[0].lower() == "all":
        return None

    stages = sorted({int(stage) for stage in stage_args})
    for stage in stages:
        if stage < 0 or stage >= stage_count:
            raise ValueError(f"Requested stage {stage} is outside available range 0-{stage_count - 1}")
    return stages


def clone_tensor(feature):
    return feature.detach().float().cpu().clone()


def clone_feature_list(features: Iterable):
    return [clone_tensor(feature) for feature in features]


def assemble_compare_grid(heatmaps: Dict[str, object], modalities: Sequence[str]) -> np.ndarray:
    rows = []
    modality_count = len(modalities)
    for key in ("dwt_before", "dwt_after", "fusion_before"):
        row_images = heatmaps[key]
        if len(row_images) != modality_count:
            raise ValueError(f"{key} has {len(row_images)} maps, expected {modality_count}")
        rows.append(np.concatenate(row_images, axis=1))

    fused = heatmaps["fused_after"]
    rows.append(np.concatenate([fused for _ in range(modality_count)], axis=1))
    return np.concatenate(rows, axis=0)


def assemble_wavelet_grid(heatmaps: Dict[str, np.ndarray]) -> np.ndarray:
    row_images = []
    for key in WAVELET_COMPONENT_KEYS:
        if key not in heatmaps:
            raise ValueError(f"Missing wavelet heatmap: {key}")
        row_images.append(heatmaps[key])
    return np.concatenate(row_images, axis=1)


class FusionFeatureHook:
    def __init__(self, multibackbone, stages: Sequence[int]):
        self.multibackbone = multibackbone
        self.stages = list(stages)
        self.records: Dict[int, Dict[str, object]] = {
            stage: {
                "dwt_before": None,
                "dwt_after": None,
                "fusion_before": None,
                "fused_after": None,
                "wavelet_components": None,
            }
            for stage in self.stages
        }
        self.handles = []
        self._previous_capture_flags = {}
        self._missing_capture_flag = object()

    def __enter__(self):
        for stage in self.stages:
            dwt_module = self.multibackbone.dwt[stage]
            self._previous_capture_flags[stage] = getattr(
                dwt_module,
                "capture_wavelet_components",
                self._missing_capture_flag,
            )
            if hasattr(dwt_module, "capture_wavelet_components"):
                dwt_module.capture_wavelet_components = True
            self.handles.append(self.multibackbone.dwt[stage].register_forward_hook(self._make_dwt_hook(stage)))
            self.handles.append(
                self.multibackbone.Fusion[stage].register_forward_pre_hook(self._make_fusion_pre_hook(stage))
            )
            self.handles.append(self.multibackbone.Fusion[stage].register_forward_hook(self._make_fusion_hook(stage)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        for stage, previous in self._previous_capture_flags.items():
            dwt_module = self.multibackbone.dwt[stage]
            if previous is self._missing_capture_flag:
                continue
            dwt_module.capture_wavelet_components = previous
            if hasattr(dwt_module, "last_wavelet_components"):
                dwt_module.last_wavelet_components = None
        self._previous_capture_flags = {}

    def _make_dwt_hook(self, stage: int):
        def hook(module, inputs, output):
            self.records[stage]["dwt_before"] = clone_feature_list(inputs[0])
            self.records[stage]["dwt_after"] = clone_feature_list(output)
            components = getattr(module, "last_wavelet_components", None)
            if components is not None:
                self.records[stage]["wavelet_components"] = {
                    key: clone_tensor(components[key])
                    for key in WAVELET_COMPONENT_KEYS
                    if key in components
                }

        return hook

    def _make_fusion_pre_hook(self, stage: int):
        def hook(module, inputs):
            self.records[stage]["fusion_before"] = clone_feature_list(inputs[0])

        return hook

    def _make_fusion_hook(self, stage: int):
        def hook(module, inputs, output):
            self.records[stage]["fused_after"] = clone_tensor(output)

        return hook


def load_config_args(config_file: str, device: str, modalities: Optional[Sequence[str]] = None):
    from util.slconfig import SLConfig

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
    import models.dqdetr  # noqa: F401 - registers the dqdetr builder
    from models.registry import MODULE_BUILD_FUNCS

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    if build_func is None:
        raise ValueError(f"Unknown modelname in config: {args.modelname}")
    model, _, _ = build_func(args)
    return model


def load_checkpoint(model, checkpoint_path: str, strict: bool = False):
    import torch
    from util.misc import clean_state_dict

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    return model.load_state_dict(clean_state_dict(state_dict), strict=strict)


def get_detector_model(model):
    return getattr(model, "detr", model)


def find_multibackbone(model):
    for module in model.modules():
        if module.__class__.__name__ == "MultiBackbone" and hasattr(module, "dwt") and hasattr(module, "Fusion"):
            return module
    raise RuntimeError("Could not find a MultiBackbone module with dwt and Fusion lists.")


def pil_to_normalized_tensor(path: str, size_hw=None):
    import torch

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
    import torch
    from util.misc import NestedTensor

    tensors = [pil_to_normalized_tensor(path, resize_hw).to(device) for path in image_paths]
    samples = []
    for tensor in tensors:
        _, height, width = tensor.shape
        mask = torch.zeros((1, height, width), dtype=torch.bool, device=device)
        samples.append(NestedTensor(tensor.unsqueeze(0), mask))
    return samples


def validate_inputs(image_paths: Sequence[str], checkpoint_path: Optional[str]) -> None:
    for image_path in image_paths:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)
    if checkpoint_path and not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)


def save_heatmap(path: Path, feature, image_size: Optional[int] = None) -> np.ndarray:
    heatmap = feature_to_heatmap(feature, image_size=image_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(heatmap).save(path)
    return heatmap


def save_stage_features(
    records: Dict[int, Dict[str, object]],
    modalities: Sequence[str],
    output_dir: str,
    image_size: Optional[int] = None,
    save_npy: bool = False,
    save_wavelet_components: bool = False,
) -> List[Path]:
    output_root = Path(output_dir)
    saved_paths = []

    for stage, stage_data in sorted(records.items()):
        stage_dir = output_root / f"stage{stage}"
        for key in ("dwt_before", "dwt_after", "fusion_before", "fused_after"):
            if stage_data[key] is None:
                raise RuntimeError(f"Missing captured {key} features for stage {stage}")

        heatmaps = {}
        for key in ("dwt_before", "dwt_after", "fusion_before"):
            heatmaps[key] = []
            for modality, feature in zip(modalities, stage_data[key]):
                name = sanitize_name(modality)
                path = stage_dir / f"{key}_{name}.png"
                heatmaps[key].append(save_heatmap(path, feature, image_size=image_size))
                saved_paths.append(path)
                if save_npy:
                    npy_path = stage_dir / f"{key}_{name}.npy"
                    np.save(npy_path, feature.numpy())
                    saved_paths.append(npy_path)

        fused_path = stage_dir / "fused_after.png"
        heatmaps["fused_after"] = save_heatmap(fused_path, stage_data["fused_after"], image_size=image_size)
        saved_paths.append(fused_path)
        if save_npy:
            fused_npy_path = stage_dir / "fused_after.npy"
            np.save(fused_npy_path, stage_data["fused_after"].numpy())
            saved_paths.append(fused_npy_path)

        compare_path = stage_dir / "compare_grid.png"
        Image.fromarray(assemble_compare_grid(heatmaps, modalities)).save(compare_path)
        saved_paths.append(compare_path)

        if save_wavelet_components:
            components = stage_data["wavelet_components"]
            if components is None:
                raise RuntimeError(f"Missing captured wavelet components for stage {stage}")
            missing = [key for key in WAVELET_COMPONENT_KEYS if key not in components]
            if missing:
                raise RuntimeError(f"Missing wavelet components for stage {stage}: {', '.join(missing)}")

            wavelet_heatmaps = {}
            wavelet_modality = sanitize_name(modalities[0] if modalities else "sla")
            for key in WAVELET_COMPONENT_KEYS:
                path = stage_dir / f"wavelet_{wavelet_modality}_{key}.png"
                wavelet_heatmaps[key] = save_heatmap(path, components[key], image_size=image_size)
                saved_paths.append(path)
                if save_npy:
                    npy_path = stage_dir / f"wavelet_{wavelet_modality}_{key}.npy"
                    np.save(npy_path, components[key].numpy())
                    saved_paths.append(npy_path)

            wavelet_grid_path = stage_dir / f"wavelet_{wavelet_modality}_compare_grid.png"
            Image.fromarray(assemble_wavelet_grid(wavelet_heatmaps)).save(wavelet_grid_path)
            saved_paths.append(wavelet_grid_path)

    return saved_paths


def run(cli_args) -> List[Path]:
    import torch

    device = cli_args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    args = load_config_args(cli_args.config, device=device, modalities=cli_args.modalities)
    modalities = list(getattr(args, "modalities", []))
    if len(cli_args.images) != len(modalities):
        raise ValueError(
            f"--images count ({len(cli_args.images)}) must match modalities count "
            f"({len(modalities)}): {modalities}"
        )

    validate_inputs(cli_args.images, cli_args.checkpoint)
    resize_hw = tuple(cli_args.resize) if cli_args.resize else get_image_size_hw(cli_args.images[0])

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
    selected_stages = parse_stages(cli_args.stages, len(multibackbone.dwt))
    if selected_stages is None:
        selected_stages = list(range(len(multibackbone.dwt)))

    samples = make_multimodal_samples(cli_args.images, device=device, resize_hw=resize_hw)
    with FusionFeatureHook(multibackbone, selected_stages) as capture:
        with torch.no_grad():
            detector.backbone(samples)
        saved_paths = save_stage_features(
            capture.records,
            modalities,
            cli_args.out_dir,
            image_size=cli_args.image_size,
            save_npy=cli_args.save_npy,
            save_wavelet_components=cli_args.save_wavelet_components,
        )

    return saved_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize DQ-DETR DWT-before, DWT-after, Fusion-before, and fused-after features."
    )
    parser.add_argument("--config", default="config/multi_sea.py", help="Path to the config .py file.")
    parser.add_argument(
        "--checkpoint",
        default=r"output\sla_ssh_sst_chl\50.5_85.6.pth",
        help="Checkpoint path. Leave empty to use randomly initialized weights.",
    )
    parser.add_argument("--images", nargs="+", required=True, help="One image path per modality, in modality order.")
    parser.add_argument("--modalities", nargs="+", default=None, help="Modality names matching --images.")
    parser.add_argument("--stages", nargs="+", default=["all"], help="Stage indexes to save, or 'all'.")
    parser.add_argument("--out-dir", default="output/fusion_feature_vis", help="Output directory for feature images.")
    parser.add_argument("--device", default="cuda:0", help="cuda, cuda:0, or cpu.")
    parser.add_argument("--resize", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=None)
    parser.add_argument("--image-size", type=int, default=645, help="Optional square output heatmap size.")
    parser.add_argument("--strict-load", action="store_true", help="Use strict=True when loading checkpoint.")
    parser.add_argument("--save-npy", action="store_true", help="Also save raw captured tensors as .npy files.")
    parser.add_argument(
        "--save-wavelet-components",
        action="store_true",
        help="Also save the SLA H_f0, H_f1, and H_f2 maps from each selected DWT stage.",
    )
    return parser

    # SLA_FILE = r"D:\pycharm\datasets\sea\sla\test\25973.png"
    # SSH_FILE = r"D:\pycharm\datasets\sea\ssh\test\25973.png"
    # SST_FILE = r"D:\pycharm\datasets\sea\sst\test\25973.png"
    # CHL_FILE = r"D:\pycharm\datasets\sea\chl\test\25973.png"

def main() -> None:
    parser = build_parser()
    cli_args = parser.parse_args()
    saved_paths = run(cli_args)
    print(f"Saved {len(saved_paths)} feature files to {Path(cli_args.out_dir).resolve()}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
