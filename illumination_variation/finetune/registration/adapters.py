"""Matcher adapters. WHY: eLoFTR emits raw tensors, vismatch emits filtered
numpy dicts — both are normalized here to MatchOutput so metrics see one shape."""
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional, Tuple
import numpy as np
import torch

ELOFTR_BASE_CKPT = "finetune/EfficientLoFTR/weights/eloftr_outdoor.ckpt"
ELOFTR_DEFAULT_CKPT = (
    "finetune/EfficientLoFTR/logs/tb_logs/"
    "lunar_full_finetune/version_1/checkpoints/"
    "epoch=0-auc@5=0.788-auc@10=0.853-auc@20=0.899.ckpt"
)

@dataclass
class MatchOutput:
    matched_kpts0: np.ndarray
    matched_kpts1: np.ndarray
    inlier_kpts0: np.ndarray
    inlier_kpts1: np.ndarray
    H: Optional[np.ndarray]
    num_candidates: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    confidences: Optional[np.ndarray] = None
    latency_match_ms: float = 0.0
    latency_ransac_ms: float = 0.0

def finalize_output(
    matched0,
    matched1,
    H,
    in0,
    in1,
    mask,
    confidences=None,
    t_match=0.0,
    t_ransac=0.0,
) -> MatchOutput:
    matched0_arr = np.asarray(matched0, dtype=np.float64).reshape(-1, 2)
    matched1_arr = np.asarray(matched1, dtype=np.float64).reshape(-1, 2)
    in0_arr = np.asarray(in0, dtype=np.float64).reshape(-1, 2)
    in1_arr = np.asarray(in1, dtype=np.float64).reshape(-1, 2)
    n = int(len(matched0_arr))
    k = int(len(in0_arr))
    return MatchOutput(
        matched_kpts0=matched0_arr,
        matched_kpts1=matched1_arr,
        inlier_kpts0=in0_arr,
        inlier_kpts1=in1_arr,
        H=H,
        num_candidates=n,
        num_inliers=k,
        inlier_ratio=(k / n) if n > 0 else 0.0,
        confidences=confidences,
        latency_match_ms=float(t_match),
        latency_ransac_ms=float(t_ransac),
    )

def _resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    # Relative to repo root
    repo_root = Path(__file__).resolve().parent.parent.parent
    p_repo = repo_root / path_str
    if p_repo.exists():
        return p_repo.resolve()
    if str(path_str).startswith("finetune/"):
        stripped = str(path_str)[len("finetune/"):]
        p_strip = repo_root / "finetune" / stripped
        if p_strip.exists():
            return p_strip.resolve()
    return p.resolve()

def normalize_model_name(name: str) -> str:
    """Normalize model string alias to canonical model key."""
    n = name.strip().lower()
    if n in ("eloftr", "eloftr_finetuned", "eloftr-finetuned", "efficient-loftr", "efficient_loftr", "eloftr_lunar"):
        return "eloftr_finetuned"
    if n in ("eloftr_base", "eloftr-base", "base-eloftr", "eloftr_outdoor"):
        return "eloftr_base"
    if n in ("lightglue", "sp-lightglue"):
        return "superpoint-lightglue"
    if n in ("sift_lightglue",):
        return "sift-lightglue"
    if n in ("superglue_outdoor",):
        return "superglue"
    return n

def load_eloftr(ckpt_path, device):
    """Load EfficientLoFTR (finetuned or base). WHY lazy imports: keeps the
    registration package importable without EfficientLoFTR on sys.path."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent.parent
    proj = repo_root / "finetune" / "EfficientLoFTR"
    if str(proj) not in sys.path:
        sys.path.insert(0, str(proj))
    from src.config.default import get_cfg_defaults
    from src.loftr import LoFTR, reparameter
    from src.utils.misc import lower_config

    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(proj / "configs" / "loftr" / "eloftr_lunar.py"))
    cfg.merge_from_file(str(proj / "configs" / "data" / "stereo_lunar_512.py"))
    model = LoFTR(config=lower_config(cfg)["loftr"])
    resolved_ckpt = _resolve_repo_path(str(ckpt_path)) if ckpt_path else None
    if resolved_ckpt and resolved_ckpt.exists():
        sd = torch.load(str(resolved_ckpt), map_location="cpu", weights_only=False)
        model.load_state_dict(sd.get("state_dict", sd), strict=False)
    model = reparameter(model).eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model

class EloftrAdapter:
    """Finetuned/base eLoFTR. Input: gray [1,512,512] float [0,1] arrays or tensors."""
    def __init__(self, ckpt_path=ELOFTR_DEFAULT_CKPT, device="cuda"):
        self.device = torch.device(device)
        self.model = load_eloftr(ckpt_path, self.device)

    @torch.inference_mode()
    def match_gray(self, img0, img1) -> MatchOutput:
        from registration.ransac import run_ransac
        i0 = torch.as_tensor(img0)
        i1 = torch.as_tensor(img1)
        while i0.ndim < 4:
            i0 = i0.unsqueeze(0)
        while i1.ndim < 4:
            i1 = i1.unsqueeze(0)
        b = {
            "image0": i0.to(self.device).float(),
            "image1": i1.to(self.device).float(),
        }
        t0 = time.perf_counter()
        self.model(b)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_match = (time.perf_counter() - t0) * 1000.0
        m0 = b["mkpts0_f"].detach().cpu().numpy().reshape(-1, 2)
        m1 = b["mkpts1_f"].detach().cpu().numpy().reshape(-1, 2)
        t1 = time.perf_counter()
        H, i0_in, i1_in, mask = run_ransac(m0, m1)
        t_r = (time.perf_counter() - t1) * 1000.0
        return finalize_output(m0, m1, H, i0_in, i1_in, mask, t_match=t_match, t_ransac=t_r)

class VismatchAdapter:
    """Any vismatch model. Input: RGB [3,512,512] float [0,1] numpy/tensor."""
    def __init__(self, name="sift-lightglue", device="cuda"):
        import vismatch
        canonical_name = normalize_model_name(name)
        self.name = canonical_name
        self.device = torch.device(device)
        self.matcher = vismatch.get_matcher(canonical_name, device=str(device))

    @torch.inference_mode()
    def match_rgb(self, img0, img1) -> MatchOutput:
        from registration.ransac import run_ransac
        if torch.is_tensor(img0):
            img0 = img0.detach().cpu().numpy()
        if torch.is_tensor(img1):
            img1 = img1.detach().cpu().numpy()
        img0 = np.asarray(img0, dtype=np.float32)
        img1 = np.asarray(img1, dtype=np.float32)
        if img0.ndim == 3 and img0.shape[-1] == 3:
            img0 = np.transpose(img0, (2, 0, 1))
        if img1.ndim == 3 and img1.shape[-1] == 3:
            img1 = np.transpose(img1, (2, 0, 1))

        t0 = time.perf_counter()
        out = self.matcher(img0, img1)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_match = (time.perf_counter() - t0) * 1000.0
        m0 = np.asarray(out.get("matched_kpts0", np.zeros((0, 2)))).reshape(-1, 2)
        m1 = np.asarray(out.get("matched_kpts1", np.zeros((0, 2)))).reshape(-1, 2)
        conf = out.get("matched_confidences")
        t1 = time.perf_counter()
        H, i0_in, i1_in, mask = run_ransac(m0, m1)  # uniform re-filter; ignore out['H']
        t_r = (time.perf_counter() - t1) * 1000.0
        return finalize_output(m0, m1, H, i0_in, i1_in, mask, confidences=conf,
                               t_match=t_match, t_ransac=t_r)

def build_adapter(model_key: str, device: str = "cuda", custom_ckpt: Optional[str] = None) -> Tuple[object, str]:
    """Factory creating (adapter, input_kind) where input_kind is 'gray' or 'rgb'."""
    norm = normalize_model_name(model_key)
    if norm == "eloftr_finetuned":
        ckpt = custom_ckpt or ELOFTR_DEFAULT_CKPT
        return EloftrAdapter(ckpt, device=device), "gray"
    if norm == "eloftr_base":
        ckpt = custom_ckpt or ELOFTR_BASE_CKPT
        return EloftrAdapter(ckpt, device=device), "gray"
    return VismatchAdapter(norm, device=device), "rgb"
