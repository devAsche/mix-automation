"""R4b: AFx-Rep audio-effects embedding (ST-ITO, Steinmetz et al. 2024) as an extra distance term.

Weights: Hugging Face csteinmetz1/afx-rep (Apache-2.0), downloaded to C:\\MIX_automation_ext\\weights\\afx-rep.
Model code: st-ito `st_ito/models/panns.py` (Apache-2.0) from the clone in C:\\MIX_automation_ext\\src\\st-ito,
loaded by file path so the rest of the st-ito package (and its dependencies) is not imported.
Dependency torchlibrosa lives in C:\\MIX_automation_ext\\pylibs. Nothing here is used by v7 or the engine.

Embedding as in st-ito `get_param_embeds`: resample to 48 kHz, peak-normalise each item, model(x) -> (mid, side)
512-d each. We use the cosine distance of the concatenated, L2-normalised mid and side embeddings.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

EXT = Path(r"C:\MIX_automation_ext")
WEIGHTS = EXT / "weights" / "afx-rep"
PANNS = EXT / "src" / "st-ito" / "st_ito" / "models" / "panns.py"
PYLIBS = EXT / "pylibs"
MODEL_SR = 48000

_MODEL = None


def available() -> bool:
    return (WEIGHTS / "afx-rep.ckpt").exists() and (WEIGHTS / "config.yaml").exists() and PANNS.exists()


def load_model(use_gpu: bool = True):
    """Build Cnn14 from config.yaml and load the encoder weights (tensors only: weights_only=True)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import torch
    import yaml

    if str(PYLIBS) not in sys.path:
        sys.path.append(str(PYLIBS))
    spec = importlib.util.spec_from_file_location("_afxrep_panns", PANNS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = yaml.safe_load((WEIGHTS / "config.yaml").read_text(encoding="utf-8"))
    enc = cfg["model"]["init_args"]["encoder"]
    cls = getattr(mod, enc["class_path"].rsplit(".", 1)[1])
    model = cls(**enc["init_args"])
    ckpt = torch.load(WEIGHTS / "afx-rep.ckpt", map_location="cpu", weights_only=True)
    sd = {k.replace("encoder.", "", 1): v for k, v in ckpt["state_dict"].items() if k.startswith("encoder")}
    model.load_state_dict(sd)
    model.eval()
    if use_gpu and torch.cuda.is_available():
        model.cuda()
    _MODEL = model
    return model


def embed(x: np.ndarray, sr: int) -> np.ndarray:
    """x: (2, n) or (n,) float -> unit-norm 1024-d vector (mid ++ side, each L2-normalised)."""
    import torch
    import torchaudio

    model = load_model()
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        a = np.stack([a, a])
    t = torch.from_numpy(a[None]).type_as(next(model.parameters()))
    if sr != MODEL_SR:
        t = torchaudio.functional.resample(t, sr, MODEL_SR)
    t = t / t.abs().max().clamp(1e-8)
    with torch.no_grad():
        mid, side = model(t)
    mid = torch.nn.functional.normalize(mid, dim=-1)
    side = torch.nn.functional.normalize(side, dim=-1)
    v = torch.cat([mid, side], dim=-1)[0].float().cpu().numpy()
    return v / (np.linalg.norm(v) + 1e-12)


def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
