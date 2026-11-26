"""DiffVox loader for paired_pilot.py: path setup, CPU shims for numba CUDA kernels (no libNVVM here),
and a minimal hydra-style instantiate. DiffVox (MIT): https://github.com/SonyResearch/diffvox, cloned to
C:\MIX_automation_ext\diffvox with torchlpc (MIT, source) and torchcomp/torch_fftconv in C:\MIX_automation_ext\pylibs."""
import sys, importlib
sys.path[:0] = [r'C:\MIX_automation_ext\pylibs', r'C:\MIX_automation_ext\torchlpc_src', r'C:\MIX_automation_ext\diffvox']
import warnings; warnings.filterwarnings('ignore')
import torchlpc
_orig = torchlpc.sample_wise_lpc
def _lpc(x, a, zi=None):
    if x.is_cuda:
        d = x.device
        return _orig(x.cpu(), a.cpu(), None if zi is None else zi.cpu()).to(d)
    return _orig(x, a, zi)
torchlpc.sample_wise_lpc = _lpc
def inst(c):
    if isinstance(c, dict):
        if '_target_' in c:
            mod, name = c['_target_'].rsplit('.', 1)
            cls = getattr(importlib.import_module(mod), name)
            args = [inst(a) for a in c.get('_args_', [])]
            kw = {k: inst(v) for k, v in c.items() if k not in ('_target_', '_args_')}
            return cls(*args, **kw)
        return {k: inst(v) for k, v in c.items()}
    if isinstance(c, list):
        return [inst(a) for a in c]
    return c
import torchcomp, torchcomp.core as _tc
_orig_cc = _tc.compressor_core
def _cc(x, zi, at, rt, *a, **k):
    if x.is_cuda:
        d = x.device
        mv = lambda t: t.cpu() if hasattr(t, 'cpu') else t
        return _orig_cc(x.cpu(), mv(zi), mv(at), mv(rt), *a, **k).to(d)
    return _orig_cc(x, zi, at, rt, *a, **k)
torchcomp.compressor_core = _cc
