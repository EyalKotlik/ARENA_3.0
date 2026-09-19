#!/opt/venv/bin/python
"""Check that this environment actually works on the GPU it's running on.

Usage:  smoke-test            full check, downloads GPT-2 small (~0.5 GB)
        smoke-test --quick    skip the model download
Exit code is non-zero if anything failed.
"""

import subprocess
import sys
from pathlib import Path

QUICK = "--quick" in sys.argv
results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def run_isolated(name: str, code: str, timeout: int = 300) -> None:
    """Run a snippet in its own process, so a crash (e.g. a CUDA segfault) can't take down the test."""
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)
        out = (p.stdout.strip().splitlines() or [""])[-1]
        err = (p.stderr.strip().splitlines() or [""])[-1]
        record(name, p.returncode == 0, out if p.returncode == 0 else err[:200])
    except subprocess.TimeoutExpired:
        record(name, False, f"timed out after {timeout}s")


# --- 1. PyTorch sees the GPU, and the wheels were built for this card ---------------------
backend_file = Path("/opt/arena-image/torch_backend.txt")
backend = backend_file.read_text().strip() if backend_file.exists() else "unknown"

import torch  # noqa: E402

print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), image built with TORCH_BACKEND={backend}")
if not torch.cuda.is_available():
    record("CUDA available", False,
           "no GPU visible. Locally: did you pass --gpus all? Otherwise the host driver is too old "
           f"for {backend} (cu126 needs >= 525, cu130 needs >= 580; check nvidia-smi).")
else:
    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    record("CUDA available", True, f"{name}, {vram:.1f} GB, compute capability {major}.{minor}")

    arch = f"sm_{major}{minor}"
    supported = torch.cuda.get_arch_list()
    record("wheel supports this GPU", arch in supported,
           f"{arch} in build" if arch in supported else f"{arch} not in {supported} - rebuild with another TORCH_BACKEND")

    try:
        a = torch.randn(512, 512)
        b = torch.randn(512, 512)
        gpu = (a.cuda() @ b.cuda()).cpu()
        record("GPU matmul matches CPU", torch.allclose(gpu, a @ b, atol=1e-2))
    except Exception as e:  # noqa: BLE001
        record("GPU matmul matches CPU", False, repr(e)[:200])

# --- 2. Every library ARENA imports, each in its own process ----------------------------
MODULES = [
    "transformer_lens", "sae_lens", "sae_vis", "circuitsvis", "eindex", "einops", "jaxtyping",
    "nnsight", "transformers", "datasets", "peft", "accelerate", "bitsandbytes",
    "plotly", "neel_plotly", "cv2", "sklearn", "hdbscan", "umap",
    "gymnasium", "envpool", "inspect_ai", "openai", "anthropic", "wandb",
]
for mod in MODULES:
    run_isolated(f"import {mod}", f"import {mod}; print(getattr({mod}, '__version__', 'ok'))")

# --- 3. JAX on the GPU (ARENA 2.3 MuJoCo section), plus headless MuJoCo rendering ------------
run_isolated("jax sees GPU", "import jax; d = jax.devices(); assert d[0].platform == 'gpu', d; print(d)")
run_isolated("import brax", "import brax; print('ok')")
run_isolated(
    "mujoco headless render",
    "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>'); "
    "r = mujoco.Renderer(m, 64, 64); d = mujoco.MjData(m); mujoco.mj_forward(m, d); "
    "r.update_scene(d); print(r.render().shape)",
)

# --- 4. End to end: the thing ARENA 1.2 actually does ------------------------------------
if not QUICK:
    run_isolated(
        "GPT-2 small forward pass + cache",
        "import torch; from transformer_lens import HookedTransformer; "
        "m = HookedTransformer.from_pretrained('gpt2-small', device='cuda'); "
        "logits, cache = m.run_with_cache('The Eiffel Tower is in the city of'); "
        "top = m.to_string(logits[0, -1].argmax()); "
        "print(f'next token {top!r}, {len(cache)} activations cached, '"
        "      f'{torch.cuda.max_memory_allocated()/2**30:.2f} GB peak')",
        timeout=900,
    )

# --- Summary --------------------------------------------------------------------------------
failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed.")
for name, _, detail in failed:
    print(f"  FAILED: {name} - {detail}")
sys.exit(1 if failed else 0)
