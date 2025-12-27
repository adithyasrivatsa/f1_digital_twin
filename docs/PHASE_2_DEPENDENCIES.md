# PHASE 2 — Environment & Dependency Strategy

## Why CPU-Only Environment First

1. **Validation without hardware dependency:** All logic errors, shape mismatches, and algorithm bugs surface on CPU. GPU does not fix broken code, it runs broken code faster.

2. **CI compatibility:** GitHub Actions, GitLab CI, and most CI systems do not have GPUs. CPU tests must pass before merge.

3. **Faster iteration:** CPU PyTorch imports in 2 seconds. CUDA PyTorch imports in 8+ seconds and initializes device. During development, you import hundreds of times.

4. **Debugging:** CPU tensors have readable stack traces. CUDA errors are cryptic. Debug on CPU, scale on GPU.

5. **Arch Linux CUDA is fragile:** Arch rolling release means CUDA driver version changes without warning. CPU environment is stable.

## Why GPU Environment Must Be Separate

PyTorch ships as different packages:
- `torch` (CPU only) — ~200MB
- `torch+cu118` (CUDA 11.8) — ~2GB
- `torch+cu121` (CUDA 12.1) — ~2GB

These are not compatible. You cannot:
- Install CPU torch, then "add" CUDA
- Install CUDA torch on CPU-only machine and expect it to work
- Mix CUDA versions

**Separate virtual environments are mandatory.**

## Version Pinning Strategy

### Pinned Versions (requirements_cpu.txt)

```
# Core ML
torch==2.2.0+cpu
numpy==1.26.4
scipy==1.12.0

# RL
gymnasium==0.29.1

# Config
pyyaml==6.0.1
jsonschema==4.21.1

# Logging
tensorboard==2.16.2

# Testing
pytest==8.0.2
pytest-cov==4.1.0

# Type checking
mypy==1.8.0

# Code quality
ruff==0.2.2
```

### Pinned Versions (requirements_gpu.txt)

```
# Core ML — CUDA 12.1 for Arch Linux compatibility
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.2.0+cu121
numpy==1.26.4
scipy==1.12.0

# Everything else identical to CPU
gymnasium==0.29.1
pyyaml==6.0.1
jsonschema==4.21.1
tensorboard==2.16.2
pytest==8.0.2
pytest-cov==4.1.0
mypy==1.8.0
ruff==0.2.2
```

### Why These Versions

| Package | Version | Reason |
|---------|---------|--------|
| torch 2.2.0 | Latest stable with compile() support | torch.compile works, 2.3 has breaking changes |
| numpy 1.26.4 | Last version before 2.0 | numpy 2.0 breaks many packages |
| gymnasium 0.29.1 | Stable Gymnasium (not Gym) | Gym is deprecated |
| scipy 1.12.0 | Compatible with numpy 1.26 | Later versions require numpy 2.0 |

## What Must NEVER Be Installed Early

| Package | Why Not |
|---------|---------|
| `cupy` | Requires CUDA at install time |
| `jax[cuda]` | Requires CUDA at install time |
| `triton` | Requires CUDA, installed by torch+cuda automatically |
| `flash-attn` | Requires CUDA, compile from source |
| `apex` | NVIDIA-specific, requires CUDA |
| `bitsandbytes` | Requires CUDA |

**Rule:** If package name contains "cuda", "cu11", "cu12", "gpu", or "nvidia", do not install on CPU machine.

## Arch Linux CUDA Pitfalls

### Problem 1: Rolling Release Driver Updates

Arch updates NVIDIA drivers frequently. Driver 545 works with CUDA 12.1. Driver 550 might not.

**Solution:**
```bash
# Pin driver version in /etc/pacman.conf
IgnorePkg = nvidia nvidia-utils
```

### Problem 2: CUDA Toolkit Version Mismatch

PyTorch cu121 expects CUDA 12.1 toolkit. Arch might have 12.3.

**Solution:**
```bash
# Install specific CUDA version
yay -S cuda-11.8  # or cuda-12.1
# Set environment
export CUDA_HOME=/opt/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Problem 3: cuDNN Version

PyTorch bundles cuDNN, but system cuDNN can interfere.

**Solution:**
```bash
# Do NOT install system cudnn
# PyTorch wheel includes correct cuDNN
```

### Problem 4: GCC Version

CUDA requires specific GCC versions. Arch has latest GCC.

**Solution:**
```bash
# Install older GCC for CUDA compilation
yay -S gcc11
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
```

## Environment Setup Commands

### CPU Environment (Today)

```bash
cd f1_digital_twin

# Create isolated environment
python -m venv .venv_cpu
source .venv_cpu/bin/activate

# Install CPU-only PyTorch
pip install torch==2.2.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install remaining dependencies
pip install -r requirements_cpu.txt

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
# Expected: PyTorch 2.2.0+cpu, CUDA available: False

# Run tests
pytest tests/ -v
```

### GPU Environment (GPU Day)

```bash
cd f1_digital_twin

# Create separate environment
python -m venv .venv_gpu
source .venv_gpu/bin/activate

# Install CUDA PyTorch
pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html

# Install remaining dependencies
pip install -r requirements_gpu.txt

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
# Expected: PyTorch 2.2.0+cu121, CUDA available: True

python -c "import torch; x = torch.randn(1000, 1000, device='cuda'); print(f'GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB')"
# Expected: GPU memory allocated: 4.0 MB
```

## Dependency Freeze Protocol

After any successful training run:

```bash
# Freeze exact versions
pip freeze > requirements_frozen_$(date +%Y%m%d).txt

# Commit to repo
git add requirements_frozen_*.txt
git commit -m "Freeze dependencies after successful run"
```

## Tuesday Protection Protocol

Before any work session:

```bash
# Check nothing changed
pip freeze > /tmp/current.txt
diff requirements_cpu.txt /tmp/current.txt

# If diff is non-empty, something changed. Investigate before proceeding.
```

Never run `pip install --upgrade` on a working environment. Create new environment instead.
