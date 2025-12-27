# PHASE 10 — GPU Ignition Day Checklist

## Prerequisites (Must Be True Before Starting)

- [ ] All tests pass on CPU: `pytest tests/ -v`
- [ ] Training completes 1000 steps on stub environment without error
- [ ] Config validation passes: `python scripts/validate_config.py`
- [ ] Git working tree is clean: `git status` shows no changes
- [ ] Current commit is tagged: `git tag cpu-verified-$(date +%Y%m%d)`

## Checklist (Execute In Order)

### 1. Verify NVIDIA Driver

```bash
nvidia-smi
```

Expected: Driver version 545+ displayed, RTX A4000 listed.

If fails: Stop. Fix driver first.

### 2. Create GPU Environment

```bash
python -m venv .venv_gpu
source .venv_gpu/bin/activate
```

### 3. Install CUDA PyTorch

```bash
pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html
```

### 4. Verify CUDA Available

```bash
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
```

If fails: Stop. Check CUDA installation.

### 5. Verify GPU Memory

```bash
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

Expected: RTX A4000, 16.0 GB

### 6. Install Remaining Dependencies

```bash
pip install -r requirements_gpu.txt
```

### 7. Run GPU Tests

```bash
pytest tests/ -v -k "not slow"
```

All tests must pass.

### 8. Run Single Training Step

```bash
python scripts/train.py --config configs/base.yaml --device cuda --max-steps 100
```

Expected: Completes without error, GPU memory usage visible in `nvidia-smi`.

### 9. Verify Checkpoint Compatibility

```bash
# Load CPU checkpoint on GPU
python -c "
import torch
ckpt = torch.load('experiments/cpu_test/checkpoints/step_1000.pt', map_location='cuda')
print('Checkpoint loaded successfully')
"
```

### 10. Run Full Training

```bash
python scripts/train.py --config configs/base.yaml --device cuda
```

Monitor with:
```bash
watch -n 1 nvidia-smi
```

Expected: GPU utilization 80%+, memory usage stable.

## Rollback Strategy

If anything fails:

1. Deactivate GPU environment: `deactivate`
2. Return to CPU environment: `source .venv_cpu/bin/activate`
3. Verify CPU still works: `pytest tests/ -v`
4. Document failure in `docs/gpu_issues.md`
5. Do not attempt GPU again until issue is understood

## What Must Be Tested First

| Test | Command | Expected |
|------|---------|----------|
| Import torch | `python -c "import torch"` | No error |
| CUDA available | `python -c "import torch; print(torch.cuda.is_available())"` | True |
| Tensor to GPU | `python -c "import torch; torch.randn(100).cuda()"` | No error |
| Model to GPU | `python -c "from src.models import WorldModel; WorldModel().cuda()"` | No error |
| Forward pass | `python -c "from src.models import WorldModel; import torch; m = WorldModel().cuda(); m(torch.randn(1,34).cuda(), torch.randn(1,3).cuda())"` | No error |
| Backward pass | Same as above + `.backward()` | No error |

## Success Criteria

GPU day is successful when:

1. `pytest tests/ -v` passes (all tests)
2. Training runs for 10,000 steps without error
3. GPU utilization is 70%+ during training
4. Memory usage is stable (no growth over time)
5. Checkpoints save and load correctly
6. Metrics match CPU training (within noise)

## Time Budget

| Task | Expected Time |
|------|---------------|
| Environment setup | 15 minutes |
| Dependency installation | 30 minutes |
| Verification tests | 15 minutes |
| First training run | 30 minutes |
| **Total** | 90 minutes |

If any step takes longer than 2x expected time, stop and investigate.
