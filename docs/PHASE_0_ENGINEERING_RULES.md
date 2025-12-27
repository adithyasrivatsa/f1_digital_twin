# PHASE 0 — Engineering Rules (Why Systems Fail)

## Rule 1: No Floating Dependencies

**Why it exists:** PyTorch 2.1 behaves differently than 2.0. A `pip install torch` today gives different code than tomorrow.

**What breaks:** Training that worked Monday fails Wednesday. Gradients differ. Checkpoints incompatible. You spend 3 days debugging what was never your code.

**Enforcement:**
- Pin EVERY dependency to exact version
- Use `pip freeze > requirements.txt` after verified working state
- Never use `>=` or `~=` in production requirements

## Rule 2: CPU and GPU Environments Are Separate

**Why it exists:** PyTorch CPU and PyTorch CUDA are different packages. Installing CUDA PyTorch on CPU machine pulls 2GB of useless binaries and creates import-time failures.

**What breaks:** `import torch` fails silently or throws CUDA initialization errors on CPU-only machine. You cannot test your code.

**Enforcement:**
- `requirements_cpu.txt` — torch CPU wheel only
- `requirements_gpu.txt` — torch+cu118 or cu121
- Never mix. Never "upgrade" CPU env to GPU.

## Rule 3: Simulator Is External, Always

**Why it exists:** Simulators (Assetto Corsa, rFactor, custom physics) have their own update cycles, APIs, and bugs. Coupling ML code to simulator internals creates untestable systems.

**What breaks:** Simulator updates, your training breaks. Simulator unavailable, you cannot run unit tests. Simulator has race condition, your policy looks broken.

**Enforcement:**
- Simulator communicates via defined interface only (see `src/env/interface.py`)
- All ML code runs against stub environment during development
- No simulator imports in `src/models/` or `src/training/`

## Rule 4: Seeds Are Sacred

**Why it exists:** RL is noisy. Without fixed seeds, you cannot distinguish "code change improved performance" from "random variance."

**What breaks:** You tune hyperparameters for a week. Results are noise. You cannot reproduce your best run. Reviewers cannot verify claims.

**Enforcement:**
- Global seed set once at entry point
- Seed propagated to: numpy, torch, random, environment
- Seed logged with every experiment
- No `random.random()` calls in core logic

## Rule 5: No Silent Device Routing

**Why it exists:** Code that "auto-detects" GPU creates machines where behavior differs based on hardware. A tensor on CPU cannot operate with tensor on GPU.

**What breaks:** Code works on your GPU machine, fails on CI. Code works in training, fails in evaluation. Device mismatch errors appear deep in stack.

**Enforcement:**
- Device is explicit config parameter
- All tensor creation uses `device=config.device`
- No `torch.cuda.is_available()` in model code
- Device checks happen once at startup, fail loudly

## Rule 6: No Hardcoded Paths

**Why it exists:** `/home/yourname/project/data` does not exist on any other machine.

**What breaks:** Collaborator clones repo, nothing works. CI fails. You move project, everything breaks.

**Enforcement:**
- All paths relative to project root or config-specified
- Use `pathlib.Path` exclusively
- Environment variables for machine-specific roots only

## Rule 7: Experiments Are Immutable

**Why it exists:** Overwriting experiment results destroys history. You cannot compare runs if old runs are gone.

**What breaks:** You run experiment, forget to save config. You cannot reproduce. You overwrite best checkpoint with worse one.

**Enforcement:**
- Each experiment gets unique directory: `experiments/{timestamp}_{name}/`
- Config copied to experiment directory at start
- No overwrites, only new directories
- Git hash logged with experiment

## Rule 8: No Print Debugging in Production Code

**Why it exists:** Print statements are not structured. They cannot be filtered, searched, or disabled.

**What breaks:** Training output is 10GB of noise. You cannot find the error. Log files are useless.

**Enforcement:**
- Use `logging` module exclusively
- Log levels: DEBUG for development, INFO for production
- Structured logging for metrics (JSON or CSV)

## Rule 9: Tests Run Without External Dependencies

**Why it exists:** If tests require GPU, simulator, or network, they will not run in CI, on laptop, or when those resources are unavailable.

**What breaks:** You skip tests because they are slow. Bugs accumulate. Refactoring becomes terrifying.

**Enforcement:**
- All tests use stub/mock environments
- Tests complete in < 60 seconds total
- No network calls in tests
- GPU tests marked and skipped on CPU

## Rule 10: No Magic Numbers

**Why it exists:** `0.001` in code means nothing. Is it learning rate? Epsilon? Threshold? You will forget.

**What breaks:** You tune the wrong constant. You cannot find where value is set. Code review is impossible.

**Enforcement:**
- All numeric parameters in config
- Constants have names: `VELOCITY_CLIP_MAX = 350.0  # km/h, F1 physical limit`
- No unnamed numbers in algorithm code

## Rule 11: Fail Fast, Fail Loud

**Why it exists:** Silent failures propagate. A NaN in layer 1 becomes garbage output in layer 10.

**What breaks:** Training runs for 12 hours, produces garbage. You do not know when it went wrong. Debugging is archaeology.

**Enforcement:**
- Assert preconditions at function entry
- Check for NaN/Inf after each forward pass during debug
- Raise exceptions, do not return error codes
- Validate config at startup, not at use time

## Rule 12: No Premature Optimization

**Why it exists:** Optimized code is hard to read, hard to debug, hard to modify. You do not know where the bottleneck is until you profile.

**What breaks:** You spend a week optimizing data loading. Bottleneck was model forward pass. Optimized code has bugs you cannot find.

**Enforcement:**
- Write clear code first
- Profile before optimizing
- Document why optimization was needed
- Keep unoptimized version in comments if optimization is non-obvious
