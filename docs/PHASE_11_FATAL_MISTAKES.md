# PHASE 11 — Common Fatal Mistakes

## Mistake 1: "I'll Just Upgrade This One Package"

Looks harmless: `pip install --upgrade numpy`

Destroys weeks: numpy 2.0 breaks scipy, pandas, and half of PyTorch internals. Your training code now produces different results. Your old checkpoints are incompatible.

Rule: Never upgrade. Create new environment instead.

## Mistake 2: "The Default Seed Is Fine"

Looks harmless: Not setting seed, or setting seed=0.

Destroys weeks: You tune hyperparameters. Results improve. You cannot reproduce. Was it the hyperparameters or random variance? You will never know.

Rule: Explicit seed in config. Log seed with every experiment.

## Mistake 3: "I'll Add Logging Later"

Looks harmless: Training without metrics logging.

Destroys weeks: Training fails at step 500,000. When did it start failing? What changed? You have no data. You restart from scratch.

Rule: Log from step 0. Log everything. Storage is cheap.

## Mistake 4: "This Hardcoded Value Is Temporary"

Looks harmless: `learning_rate = 0.001` in code.

Destroys weeks: You forget it is there. You tune learning rate in config. Nothing changes. You spend days debugging why learning rate has no effect.

Rule: No magic numbers. All parameters in config.

## Mistake 5: "I'll Clean Up The Code Later"

Looks harmless: Quick hack to test an idea.

Destroys weeks: The hack works. You build on it. More hacks. Now you have 5000 lines of spaghetti. Refactoring breaks everything. You cannot add features.

Rule: Clean code from day 1. Technical debt compounds.

## Mistake 6: "The Simulator Works, I Saw It"

Looks harmless: Testing simulator manually, not programmatically.

Destroys weeks: Simulator has race condition. Works 99% of the time. Fails randomly during overnight training. You wake up to garbage results.

Rule: Automated tests for everything. If it is not tested, it is broken.

## Mistake 7: "I'll Use The Latest PyTorch"

Looks harmless: `pip install torch` (gets latest).

Destroys weeks: Latest PyTorch has breaking change in autograd. Your custom backward pass silently produces wrong gradients. Policy learns garbage.

Rule: Pin exact versions. Test before upgrading.

## Mistake 8: "Reward Shaping Is Easy"

Looks harmless: Adding reward term for "going fast".

Destroys weeks: Agent learns to spin in circles (high angular velocity = "fast"). Or drives backwards. Or exploits physics glitch. Reward hacking is creative.

Rule: Every reward term must be bounded and physically meaningful. Test with adversarial mindset.

## Mistake 9: "I'll Just Train Longer"

Looks harmless: Training not converging, increase steps.

Destroys weeks: Training is diverging, not converging. More steps = more divergence. You wait 3 days for garbage.

Rule: Monitor stability metrics. Stop early if diverging.

## Mistake 10: "The GPU Will Make It Faster"

Looks harmless: Moving to GPU before CPU works.

Destroys weeks: Bug exists in code. On CPU, you get clear error message. On GPU, you get "CUDA error: unspecified launch failure". Debugging is impossible.

Rule: CPU first. GPU only after CPU works perfectly.

## Mistake 11: "I Don't Need Version Control For Experiments"

Looks harmless: Running experiments without git commits.

Destroys weeks: Best result ever. What code produced it? You modified 5 files since then. You cannot reproduce.

Rule: Commit before every experiment. Tag successful runs.

## Mistake 12: "This Warning Is Probably Fine"

Looks harmless: Ignoring deprecation warning.

Destroys weeks: Warning becomes error in next version. You upgrade (see Mistake 1). Code breaks. You do not remember what the warning was about.

Rule: Fix warnings immediately. Warnings are future errors.

## Mistake 13: "I'll Use Float16 For Speed"

Looks harmless: `model.half()` for faster training.

Destroys weeks: Gradients underflow. Loss becomes NaN. Training produces garbage. You do not notice for 100,000 steps because you were not monitoring.

Rule: Float32 until everything works. Float16 only with gradient scaling and careful monitoring.

## Mistake 14: "The Environment Is Stateless"

Looks harmless: Assuming `env.reset()` gives fresh state.

Destroys weeks: Environment has hidden state (RNG, cached values). Results depend on order of episodes. Reproducibility is impossible.

Rule: Explicit seed on every reset. Verify determinism with tests.

## Mistake 15: "I'll Parallelize Later"

Looks harmless: Single-environment training.

Destroys weeks: Single environment is 10x slower. You wait 10x longer for results. You make 10x fewer experiments. You learn 10x slower.

Rule: Vectorized environments from day 1. The infrastructure cost is fixed; the speedup is permanent.

## Mistake 16: "My Normalization Is Correct"

Looks harmless: Normalizing observations by hand.

Destroys weeks: Off-by-one in normalization. One feature is 100x larger than others. Network ignores all other features. Policy is blind.

Rule: Validate normalization with statistics. All features should have similar scale.

## Mistake 17: "I'll Handle Edge Cases Later"

Looks harmless: Not handling episode termination properly.

Destroys weeks: Value bootstrap at terminal state. Reward is attributed to wrong actions. Policy learns superstitions.

Rule: Handle termination explicitly. Test with short episodes.

## Mistake 18: "The Loss Is Decreasing, So It's Working"

Looks harmless: Only monitoring loss.

Destroys weeks: Loss decreases because policy collapsed to constant action. Entropy is zero. Policy is useless.

Rule: Monitor entropy, KL, explained variance. Loss alone means nothing.

## Mistake 19: "I Can Eyeball The Results"

Looks harmless: Watching agent drive, declaring success.

Destroys weeks: Agent looks good on one track section. Fails catastrophically on another. You did not notice because you were not measuring.

Rule: Quantitative evaluation. Multiple episodes. Statistical significance.

## Mistake 20: "Documentation Is For Later"

Looks harmless: Not documenting design decisions.

Destroys weeks: You return after 2 weeks. Why is this parameter 0.95? Why is this layer 256 units? You do not remember. You change it. Everything breaks.

Rule: Document why, not what. Code shows what. Comments show why.

## Summary

Every mistake above:
- Looks harmless in the moment
- Compounds over time
- Is obvious in hindsight
- Is preventable with discipline

The difference between a working system and a failed project is not intelligence. It is discipline.
