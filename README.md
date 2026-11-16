# Optimized VisionAlphabet A — terminal raw G, ImageNet-1K

This branch runs the exact optimized A backbone used in the completed IN100 experiment, changing only the terminal energy readout to raw mean(2s·conj(u)−|u|²), with real/imaginary concatenation. No readout power normalization, log transform or denominator stop-gradient is applied. The classifier keeps BatchNorm(1024, affine=False) and uses Linear(1024,1000). Intermediate fused scans and path readers are unchanged. 3,910,504 parameters; scratch initialization, no trained IN100 weights.

Authorized run: seed0, 100epochs, effective batch1024, LR0.003. The established IN1K runtime retains AdamW, weight decay0.05, five-epoch warmup, cosine schedule, label smoothing0.1 and MixUp alpha0.8. Physical batch256 with accumulation4 is preferred; a CUDA OOM in the separate smoke process alone triggers fallback to128×8. The optimizer-update count differs from the earlier effective-batch256 protocol.

Entry: `bash h200/terminal_rawg/run.sh`. It verifies the pinned environment, whole H200 allocation and the complete IN1K dataset; runs tests and a full224px two-update smoke plus checkpoint restore; then trains from scratch. Any non-OOM preflight failure stops the launch. The existing supervisor reports logs, epochs and metrics to a dedicated relay and honors a remote stop. Latest/best checkpoints are local during training. At normal completion, latest/best, contracts and results are uploaded for external hash verification and private W&B archival before completion is acknowledged.

Source integrity: SOURCE_MANIFEST.json. Secrets are stored outside this repository and are never passed in the GitHub issue or public command. This branch uses a dedicated relay/observer, leaving previous H200 runs untouched.
