# VisionAlphabet

This is the canonical VisionAlphabet project directory. The vision model,
experiment runners, tests, paper, and reviewer materials live here as real
files, not as duplicate copies.

| Directory | Purpose |
| --- | --- |
| `src/lnet/` | Vision backbone, complex spatial scan, and optimized kernels |
| `scripts/` | Vision training, evaluation, queue, and benchmark entry points |
| `tests/` | Vision model and experiment regression tests |
| `paper/` | Live ICLR manuscript, figures, tables, and analysis |
| `reviewer-code/` | Frozen reviewer reproduction package and checkpoints |
| `releases/` | Previously generated submission archives |
| `h200/`, `side_experiments/`, `figures/` | Vision experiment support and assets |
| `docs/`, `outputs/`, `artifacts/` | Hardware notes, local measurements, and generated visuals |

Work on model code in `src/lnet/` and `scripts/`, not in `reviewer-code/`.
The latter is an export and must be regenerated after a model change. Work on
the manuscript in `paper/`; its web preview uses `paper/build/main.pdf`.

The neighboring `lnet` checkout keeps compatibility links for the previous
model, script, test, and reviewer-package paths. Its old paper-project path is
an empty Docker mount point; edit the manuscript here in `paper/`, not through
that host path. The web services mount this directory directly. The old
`src/lnet/` still owns the separate ALPHABET-LM experiment, while
`scripts/paper_platform/` and `tests/paper_platform/` remain with the paper
workspace application. New VisionAlphabet development belongs here.

The live paper mount is declared in
`/home/qlab/.local/state/paper-workspace/accounts/project-mounts.yaml`.
Do not replace the empty legacy project mount point with a symlink: the
project-runtime service rejects symlinked project roots.

From this directory, with dependencies installed:

```bash
PYTHONPATH=src:scripts python -m pytest -q tests
PYTHONPATH=src:scripts python scripts/run_lnet_k128_p160_160_160_128_d2262_imagenet1k.py --help
```

The second command only inspects the launcher; training requires data and an
explicit output directory. To refresh the manuscript preview, run this from
`paper/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
```

## Project map and rebuttal preparation

- [코드·폴더 구조 안내](docs/PROJECT_MAP.ko.md)
- [2026-11-06 리뷰 쟁점·근거·영문 리버탈 초안](docs/rebuttal-2026-11-06/README.md)
