# PAFT

Fine-tuning that attaches a linear bottleneck and a classification head to the penultimate
layer of a classifier and trains them jointly with it, so that the representation is
preserved more faithfully when it is projected to two dimensions.

## Layout

| file | what it does |
|---|---|
| `src/model.py` | backbone, optional narrow penultimate layer, main head, auxiliary bottleneck and head |
| `src/objective.py` | cross-entropy plus one coefficient times three terms: prediction consistency between the two heads, CKA alignment and neighborhood preservation between the penultimate and the bottleneck representation. Dropping terms gives the objective-term ablations |
| `src/train.py` | stage A: train one cell, that is one (backbone, dataset, arm, seed) |
| `src/evaluate.py` | stage B: fit the four projections (PCA, t-SNE, UMAP, PaCMAP) and compute the measures |
| `src/metrics.py` | the 20 projection-quality measures and the four families they group into |
| `src/stats.py` | the accuracy gate and the win-loss score |
| `src/compare.py` | selection on validation data, comparisons on test data |
| `src/registry.py`, `src/paths.py`, `src/data.py` | the grid, the output paths, the datasets |
| `protocol.json` | splits, schedule, seeds, early stopping, arms and doses |

## Run

```bash
export PAFT_ROOT=$PWD
python src/train.py    --backbone r50 --dataset cub200 --cell kl_cka_nbr_1_d64 --seed 0
python src/evaluate.py --backbone r50 --dataset cub200 --cell kl_cka_nbr_1_d64 --seed 0
python src/compare.py
```

`run_train.py` and `run_evaluate.py` walk the whole grid, and any part of it runs with for
example `python run_train.py --backbones r50 --datasets cub200`. The stages write per-seed outputs
under `results/` and their aggregates under `analysis/`; neither is shipped here, since the
feature and embedding archives are hundreds of gigabytes.

## Two conventions used throughout

* **Accuracy gate.** A setting is admissible if its mean validation accuracy is at least
  the baseline's; otherwise it is rejected when the relative drop reaches 1% or a one-sided
  one-sample t-test rejects at p <= 0.05. When no setting of an arm is admissible, that arm
  falls back to the untreated baseline.
* **Win-loss difference.** Per measure, a paired t-test over the shared seeds,
  Benjamini-Hochberg at q = 0.05 within the comparison, and a 1% relative threshold; the
  score is wins minus losses.

## Environment

Python 3.10, one GPU per training run. t-SNE comes from openTSNE rather than
scikit-learn, whose implementation cannot transform out-of-sample points.

```
torch==2.6.0+cu124
torchvision==0.21.0+cu124
timm==1.0.26
numpy==1.26.4
scipy==1.15.3
scikit-learn==1.6.0
pandas==2.3.2
umap-learn==0.5.11
statsmodels==0.14.5
openTSNE==1.0.4
pacmap==0.9.1
```
