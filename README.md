# Transformer & Modern-Architecture Evaluation on Fashion-MNIST
**CNN vs ViT vs Hybrid Conv-Transformer vs ConvNeXt-style**

| | |
|---|---|
| **Student Name** | Krishna |
| **Registration Number** | 2430010322 |
| **Section** | _[Fill in]_ |
| **Course / Project** | _[Fill in, e.g. DSE3120 Deep Learning]_ |
| **Submission date** | _[Fill in]_ |

---

## 1. Project Overview

This project compares five image classifiers on **Fashion-MNIST** (10 clothing classes, 28x28 grayscale)
under *identical* data splits, augmentation, optimizer and learning-rate schedule, so that only the
architecture changes:

| Family | Model | Idea |
|---|---|---|
| CNN (reference) | Baseline CNN, CBAM-CNN | Conv blocks, optional channel + spatial attention |
| Transformer | **ViT-Small** | Image cut into patches, pure self-attention encoder |
| CNN + Transformer | **Hybrid Conv-Transformer** | Conv stem for local features, transformer encoder for global context |
| Modern CNN | **ConvNeXt-style** | CNN redesigned with transformer-era choices (7x7 depthwise conv, LayerNorm, GELU, LayerScale) |

**Evaluation goes beyond accuracy:** per-class precision / recall / F1 / specificity, macro ROC-AUC,
calibration (ECE, NLL), efficiency (parameters, training time, latency), pairwise McNemar significance
tests, and attention-rollout visualisation for the ViT. An optional multi-seed study is included in the code
but **was not run**.

**Hypothesis stated before running:** transformers trained from scratch on 54k tiny images have little
inductive bias to lean on, so a plain ViT was expected to land at or below a good CNN, with the hybrid and
ConvNeXt-style models the most likely to match or beat it. Section 6 reports how that held up.

---

## 2. Folder Structure

```text
2430010322/
│
├── README.md                         <- this file
├── requirements.txt                  <- Python dependencies
│
├── data/
│   └── dataset_information.txt       <- dataset source, classes, split, preprocessing
│
├── src/
│   ├── model.py                      <- all 5 architectures + REGISTRY (builders, hyper-parameter grids)
│   ├── preprocessing.py              <- loading, normalisation, stratified split, dataset analysis
│   ├── train.py                      <- hyper-parameter tuning + final training (+ optional seed study)
│   └── test.py                       <- test-set evaluation, tables, plots, significance tests
│
├── notebooks/
│   └── experiments.ipynb             <- the end-to-end experiment notebook (the run behind every number below)
│
├── results/
│   ├── results.csv                   <- main metrics per model (accuracy, macro P/R/F1, specificity, AUC)
│   ├── comparison.csv                <- this work vs published reference results (references UNVERIFIED)
│   ├── confusion_matrix.png          <- row-normalised confusion matrices, all models
│   ├── training_curve.png            <- train/validation loss and accuracy curves, all models
│   ├── efficiency_calibration.csv    <- params, epochs, train time, latency, ECE, NLL
│   ├── mcnemar_pairwise.csv          <- pairwise McNemar exact tests
│   ├── tuning_results.csv            <- every hyper-parameter configuration tried
│   ├── convergence.csv               <- epochs run, best epoch, train-validation gap
│   ├── per_class_f1.csv              <- per-class F1 for every model
│   ├── per_class_baseline_cnn.csv    <- full per-class breakdown of the best model
│   ├── class_distribution.csv        <- class counts of the training set
│   ├── findings_summary.txt          <- key findings computed from the numbers above
│   └── SOURCE.txt                    <- where each file came from + known gaps (read this)
│
├── models/
│   ├── model_description.txt         <- architecture + training description of every model
│   └── best_hyperparameters.json     <- selected hyper-parameters (read by src/train.py and src/test.py)
│                                        (trained *.weights.h5 files are NOT included, see Section 7)
│
└── figures/
    ├── class_distribution.png        <- dataset analysis
    ├── sample_images.png             <- dataset analysis
    ├── per_class_f1_heatmap.png      <- where the architectures differ per class
    ├── mcnemar_pvalues.png           <- significance heatmap
    └── vit_attention_rollout.png     <- what the ViT attends to
```

`results/` and `figures/` were taken from the executed notebook (figures unchanged, tables transcribed
value-for-value). Running `src/train.py` + `src/test.py` regenerates them and also adds
`history_<model>.csv`, `training_times.csv`, `per_class_metrics.csv` and (with `--seed-study`)
`seed_study.csv`. Details and known gaps: `results/SOURCE.txt`.

---

## 3. Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires **Python 3.10+** and **Keras 3** (TensorFlow >= 2.16). A GPU is strongly recommended.
The reported results were produced on a Google Colab GPU runtime with TensorFlow 2.21.0 and Keras 3.13.2.
Fashion-MNIST is downloaded automatically on first use (see `data/dataset_information.txt`).

---

## 4. How to Run

Run all commands from the project root, in this order:

```bash
# 1. (optional) dataset analysis -> figures/class_distribution.png, figures/sample_images.png
python src/preprocessing.py

# 2. tune + train all five models -> models/*.weights.h5, results/history_*.csv, tuning_results.csv, ...
python src/train.py

# 3. evaluate on the held-out test set -> results/*.csv, results/*.png, figures/*.png
python src/test.py

# 4. (optional, slow, ~3x training time) seed-variance study -> results/seed_study.csv
python src/train.py --skip-tuning --seed-study
```

| Command | Purpose |
|---|---|
| `python src/train.py --quick` then `python src/test.py --quick` | Tiny subset, 2 epochs. **Pipeline check only; numbers are meaningless.** |
| `python src/train.py --models "ViT-Small"` | Train/tune a single model |
| `python src/train.py --skip-tuning` | Reuse `models/best_hyperparameters.json` (already contains the selection below) |
| `python src/train.py --epochs 50 --lr 5e-4` | Override training budget / learning rate |

`train.py` skips models that already have weights and a history file, so an interrupted run can be restarted.
If you ran `--quick`, delete the contents of `models/*.weights.h5` and `results/history_*.csv` before the real run.

`notebooks/experiments.ipynb` performs the same experiment end-to-end (set `QUICK_RUN = True` for a fast check).
The scripts in `src/` are the modular version of that notebook.

**Runtime (Colab GPU, from the executed notebook):** final training took about 48 minutes in total for the five
models (212 s to 871 s each, see Section 6.3); the 12-configuration tuning sweep is additional.

---

## 5. Method Summary

* **Data:** 60,000 official training images split 90/10 (stratified, seed 42) into 54,000 train and 6,000
  validation; the official 10,000 test images are used only for final evaluation. Classes are perfectly
  balanced (6,000 per class, imbalance ratio 1.0), so no resampling or class weights are used.
* **Preprocessing:** divide by 255, add channel axis. Training-time augmentation (horizontal flip, 10% translation)
  is built into every model identically.
* **Training:** AdamW (lr 1e-3, weight decay 0.05, grad-clip 1.0), 5% warm-up then cosine decay, batch 128,
  up to 30 epochs, early stopping on validation loss (patience 6, best weights restored).
* **Tuning:** 6-epoch grid per model (dropout in {0.1, 0.3}; ViT additionally patch size in {4, 7}), selected by
  **validation** accuracy only.
* **Evaluation:** accuracy, macro P/R/F1, macro specificity, macro one-vs-rest ROC-AUC, ECE (15 bins), NLL,
  latency per 1,000 images, McNemar exact test with Bonferroni correction (0.05 / 10 pairs = 0.005).

Full architecture details: `models/model_description.txt`.

---

## 6. Results

All numbers below come from one training run per model (seed 42) on the 10,000-image test set. Source files are
in `results/`.

### 6.1 Test-set performance (`results/results.csv`)

| Model | Family | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Specificity (macro) | ROC-AUC (macro) |
|---|---|---|---|---|---|---|---|
| **Baseline CNN** | CNN | **0.9291** | 0.9292 | 0.9291 | 0.9290 | 0.9921 | 0.9962 |
| Hybrid Conv-Transformer | CNN+Transformer | 0.9272 | 0.9268 | 0.9272 | 0.9269 | 0.9919 | 0.9961 |
| CBAM-CNN | CNN | 0.9097 | 0.9093 | 0.9097 | 0.9092 | 0.9900 | 0.9942 |
| ConvNeXt-style | Modern CNN | 0.8947 | 0.8949 | 0.8947 | 0.8946 | 0.9883 | 0.9924 |
| ViT-Small | Transformer | 0.8636 | 0.8625 | 0.8636 | 0.8629 | 0.9848 | 0.9885 |

Selected hyper-parameters: dropout 0.1 for every model; patch size 4 for ViT-Small (`results/tuning_results.csv`).

### 6.2 Statistical significance (`results/mcnemar_pairwise.csv`, `figures/mcnemar_pvalues.png`)

McNemar exact test, Bonferroni threshold 0.005. **Baseline CNN vs Hybrid Conv-Transformer: p = 0.422, not
distinguishable.** Every other pair of models differs significantly (p < 0.005) on this test set. This test
does not capture training-seed variance.

### 6.3 Efficiency and calibration (`results/efficiency_calibration.csv`)

| Model | Params | Epochs run | Train time (s) | Latency (ms / 1000 imgs) | ECE | NLL |
|---|---|---|---|---|---|---|
| Baseline CNN | 111,370 | 30 | 212.0 | 95.7 | 0.0102 | 0.2005 |
| Hybrid Conv-Transformer | 305,194 | 30 | 763.1 | 129.9 | 0.0147 | 0.2044 |
| CBAM-CNN | 117,295 | 17 | 243.0 | 105.6 | 0.0077 | 0.2494 |
| ConvNeXt-style | 221,146 | 30 | 810.2 | 165.4 | 0.0111 | 0.2822 |
| ViT-Small | 205,834 | 30 | 871.0 | 121.4 | 0.0254 | 0.3762 |

### 6.4 Per-class F1 (`results/per_class_f1.csv`, `figures/per_class_f1_heatmap.png`)

| Class | Baseline CNN | CBAM-CNN | ConvNeXt-style | ViT-Small | Hybrid |
|---|---|---|---|---|---|
| T-shirt/top | 0.888 | 0.866 | 0.841 | 0.800 | 0.884 |
| Trouser | 0.989 | 0.986 | 0.981 | 0.974 | 0.989 |
| Pullover | 0.901 | 0.869 | 0.830 | 0.780 | 0.897 |
| Dress | 0.927 | 0.901 | 0.900 | 0.875 | 0.924 |
| Coat | 0.895 | 0.849 | 0.817 | 0.789 | 0.894 |
| Sandal | 0.983 | 0.978 | 0.971 | 0.945 | 0.982 |
| **Shirt** | 0.793 | 0.742 | 0.717 | 0.647 | 0.792 |
| Sneaker | 0.961 | 0.959 | 0.950 | 0.911 | 0.957 |
| Bag | 0.988 | 0.980 | 0.977 | 0.967 | 0.988 |
| Ankle boot | 0.966 | 0.963 | 0.963 | 0.941 | 0.963 |

Hardest classes (mean F1 across models): Shirt 0.738, Coat 0.849, Pullover 0.855.

### 6.5 Convergence (`results/convergence.csv`, `results/training_curve.png`)

| Model | Epochs run | Best epoch (val loss) | Final train acc | Final val acc | Train-val gap |
|---|---|---|---|---|---|
| Baseline CNN | 30 | 30 | 0.9437 | 0.9358 | +0.0079 |
| CBAM-CNN | 17 | 11 | 0.9286 | 0.9150 | +0.0136 |
| ConvNeXt-style | 30 | 28 | 0.9056 | 0.9057 | -0.0000 |
| ViT-Small | 30 | 30 | 0.8547 | 0.8703 | -0.0157 |
| Hybrid Conv-Transformer | 30 | 27 | 0.9431 | 0.9378 | +0.0053 |

No model overfits (largest gap +0.014). Training accuracy is measured on augmented images, so a negative gap is
not a bug.

### 6.6 Comparison with published results (`results/comparison.csv`)

The five reference rows (VGG16 93.5%, GoogLeNet 93.7%, MobileNet 95.0%, DenseNet-BC 95.4%, WRN-28-10 + Random
Erasing 96.3%) are **unverified** (see Section 7). Taken at face value they all sit above the best model here
(92.91%). They are context, not a like-for-like contest: published models were trained longer, some with heavy
augmentation or pretraining.

---

## 7. Conclusion

**Best model.** The Baseline CNN is the most accurate (92.91%, macro-F1 0.9290, AUC 0.9962), but its margin over
the Hybrid Conv-Transformer (92.72%) is 0.19 points and not statistically distinguishable (McNemar p = 0.422).
With one seed and no seed study, the order of these two is not established; they should be read as tied.

**CNN vs transformer.** The hypothesis held for the pure transformer: ViT-Small is last at 86.36%, 6.55 points
below the Baseline CNN, with the worst calibration (ECE 0.0254, NLL 0.3762). Adding a convolutional stem
(Hybrid) recovers +6.36 points, which is consistent with locality and weight sharing mattering a lot when
training from scratch on 54k tiny images. The hypothesis did **not** hold for the ConvNeXt-style model: it was
expected to match or beat the CNNs but finished 3.44 points below the Baseline CNN (89.47%, significant).
Possible explanations, **not tested here**: a design built for much larger data and longer schedules, and the
short 30-epoch budget.

**The 30-epoch budget probably favours the CNNs.** ViT-Small's best validation loss came at epoch 30 of 30 and
ConvNeXt-style's at epoch 28, so both were still improving when training stopped. The conclusion is therefore
"under this budget", not "ViTs are worse at this task".

**Attention (CBAM) did not help.** CBAM-CNN scored 1.94 points below the plain CNN (significant). However, it
stopped early at epoch 17 (best epoch 11) after a noisy validation curve, so this comparison is not clean and
should not be read as "attention hurts".

**Where models differ.** Shirt is the hardest class for every model (F1 0.647 for ViT-Small to 0.793 for the
Baseline CNN), followed by Coat and Pullover. In `results/confusion_matrix.png`, roughly 9% of true Shirts are
predicted as T-shirt/top by the Baseline CNN and about 16% by ViT-Small. Trouser, Bag and Sandal are near-perfect
for all models. The ViT loses most on the upper-body classes.

**Cost trade-off.** The Baseline CNN is also the cheapest model on every efficiency measure: 111,370 parameters,
212 s training, 95.7 ms per 1,000 images. The Hybrid needs 2.7x the parameters, 3.6x the training time and 1.4x
the latency for a 0.19-point difference that is not significant. On this dataset the extra compute of the
transformer-based models bought nothing.

**Calibration.** All models are reasonably calibrated (ECE at most 0.0254). CBAM-CNN has the lowest ECE (0.0077),
the Baseline CNN the lowest NLL (0.2005); ViT-Small is worst on both.

**Attention rollout** (`figures/vit_attention_rollout.png`). For footwear, bags and trousers the ViT's
attention concentrates on the object; for several upper-body items (T-shirt, Pullover, Coat, Shirt) it also puts
weight on border patches. Rollout is an approximation, not a causal explanation.

**Tuning.** Dropout 0.1 beat 0.3 for all models and patch size 4 beat 7 for the ViT. Differences between
configurations are small for the CNNs (for example 0.9150 vs 0.9127), and a 6-epoch sweep favours low
regularisation, so these choices are only a coarse filter.

---

## 8. Limitations and Honest Caveats

* **Single dataset**, grayscale 28x28 images; conclusions may not transfer to larger or colour images.
* **Single seed** for all headline numbers; the seed study was not run. Gaps below roughly half a percentage
  point (such as Baseline CNN vs Hybrid) should not be treated as a ranking.
* **Coarse tuning:** 6 epochs, one seed, very small grids. ViT-Small and ConvNeXt-style were still improving at
  epoch 30, so a longer budget could change their position.
* **Published reference rows are unverified.** They were carried over from an earlier notebook and cited there as
  arXiv:1802.07589; that citation could not be confirmed. Verify each figure and its source against the original
  papers before relying on them, or remove `comparison.csv` rows you cannot source.
* **Reported numbers come from the Colab notebook, not from `src/`.** The scripts reproduce the same pipeline and
  were smoke-tested on synthetic data only; they have not been run on real Fashion-MNIST. A fresh run can differ
  slightly (GPU non-determinism), so do not mix numbers from the two runs.
* **No trained weights are shipped** (the notebook did not save them). `python src/train.py` creates them.
* The printed McNemar table was cut off; two of its ten rows are reconstructed from the heatmap (`results/SOURCE.txt`).
* Latency is wall-clock `predict` time on the Colab runtime; use it for relative comparison only.

---

## 9. References

1. Xiao, Rasul, Vollgraf. *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.* arXiv:1708.07747, 2017.
2. Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021 (ViT).
3. Liu et al. *A ConvNet for the 2020s.* CVPR 2022 (ConvNeXt).
4. Woo et al. *CBAM: Convolutional Block Attention Module.* ECCV 2018.
5. Abnar, Zuidema. *Quantifying Attention Flow in Transformers.* ACL 2020 (attention rollout).
6. Loshchilov, Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019 (AdamW).
7. McNemar, Q. *Note on the sampling error of the difference between correlated proportions or percentages.* Psychometrika, 1947.
8. Zhong et al. *Random Erasing Data Augmentation.* AAAI 2020.
