# Pure-Transformer Image Classification on Fashion-MNIST

Name: Krishna Gupta

Registration Number: 2430010322

Section: D

A transformer-only classifier (Swin-style hierarchical transformer, no CNN backbone) for Fashion-MNIST, evaluated with the same split, augmentation, optimizer and metrics as the earlier CNN / ViT / Hybrid / ConvNeXt notebook.

## Motivation

**Question:** Can a transformer-only model, with no CNN backbone, match a CNN on small, low-resolution images when trained from scratch?

- A plain ViT-Small reached 86.4% test accuracy versus 92.9% for a small CNN (earlier notebook). A flat ViT has no locality prior, and 54k images of 28x28 is too little data to learn one.
- This project tests a Swin-style hierarchical transformer. Shifted-window attention builds locality *inside* the attention mechanism, so the model stays a pure transformer.
- The comparison is controlled: same dataset, stratified split, augmentation, optimizer, schedule and evaluation protocol as the earlier baselines.
- Evaluation goes beyond accuracy: calibration (ECE, NLL), latency, bootstrap confidence intervals, McNemar test, seed variance and saliency maps.

## Folder structure

```
2430010322/
├── README.md                    this file
├── requirements.txt             Python dependencies
├── data/
│   └── dataset_information.txt  dataset source, split and preprocessing (data is streamed by Keras, not stored here)
├── src/
│   ├── model.py                 Swin-style transformer (windowed attention, shifted windows, patch merging)
│   ├── preprocessing.py         loading, normalisation, stratified train/validation split
│   ├── train.py                 optional tuning + final training; saves weights, config, training curve
│   └── test.py                  test-set evaluation; saves confusion matrix, results.csv, comparison.csv
├── notebooks/
│   └── experiments.ipynb        full experiment notebook (tuning, training, metrics, calibration, significance, saliency)
├── results/                     filled by train.py / test.py (confusion_matrix.png, training_curve.png, results.csv, comparison.csv)
├── models/
│   └── model_description.txt    architecture and training description; weights + config.json appear after training
└── figures/                     extra figures exported from the notebook
```

## How to run

```bash
pip install -r requirements.txt
python src/train.py --tune     # sweep dropout / drop-path, then train (drop --tune to use defaults)
python src/test.py             # evaluate on the 10,000 test images
```

Use `--quick` on either script for a tiny pipeline check. A GPU is strongly recommended. The notebook in
`notebooks/experiments.ipynb` runs the same pipeline end to end and adds per-class metrics, calibration, bootstrap
confidence intervals and saliency maps.

## Results

`results/` is empty until you run the scripts or the notebook. Fill in the final numbers here after your run:

| Model | Test accuracy |
|---|---|
| Swin-Style (small), this work | _fill in from results/results.csv_ |

Earlier notebook, single seed, for comparison: Baseline CNN 0.9291, Hybrid Conv-Transformer 0.9272, CBAM-CNN 0.9097,
ConvNeXt-style 0.8947, ViT-Small 0.8636.

## Future Scope

- **Data-scarce domains:** locality-aware transformers could suit areas with few labelled images, such as medical imaging and industrial defect inspection.
- **Edge deployment:** windowed attention scales roughly linearly with image size, which makes small hierarchical transformers candidates for on-device vision.
- **Unified models:** a transformer vision backbone can share one architecture with text and multimodal models.
- **Retail and fashion:** automated catalogue tagging, visual search, recommendation, and returns/inventory sorting.
- **Trustworthy AI:** calibration and saliency analysis support confidence-aware, explainable predictions.

> Fashion-MNIST results do not automatically transfer to real-world images. The applications above are research directions, not validated claims.


## Notes

* Results come from one seed unless the seed study in the notebook is enabled.
* Hyper-parameters are selected on validation accuracy only, never on the test set.
* The patch embedding is implemented with a stride-`p` `Conv2D`. This is mathematically the same linear projection of non-overlapping `p x p` patches used in every ViT/Swin, not a convolutional feature extractor.
