# Pure-Transformer Image Classification on Fashion-MNIST

Registration Number: 2430010322

A transformer-only classifier (Swin-style hierarchical transformer, no CNN backbone) for Fashion-MNIST, evaluated with the same
split, augmentation, optimizer and metrics as the earlier CNN / ViT / Hybrid / ConvNeXt notebook.

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
| Swin-Tiny (this work) | _fill in from results/results.csv_ |

Earlier notebook, single seed, for comparison: Baseline CNN 0.9291, Hybrid Conv-Transformer 0.9272, CBAM-CNN 0.9097,
ConvNeXt-style 0.8947, ViT-Small 0.8636.

## Notes

* Results come from one seed unless the seed study in the notebook is enabled.
* Hyper-parameters are selected on validation accuracy only, never on the test set.
