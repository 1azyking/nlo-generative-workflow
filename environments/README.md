# Environments

## Feature extraction and classification

Dependency lists:

- requirements-ml.in: feature extraction, classifiers, and pretrained-model interfaces.
- requirements-analysis.in: statistics, visualization, Materials Project queries, and crystal-analysis tools.

```bash
python -m pip install -r environments/requirements-ml.in
python -m pip install -r environments/requirements-analysis.in
python -m pip install -e . --no-deps
```

The ML dependencies are not version-locked. Use mutually compatible versions of ALIGNN, DGL, JARVIS, PyTorch, matminer, and pymatgen. Existing classifier files also require compatible LightGBM/scikit-learn serialization formats.

## CDVAE

CDVAE uses a separate environment and its full source tree. Key versions in the cdvaecpu reference environment are:

| Component | Reference version |
|---|---|
| Python | 3.9.18 |
| PyTorch | 1.8.1 |
| PyTorch Lightning | 1.3.8 |
| torch-geometric | 2.0.1 |
| hydra-core | 1.1.0 |
| omegaconf | 2.1.2 |

The sampling interface depends on CDVAE's scripts/eval_utils.py, model configuration, and GemNet companion files. See [integrations/cdvae/README.md](../integrations/cdvae/README.md).
