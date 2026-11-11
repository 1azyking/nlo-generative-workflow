# CDVAE interface

`evaluate.py` runs within the full CDVAE source tree and depends on `scripts/eval_utils.py` and the `cdvae` package.
Use this file as `scripts/evaluate.py` in the corresponding CDVAE working copy, and set the model path and sampling parameters for the trained checkpoint.

The interface includes reconstruction, generation, and optimization. The paper workflow uses generation; it does not optimize NLO properties in latent space using NLO labels.
Checkpoints, hparams.yaml, and scalers are available from the [paper Figshare archive](https://doi.org/10.6084/m9.figshare.31849795.v2).
