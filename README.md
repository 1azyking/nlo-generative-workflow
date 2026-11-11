# NLO generative workflow

Companion code for discovering inorganic nonlinear optical crystals, including the CDVAE generation interface, compositional and structural feature extraction, LightGBM classification, SI-based chemical-neutrality screening, and result analysis.

Yu et al., *J. Phys. Chem. Lett.* **2026**, 17, 6463–6471.
Paper: https://doi.org/10.1021/acs.jpclett.6c01026
Data and trained models: https://doi.org/10.6084/m9.figshare.31849795.v2

## Directory structure

```text
src/
  CGCNN/                Crystal graph data and network modules
  predictor/            Feature extraction, classifier training, and utilities
  generated/            Generated structures, ensemble predictions, and charge screening
  analysis/             Structure statistics
  resources/            Oxidation-state table from SI Table S3
scripts/data/           Materials Project data preparation
scripts/postprocess/    Structure export and optical result extraction
integrations/cdvae/     CDVAE sampling and evaluation interface
notebooks/              Composition, structure, stability, and phonon-path analyses
assets/cgcnn/           Atomic initialization features
environments/           Dependencies and environment guidance
data/                   Paper dataset sources
models/                 Trained model and official checkpoint sources
docs/                   Workflow interfaces and chemical-neutrality rules
licenses/               Third-party license notices
```

## Environment and installation

See [environments/README.md](environments/README.md) for dependencies. Install the project code in the corresponding environment:

```bash
python -m pip install -e . --no-deps
```

Packages under `src/` can be imported directly, for example `from generated.structures import pt_to_structures`.
Set the data, model, and output directories before using scripts or notebooks.
Provide Materials Project credentials through the `MP_API_KEY` environment variable; see `.env.example`.

## Data and models

- [Paper Figshare archive](https://doi.org/10.6084/m9.figshare.31849795.v2): CDVAE datasets, checkpoints and scalers; LightGBM feature datasets and four classifier ensembles; generated structures; and first-principles results.
- [Official CGCNN checkpoints](https://github.com/txie-93/cgcnn/tree/master/pre-trained): use formation-energy-per-atom.pth.tar with assets/cgcnn/atom_init.json.
- [Official ALIGNN model interface](https://github.com/usnistgov/alignn/blob/main/alignn/pretrained.py): formation energy uses the JARVIS model jv_formation_energy_peratom_alignn.

See [data/README.md](data/README.md) and [models/README.md](models/README.md) for paths and model links.
See [docs/WORKFLOW.md](docs/WORKFLOW.md) for workflow interfaces and [docs/CHEMICAL_NEUTRALITY.md](docs/CHEMICAL_NEUTRALITY.md) for charge-screening rules.

## Citation and third-party licenses

The paper citation is provided in `CITATION.cff`. Third-party notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`.
The paper's Figshare dataset is licensed under CC BY 4.0.
