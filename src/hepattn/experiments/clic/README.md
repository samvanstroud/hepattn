# Glow: Particle Flow with CLIC

This work is described in our preprint: [GLOW: A Unified Transformer for Diverse Reconstruction Tasks in Particle Physics](https://arxiv.org/abs/2508.20092)

We present GLOW, a transformer-based particle flow model that combines incidence matrix supervision from HGPflow with a MaskFormer architecture. Evaluated on CLIC detector simulations, GLOW achieves state-of-the-art performance and, together with prior work, demonstrates that a single unified transformer architecture can effectively address diverse reconstruction tasks in particle physics.

## Running the Model

First, set up your environment (see the top level [README.md](../../../../README.md) for more details):

```shell
git clone git@github.com:samvanstroud/hepattn.git
cd hepattn
apptainer shell --nv --bind /share/ pixi.sif
pixi shell -e clic
cd hepattn/src/hepattn/experiments/clic/
```

To run the model, use the following commands:

```shell
# interactive job
python main.py fit --config configs/base.yaml

# slurm batch
sbatch hepattn/src/hepattn/experiments/clic/submit_training_sam.sh
```

## Evaluation

To evaluate a trained model, run the following command:

```shell
python main.py test \
    --config <path to config.yaml> \
    --data.test_path test_clic_common_infer.root \
    --data.is_inference true \
    --trainer.precision 32-true \
    --matmul_precision highest
```

**Important Notes:**
- Flags `--data.is_inference true` and `--trainer.precision 32-true` are required for correct evaluation of model performance.
- Change the attention type to `torch` in the config file.
- Remove the compile callback if present in the config file.

You can then produce the performance plots using the [provided notebook](notebooks/performance.ipynb).
To start a Jupyter notebook on a compute node:

```shell
jupyter notebook --no-browser --ip=0.0.0.0 --port 8888
```

## CLIC Data

### Data Locations

- **plus1**: `/unix/atlastracking/svanstroud/dmitrii_clic`
- **hypatia**: `/share/gpu1/syw24/dmitrii_clic`
- **Isambard**: `/projects/u5ar/data/clic`



### Data Files Overview

| File Name | Purpose | Preprocessing | Notes |
| :------------------------------ | :------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train_clic` | Training | Train-like | Cuts on tracks/topoclusters/truth particles; creates target incidence matrix |
| `val_clic` | Validation | Train-like | Cuts on tracks/topoclusters/truth particles; creates target incidence matrix |
| `test_clic_raw.root` | Performance evaluation | None (raw) | - |
| `test_clic_fix.root` | MPflow comparison | Train-like | Cuts on tracks/topoclusters/truth particles; creates target incidence matrix |
| `test_clic_common_raw.root` | Performance evaluation | None (raw) | Same events as Nilotpal's evaluation |
| `test_clic_common_infer.root` | Inference evaluation | Infer-like | No cuts; CLIC format conversion; correct truth particles; use `data.is_inference true` |

### Preprocessing Definitions

- **"Train-like"**: Applies cuts on tracks, topoclusters, and truth particles, and creates target incidence matrix.
- **"Infer-like"**: No cuts applied; converts CLIC format, removes unused variables, and correctly defines truth particles.
- **"Raw"**: Original CLIC files with correctly defined truth particles.

**Truth Particles**: Refer to Section 5.1 in [https://arxiv.org/pdf/2410.23236](https://arxiv.org/pdf/2410.23236).