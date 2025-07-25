## Running the model

```
cd hepattn
apptainer shell --nv --bind /share/ pixi.sif
pixi shell
cd hepattn/src/hepattn/experiments/clic/
python main.py fit --config configs/base.yaml
```

## CLIC Data

At UCL, files are available on `plus1` under `/unix/atlastracking/svanstroud/dmitrii_clic`, and also on `hypatia` under `/share/gpu1/syw24/dmitrii_clic`.

| File Name | Purpose / Usage | Preprocessing Applied | Notes / Details |
| :------------------------------ | :------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train_clic` | For **training** the model. | "Train-like" preprocessing | Applies cuts on tracks, topoclusters, and truth particles; creates target incidence matrix. |
| `val_clic` | For **validation** during model development. | "Train-like" preprocessing | Applies cuts on tracks, topoclusters, and truth particles; creates target incidence matrix. |
| `test_clic_raw.root` | For **performance evaluation**. | **None** ("raw" file) | - |
| `test_clic_fix.root` | Used by MPflow to compare preprocessed targets with model predictions. | "Train-like" preprocessing | Applies cuts on tracks, topoclusters, and truth particles; creates target incidence matrix. |
| `test_clic_common_raw.root` | For **performance evaluation**. | **None** ("raw" file) | Contains the **same events as Nilotpal's evaluation**. |
| `test_clic_common_infer.root` | Evaluates the **real performance** of the model during inference. | "Infer-like" preprocessing | Does not apply cuts; converts CLIC format, removes unused variables, correctly defines truth particles. Should be launched with `data.is_inference true` flag. Contains the **same events as Nilotpal's evaluation**. |

**Definition of Preprocessing Types:**

* **"Train-like"**: Applies cuts on tracks, topoclusters, and truth particles, and creates target incidence matrix.
* **"Infer-like"**: No cuts applied; converts CLIC format, removes unused variables, and correctly defines truth particles.
* **"Raw"**: Original CLIC files with correctly defined truth particles.

**"Truth Particles"**: Refer to Section 5.1 in [https://arxiv.org/pdf/2410.23236](https://arxiv.org/pdf/2410.23236).