# TrackML

Setup

```shell
srun --pty  --cpus-per-task 15  --gres gpu:l40s:1 --mem=100G -p GPU bash
apptainer shell --nv --bind /share/rcifdata/maxhart/data/trackml/ hepattn/pixi.sif
cd hepattn && pixi shell
cd hepattn/src/hepattn/experiments/trackml/
```

## Data
You can use the `detectors.csv` file available in `data/trackml/` directory.
This file was obtained by unzipping the file provided on the [Kaggle trackML webpage](https://www.kaggle.com/competitions/trackml-particle-identification/data).
For training, use the codalab data from [this webpage](https://competitions.codalab.org/competitions/20112#participate-get_data).
Sample data is available for testing in the `data/trackml/raw/` directory.

## Hit Filter

```shell
# train
python run_filtering.py fit --config configs/filtering.yaml --trainer.fast_dev_run 10

# test
python run_filtering.py test --config PATH

# evaluate on train/val
python run_filtering.py test --config PATH --data.test_dir /share/rcifdata/maxhart/data/trackml/prepped/train
```

## Tracking

```shell
# train
python run_tracking.py fit --config configs/tracking.yaml --trainer.fast_dev_run 10

# test
python run_tracking.py test --config PATH
```


## Batch Submit

```shell
sbatch /share/rcifdata/svanstroud/hepattn/src/hepattn/experiments/trackml/submit/submit_trackml_filtering.sh
sbatch /share/rcifdata/svanstroud/hepattn/src/hepattn/experiments/trackml/submit/submit_trackml_tracking.sh
```


## Configurations

A full pixel detector with a reasonable pt cut for targeting >1GeV particles:

```yaml
hit_volume_ids: [7, 8, 9] # pixel barrel and endcaps
particle_min_pt: 0.9
particle_max_abs_eta: 4.0
```





A more lightweight configuration for testing purposes:

```yaml
hit_volume_ids: [8] # pixel barrel only
particle_min_pt: 1.0
particle_max_abs_eta: 2.5
```

For which the following pre-trained hit filter evaluations can be used:

```yaml
hit_eval_train: /share/rcifdata/svanstroud/hepattn/src/hepattn/experiments/trackml/logs/HC-v3-lite_20250620-T114025/ckpts/epoch=016-val_loss=0.15730_train_eval.h5
hit_eval_val: /share/rcifdata/svanstroud/hepattn/src/hepattn/experiments/trackml/logs/HC-v3-lite_20250620-T114025/ckpts/epoch=016-val_loss=0.15730_val_eval.h5
hit_eval_test: /share/rcifdata/svanstroud/hepattn/src/hepattn/experiments/trackml/logs/HC-v3-lite_20250620-T114025/ckpts/epoch=016-val_loss=0.15730_test_eval.h5
```
