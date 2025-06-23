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
