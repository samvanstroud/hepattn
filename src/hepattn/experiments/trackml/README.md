# TrackML

Setup

```shell
srun --pty  --cpus-per-task 15  --gres gpu:l40s:2 --mem=100G -p GPU bash
apptainer shell --nv --bind /share/rcifdata/maxhart/data/trackml/ hepattn/pixi.sif
cd hepattn && pixi shell
cd hepattn/src/hepattn/experiments/trackml/
```

## Hit Filter

```shell
python run_filtering.py fit --config configs/filtering.yaml --trainer.fast_dev_run 10 --trainer.devices 2
```

## Tracking

```shell
python run_tracking.py fit --config configs/tracking.yaml --trainer.fast_dev_run 10
```


## Batch Submit

```shell
sbatch /share/rcifdata/svanstroud/hepattn/src/hepattn/experiments/trackml/submit_training_hypatia.sh
```
