# hep-attn

Goals:
- cleanly switch between different attention backends (sdpa, flash, flex)
- include some recent transformer advances (layerscale, value residuals, local attention)
- full torch.compile and nested tensor support
- don't use conda

## Setup

First clone the repo

```shell
git clone git@github.com:samvanstroud/hepattn.git
cd hepattn
```

We recommend using a container to setup and run the code.
This is necessary if your system's `libc` is version is `<2.28` 
due to requirements of recent `torch` versions.
We use `pixi`'s cuda image, which you can access with

```shell
apptainer pull pixi.sif docker://ghcr.io/prefix-dev/pixi:noble-cuda-12.6.3
apptainer shell --nv pixi.sif
```

Note: if you are not using the `pixi` container, you will need to make sure 
`pixi` is installed according to https://pixi.sh/latest/. 

You can then install the project with locked dependencies with

```shell
pixi install --locked
```

Note that `flash-attn` is commented out in the [`pyproject.toml`](pyproject.toml).
In order to install it, first install the package as above, then uncomment the 
`flash-attn` requirement, and rerun the installation. This is because `flash-attn`
depends on `torch` in order to be installed.

## Contributing

If you would like to contribute, please set up pre-commit hooks with

```shell
pre-commit install
```



## Activing the environment

To run the installed environment, use

```shell
pixi shell
```

You can close the environment with `exit`.

See the [`pixi shell` docs](https://pixi.sh/latest/reference/cli/pixi/shell/) for more info.

## Running tests

Once inside the environemnt, just run 

```shell
pytest
```

## Run experiments

TrackML:

```shell
cd src/hepattn/experiments/trackml/
python hit_filter.py fit --config hit_filter.yaml --trainer.fast_dev_run 10
```

## Todo

- [ ] maskformer
    - [ ] mask decoder
    - [ ] matcher
    - [ ] maskformer loss
    - [ ] order queries by phi in decoder
- [ ] pe
    - [x] positional embeddings from hepformer repo
    - [ ] segment anything random positional embeddings
    - [ ] add pe to inputs and queries and check impact on mask attention pattern
- [ ] flex
    - [x] Flex transformer
    - [x] Flex local
    - [x] Flex local with wrapping
    - [x] fix flex with dynamic shapes
    - [ ] flex with nested tensors
    - [ ] flex decoder
    - [ ] flex mask attention (fully realised mask)
    - [ ] flex local CA
- [ ] better transformer
    - [x] gated dense network
    - [x] layerscale
    - [x] value residuals including learnable per token
    - [ ] input pad mask
        - [ ] otherwise pad mask
        - [ ] dispatch to flash varlen if flash
        - [ ] also support flex
    - [ ] alphafold2 attention gating
    - [ ] register tokens but interspersed for local attention
    - [ ] moe
    - [ ] CLS token (for global with context from inputs and queries)
    - [ ] [laser](https://github.com/lucidrains/x-transformers/commit/57efd7770f2f5df0ff7b4ffcbd623750b584e850#diff-b335630551682c19a781afebcf4d07bf978fb1f8ac04c6bf87428ed5106870f5R2360)
