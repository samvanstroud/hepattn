# hepattn

We present a general end-to-end ML approach for particle physics reconstruction by adapting cutting-edge object detection techniques.
We demonstrate that a single encoder-decoder transformer can solve many different reconstruction problems that traditionally required specialised, task-specific approaches.

This general approach has been applied to various reconstruction tasks and detector setups:

- **Pixel cluster splitting** - ATLAS [[Internal][tide]]
- **Hit filtering** - TrackML [[arXiv][trackml]], ITk [WIP]
- **Tracking** - TrackML [[arXiv][trackml]], ATLAS [[Internal][tide]]
- **Primary vertexing** - *Interested in working on this? Get in touch!*
- **Secondary vertexing** - Delphes [[EPJC][vertexing]]
- **Particle flow** - CLIC [[arXiv][glow]]
- **End-to-end reconstruction** - CLD [[ML4Jets][ml4jets]]

[tide]: https://indico.cern.ch/event/1550297/contributions/6559827/
[trackml]: https://arxiv.org/abs/2411.07149
[vertexing]: https://link.springer.com/article/10.1140/epjc/s10052-024-13374-5
[glow]: https://arxiv.org/abs/2508.20092
[ml4jets]: https://indico.cern.ch/event/1526677/contributions/6530938/

## ✨ Key Features

- **🏗️ Modular architecture**: Encoder, decoder, and task modules for flexible experimentation
- **⚡ Efficient attention**: Seamlessly switch between torch SDPA, FlashAttention, and FlexAttention
- **🔬 Cutting-edge transformers**: HybridNorm, LayerScale, value residuals, register tokens, local attention
- **🚀 Performance optimised**: Full `torch.compile` and nested tensor support
- **🧪 Thoroughly tested**: Comprehensive tests across multiple reconstruction tasks
- **📦 Easy deployment**: Packaged with Pixi for reproducible environments


## 🛠️ Setup

First clone the repository:

```shell
git clone git@github.com:samvanstroud/hepattn.git
cd hepattn
```

We recommend using a container to set up and run the code.
This is necessary if your system's `libc` version is `<2.28` 
due to requirements of recent `torch` versions.
We use `pixi`'s CUDA image, which you can access with:

```shell
apptainer pull pixi.sif docker://ghcr.io/prefix-dev/pixi:0.54.1-jammy-cuda-12.8.1
apptainer shell --nv pixi.sif
```

**📝 Note**: If you are not using the `pixi` container, you will need to make sure 
`pixi` is installed according to https://pixi.sh/latest/installation/. 

You can then install the project with locked dependencies:

```shell
pixi install --locked
```

**📝 Note**: The `default` environment targets GPU machines and installs FA2.
See the [pyproject.toml](pyproject.toml) or [setup/isambard.md](setup/isambard.md)
for more information.

## 🌟 Activating the Environment

To run the installed environment, use:

```shell
pixi shell
```

You can close the environment with `exit`.
See the [`pixi shell` docs](https://pixi.sh/latest/reference/cli/pixi/shell/) for more information.

## 🧪 Running Tests

Once inside the environment, if a GPU and relevant external data are available, just run: 

```shell
pytest
```

To test parts of the code that don't require a GPU, run:

```shell
pytest -m 'not gpu'
```

To test parts of the code that don't require external input data, run:

```shell
pytest -m 'not requiresdata'
```

Please note that the current CI only tests the parts of the code that don't require a GPU or external input data:

```shell
pytest -m 'not gpu and not requiresdata'
```

## 🏃 Run Experiments

See experiment directories for instructions on how to run experiments.

- [TrackML Tracking](src/hepattn/experiments/trackml/)
- [CLIC Particle Flow](src/hepattn/experiments/clic/)

## 📖 Terminology

To ensure clarity and consistency throughout this project, we use the following definitions:

- **constituent** - input entities that go into the encoder/decoder, e.g. inner detector hits
- **object** - reconstructed outputs from the decoder, e.g. reconstructed charged particle tracks
- **input** - (also `input_object`) generic term for any input to a module (could be constituents, objects, etc)
- **output** - generic term for any output from a module (could be objects, predictions, or intermediates)

## 🤝 Contributing

If you would like to contribute, please lint and format code with

```shell
ruff check --fix .
ruff format .
```

You can also set up pre-commit hooks to automatically run these checks before committing:

```shell
pre-commit install
```

## 📄 Citing

If you use this software in your research, please cite it using the citation information available in the GitHub repository sidebar (generated from [`CITATION.cff`](CITATION.cff)).
Please also cite [our papers](#hepattn) if they are relevant to your work.
