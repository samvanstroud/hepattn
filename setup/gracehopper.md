# Grace Hoppper 200 Superchip Setup Guide

Setup guide for Grace Hopper 200 Superchip with a container. This is very similar to the isambard procedure, except we need to target the ARM architecture.

## Image Installation (Optional)

If a container is required, we need an ARM specific image. Download it from [here](https://github.com/prefix-dev/pixi-docker/pkgs/container/pixi/559545753?tag=jammy-cuda-13.0.0) via:

```bash
apptainer pull pixi_arm.sif docker://ghcr.io/prefix-dev/pixi@sha256:1176d181ee69035ef0ec689e6e56e226a7c9bddb348b70db3ffc755aa9fe8940
```

From here, you can shell into the ARM apptainer image as usual.

## Environment Setup

Install the environment that has the necessary requirements for FA3 with Grace Hopper:

```bash
pixi install -e grace
```

## FA3 Install

Activate the environment

```bash
pixi shell -e grace
```

From here, the FA3 installation proceeds in the same was as usual. See `setup/isambard.md`. 