# Isambard Cluster Setup Guide

- [Login & Setup Documentation](https://docs.isambard.ac.uk/user-documentation/getting_started/)
- [VS Code Usage Guide](https://docs.isambard.ac.uk/user-documentation/guides/vscode/)

Isambard AI uses SLURM for job scheduling.


## Environment Setup

The `pyproject.toml` is already adapted for Isambard compatibility.

**Note:** On Isambard, you do not need the Apptainer image, as the system’s `libc` version supports recent PyTorch releases.

Install pixi:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Get the code:

```bash
git clone git@github.com:samvanstroud/hepattn.git
cd hepattn
git checkout isambard
```

Install dependencies using `pixi`:

```bash
pixi install -e isambard
```

For some reason, when (re)activing the environment, you may see an error.
To fix it just `rm` the offending dirctory:

```bash
rm -r /home/u5ar/svanstroud.u5ar/.cache/rattler/cache/uv-cache/archive-v0/
```


## FlashAttention3 (FA3) Beta

- [FlashAttention 3 Beta Release](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#flashattention-3-beta-release)
- **Note:** Building FA3 can take up to 1 hour and must be done on a compute node.

```bash
pixi shell -e isambard
cd ..
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper/
MAX_JOBS=16 python setup.py install
```

If the build runs for a while, but ends in an error related to `ninja`, this can be ignored as FA3 is usually installed successfully by this point.
Note that you have to import FA3 from a different namespace: `flash_attn_interface` rather than `flash_attn` (see [here](https://github.com/samvanstroud/hepattn/blob/main/src/hepattn/models/attention.py#L4)).


## Data Access & Transfer

Project data is available to all project members. Input data should be stored at:

```
/projects/u5ar/data/
```

- [System Storage Info](https://docs.isambard.ac.uk/user-documentation/information/system-storage/)
