# self-prior-mirror

***Active Inference with a Self-Prior in the Mirror-Mark Task***

Dongmin Kim, Hoshinori Kanazawa, and Yasuo Kuniyoshi

arXiv preprint URL: https://arxiv.org/abs/2604.09673

---

This repository provides the open source implementation of the Self-Prior agent in the paper "Active Inference with a Self-Prior in the Mirror-Mark Task".

## Scope

- This repository includes only the `robot_mirror` environment configuration.
- (Baby simulation environments and related assets are intentionally removed)

## Requirements

- `uv`

## Installation

```bash
uv sync
```

## Training

```bash
export MUJOCO_GL=egl
MUJOCO_EGL_DEVICE_ID=0 uv run 01train.py --seed 0 --device 0 --config robot_mirror rep5easy eval_period500 nodecay usecompile
```

## Visualization

```bash
export MUJOCO_GL=egl
uv run tensorboard --logdir log --bind_all --samples_per_plugin "images=10000"
```

## Evaluation (Making data)

```bash
export MUJOCO_GL=egl
uv run 02eval.py --seed 0 --device 0 --checkpoint_episode 5000 "log/robot_mirror/rep5easy/eval_period500/nodecay/usecompile/0-0"
```

## Video samples
### Robot

<img src="./img/robot-seed20.gif" width="128">

### Baby

<img src="./img/baby-seed2.gif" width="128">
<img src="./img/baby-seed5.gif" width="128">


## Acknowledgement
This work was conducted at the [Intelligent Systems and Informatics Laboratory (ISI Lab)][lab], The University of Tokyo, Japan.

This paper was supported by JST PRESTO (Grant Number JPMJPR23S4), JSPS KAKENHI (Grant Numbers 25H00448), and the Next Generation Artificial Intelligence Research Center (AI Center), The University of Tokyo.

[lab]: https://www.isi.imi.i.u-tokyo.ac.jp