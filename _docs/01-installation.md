---
title: "Installation"
permalink: /docs/installation/
excerpt: "About installation."
last_modified_at: 2022-08-04T08:48:05-04:00
redirect_from:
  - /theme-setup/
toc: true
layout: tuto
---


First of all, users need to clone the source code and install the required packages (we suggest python version >= 3.9). You can choose between the following two installation methods (via docker or conda) to install FederatedScope.

```bash
git clone https://github.com/alibaba/FederatedScope.git
cd FederatedScope
```
### Use Docker

You can build docker image and run with docker env (cuda 11 and torch 1.10):
```
docker build -f environment/docker_files/federatedscope-torch1.10.Dockerfile -t alibaba/federatedscope:base-env-torch1.10 .
docker run --gpus device=all --rm -it --name "fedscope" -w $(pwd) alibaba/federatedscope:base-env-torch1.10 /bin/bash
```
If you need to run with down-stream tasks such as graph FL, change the requirement/docker file name into another one when executing the above commands:
```
# environment/requirements-torch1.10.txt -> 
environment/requirements-torch1.10-application.txt

# environment/docker_files/federatedscope-torch1.10.Dockerfile ->
environment/docker_files/federatedscope-torch1.10-application.Dockerfile
```
Note: You can choose to use cuda 10 and torch 1.8 via changing `torch1.10` to `torch1.8`.
The docker images are based on the nvidia-docker. Please pre-install the NVIDIA drivers and `nvidia-docker2` in the host machine. See more details [here](https://github.com/alibaba/FederatedScope/tree/master/environment/docker_files).

### Use Conda

We recommend using a new virtual environment to install FederatedScope:

```bash
conda create -n fs python=3.9
conda activate fs
```

If your backend is torch, please install torch in advance ([torch-get-started](https://pytorch.org/get-started/locally/)). For example, if your cuda version is 11.3 please execute the following command:

```bash
conda install -y pytorch=1.10.1 torchvision=0.11.2 torchaudio=0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

For users with Apple M1 chips:
```bash
conda install pytorch torchvision torchaudio -c pytorch
# Downgrade torchvision to avoid segmentation fault
python -m pip install torchvision==0.11.3
```

Finally, after the backend is installed, you can install FederatedScope from `source`:

### From source

```bash
python setup.py install

# Or (for dev mode)
pip install -e .[dev]
pre-commit install
```

Now, you have successfully installed the minimal version of FederatedScope. (**Optinal**) For application version including graph, nlp and speech, run:

```bash
bash environment/extra_dependencies_torch1.10-application.sh
```
