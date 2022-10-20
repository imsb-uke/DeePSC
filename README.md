# DeePSC

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://github.com/Project-MONAI/MONAI"><img alt="MONAI" src="https://img.shields.io/badge/-MONAI-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

This repository contains an exemplary implementation of the DeePSC model proposed in "DeePSC - deep learning-based decision support for the diagnosis of primary sclerosing cholangitis on 2D magnetic resonance cholangiopancreatography".

<div align="center">

![](deepsc_architecture.png)

</div>

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/FabianWesth/DeePSC
cd DeePSC

# create conda environment
conda create -n deepsc python=3.6 -y
conda activate deepsc

# OR

# create virtual environment
python3.6 -m venv .venv
source .venv/bin/activate

# install python requirements
pip install -r requirements.txt
```

Train DeePSC ensemble model 

```bash
# train 
python deepsc.py
```
