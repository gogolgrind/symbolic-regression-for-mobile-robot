
## Description

Symbolic regreession for optimization and control of mobile robot

## Installation

### Setup Symbolic Regression Tools (on colab)
- git clone https://github.com/facebookresearch/symbolicregression symbolic
- mv ./symbolic/* ./
- rm -rf symbolic
- !conda env create --name symbolic regression --file=environment.yml
- !conda init
- !activate symbolic
- !pip install git+https://github.com/pakamienny/sympytorch
- !conda install -c conda-forge julia
- !conda install -c conda-forge pysr
- python3 -m pysr install

### ACADOS

- SEE offfical docs on docs.acados.org/installation, it works even in colab


## Authors

- [Konstantin Sozykin](https://github.com/gogolgrind)
- [Maxim Buza](https://github.com/MReborn)

## Misc

- Data : https://docs.google.com/spreadsheets/d/1Mhgrcfclefd594uNYzIuugOy4exVOt4fNIpBrMMWQNs/edit?usp=sharing
- Slides : https://docs.google.com/presentation/d/1_6ILMUYMq2KbW0xux1jBfj_DXUpfBjTbJms1Ftg6lQE/edit?usp=sharing
