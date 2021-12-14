# NL—Propaganda
Main repo for ETHZ Natural Language Processing course project.

## Logs models results
- [Logs models results](models/logmodels.csv)

## Latex editable documents
- [Project proposal paper](papers/proposal/nlpropaganda-proposal.pdf)
- [Project progress report paper](https://www.overleaf.com/8789365945bcnsfwsyzqdk)

## Running the models
```
allennlp train configs/abcd.jsonnet --include-package src --serialization-dir models/models_{[si, tc]}/model{x}
allennlp evaluate {model_serialized_dir} {dataset_dir} --include-package src
allennlp predict {model_serialized_dir} {dataset_dir} --include-package src --predictor {predictor_name}
```

## `allennlp` installation on M1
```
# create env (Python3.9 will not work!)
conda create --name env_name python=3.8
conda activate env_name

# install torch 
pip install torch 

# install numpy through cython (required for scipy)
python -m pip install --no-cache --no-use-pep517 pythran cython pybind11 gast"==0.4.0"

# install scipy (required for sklearn)
OPENBLAS="$(brew --prefix openblas)" pip install scipy

# install sklearn (required for allennlp)
OPENBLAS="$(brew --prefix openblas)" pip install sklearn

# install Rust compiler (required for transformers)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install setuptools_rust

# install transformers (required for allennlp)
git clone https://github.com/huggingface/tokenizers
cd tokenizers/bindings/python
python setup.py install
pip install git+https://github.com/huggingface/transformers

# install hdf5 (required for sentencepiece)
brew install hdf5
export HDF5_DIR=/opt/homebrew/Cellar/hdf5/1.12.0_4
pip install --no-build-isolation h5py

# install sentencepiece (required for allennlp)
brew install cmake
brew install pkg-config
pip install sentencepiece

# install allennlp (finally)
pip install allennlp
pip install allennlp-models
```

## ETH Euler cluster setup
```
# connect to cluster
ssh creds@euler.ethz.ch

# install miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
sh miniconda.sh
rm miniconda.sh

# disable base env.
conda config --set auto_activate_base false

# create new env.
conda create --name env_name python=3.8
conda activate env_name

# install torch
pip install torch

# install allennlp
pip install allennlp
pip install allennlp-models

# clone GitHub repo
git clone https://github.com/andreakiro/nlpropaganda
```

## Euler training with GPUs - interactive session
```
# request interactive session on a compute node
bsub -n 2 -R "rusage[mem=32000,ngpus_excl_p=1]" -Is bash

# load new software stack
env2lmod

# load modules for working with gpu computing
module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy hdf5/1.10.1

# optional - might be already installed 
pip install allennlp allennlp-models

# 
cd nlpropaganda/
allennlp train configs/si_config_cuda.jsonnet --include-package src --serialization-dir models/models_si/model[n]
```

## Running long processes off-line
You can start a long running process in cluster background and than exit it:
```
# e-g. for training allennlp models
$ nohup long-running-process &
$ exit
```
