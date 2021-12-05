# NL—Propaganda
Main repo for ETHZ Natural Language Processing course project.

## Latex editable documents
- [Project proposal paper](https://www.overleaf.com/6931691827vgjtshtbyrbp) 
- [Project paper](https://www.overleaf.com/8789365945bcnsfwsyzqdk)

## Running the models
```
allennlp train configs/abcd.jsonnet --include-package src --serialization-dir models/model{x}
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

# install Rust compiler (required for transformers)
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
