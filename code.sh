# From kan_mamote_project/
mkdir src
mkdir src/models
mkdir src/models/experts
mkdir src/layers
mkdir src/losses
mkdir src/data
mkdir src/utils
mkdir scripts
mkdir tests

touch src/__init__.py # Makes src a Python package
touch src/models/__init__.py
touch src/models/kan_mamote.py
touch src/models/k_mote.py
touch src/models/c_mamba.py
touch src/models/experts/__init__.py
touch src/models/experts/fourier_kan.py
touch src/models/experts/spline_kan.py
touch src/models/experts/rkhs_kan.py
touch src/models/experts/wavelet_kan.py
touch src/layers/__init__.py
touch src/layers/kan_base_layer.py
touch src/layers/basis_functions.py
touch src/losses/__init__.py
touch src/losses/regularization_losses.py
touch src/data/__init__.py
touch src/data/dataset.py
touch src/data/dataloader.py
touch src/utils/__init__.py
touch src/utils/config.py
touch src/utils/functional.py
touch src/utils/plotting.py
touch scripts/train.py
touch scripts/evaluate.py
touch scripts/visualize.py
touch tests/__init__.py
touch README.md