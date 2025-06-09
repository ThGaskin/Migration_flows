![Flows](Images/Github/global_flows.png)

Deep learning-based estimates of global migration flows
---
> **Note**:
> See the [Zenodo repository](https://zenodo.org/records/15623216) for the datasets. Except for the full flow table, 
> all data is also stored in the repository in the `Estimates` folder.

> **Warning**:
> The full flow table $\boldsymbol{T}$ is quite large -- around 3GB! Make sure you have enough system memory to load it. 
> If you are only interested in bilateral total flows (without the disaggregation by country of birth), the `flows` file 
> is considerably smaller and easier to handle.

This repository contains all code and data needed to train and evaluate a deep neural network 
used to infer annual bilateral migration flows between all countries since 1990. If you have downloaded the datasets from
the Zenodo repository given above, you can directly evaluate the data from there.

## Evaluation
The `Evaluate.ipynb` Jupyter notebook contained in this folder will guide you through the model evaluation step-by-step. Before you can run
it, you will need to make sure you have installed all the required Python packages. We recommend creating a new virtual
environment; then you can easily install all packages using the command
```commandline
pip install -r requirements.txt
```
If you are not executing the command from within this folder, adjust the command to point to the `requirements.txt` file
contained in this folder.

## Installing GPU-accelerated PyTorch
We use [`PyTorch`](https://pytorch.org) to build and train the model. Given the size of the model, we recommend using 
GPU-accelerated PyTorch for faster evaluation and training --- but this is optional. Follow the [correct installation 
guide](https://pytorch.org/get-started/locally/) for your system. In the notebook, you can then set the default 
device to use:

```python
device = 'cuda'
torch.set_default_device(device)
```
If you are using macOS on Apple Silicon, the GPU device is called `mps`. Note that printing `torch.tensors` from the GPU
to the jupyter console is sometimes not supported; move a tensor to the CPU first by running
```python
print(tensor.cpu())
```

## Train your own model
The neural network weights are stored in the `Trained_networks` folder, alongside the configuration file used to create it.
The training code is fully configuration-based, meaning you do not need to edit any Python code to configure the training procedure.
Instead, you can adjust the settings in the `Code/cfg.yml` file, and then call
```python
python Code/train_model.py Code/cfg.yaml
```
This will load all the training data, located in `Data/Training_data`, and train a neural network. We recommend training
on a GPU. Below you will find a guide to all the settings provided in the `cfg.yaml` file; the settings shown are
the original settings used to train the network:

```yaml
# Set this to point to this folder
BASE_PATH: "~/Migration_flows"

# Training device to use
device: 'mps'

# Optional note that is added to the output path. Output data is stored in a time-stamped folder
# in `Results/`, alongside the configuration file used to run the model. That way, everything you do is
# stored and fully reproducible
path_note: ~ 

# Set this to true to run a model without saving any output -- useful for debugging so that 
# your Results folder doesn't get cluttered
dry_run: True 

# Settings for loading the training data
Data_loading:

  # Path to data, relative to base path
  data_path: 'Data/Training_data'

  # Passed to `torch.load`.
  load_args: {weights_only: True}

  # If you have trained a model and want to pick up where you left off, point this to the
  # directory from where you wish to load and continue training the model. 
  # Note that this will OVERWRITE the existing model, so proceed with caution.
  load_from_dir: ~

  # Rescaling constant for the target data -- we measure everything in 1000 people to
  # prevent numerical overflow. Results must afterwards then be scaled again by this value.
  data_rescale: 1e3

  # Covariates to use; this is order-specific. The 'idx' key is a list of unilateral or
  # bilateral keys to use; e.g. [i, j, k] creates three covariates, one for each country. 
  # [[i, j], [j, k]] adds two covariates for a bilateral covariate, one for i-j and one for the 
  # j-k edge
  covariates:
    - GDP_cap:
        path: 'input_covariates/GDP_cap' # GDP per capita
        idx: [i, j, k]
    - GDP_growth:
        path: 'input_covariates/GDP_growth' # GDP growth
        idx: [i, j, k]
    - Trade:
        path: 'input_covariates/Trade'
        idx: [[j, k], [k, j]]
    - Population:
        path: 'input_covariates/Population'
        idx: [i, j, k]
    - Life_expectancy:
        path: 'input_covariates/Life_expectancy'
        idx: [i, j, k]
    - Birth_rate:
        path: 'input_covariates/Birth_rate'
        idx: [j, k]
    - Death_rate:
        path: 'input_covariates/Death_rate'
        idx: [j, k]
    - Distance:
        path: 'input_covariates/Distance'
        idx: [[j, k]]
    - Linguistic_similarity:
        path: 'input_covariates/Linguistic_similarity'
        idx: [[i, k], [j, k]]
    - Religious_similarity:
        path: 'input_covariates/Religious_similarity'
        idx: [[i, k], [j, k]]
    - Conflict_deaths:
        path: 'input_covariates/Conflict_deaths'
        idx: [j, k]
    - Refugees:
        path: 'input_covariates/Refugees'
        idx: [[i, j], [i, k]]
    - Refugees_diff:
        path: 'input_covariates/Refugees_diff'
        idx: [[i, j], [i, k]]
    - Colonial_ties:
        path: 'input_covariates/Colonial_ties'
        idx: [[i, k], [j, k]]
    - EU:
        path: 'input_covariates/EU'
        idx: [i, j, k]

# Neural network settings. Adjust these to determine the neural network architecture
# Ensure your system is capable enough to run larger networks --- the settings below are the 
# original training settings.
NeuralNet:
  num_layers: 7
  nodes_per_layer:
    default: 60
  activation_funcs:
    default: tanh
    layer_specific:
      -1:
        name: celu
        args: [-12]
  biases:
    default: [-1, 1]
  learning_rate: 0.002
  optimizer: Adam
  latent_space_dim: 100

# Training settings
Training:

  # Number of epochs
  N_epochs: 10
  
  # Due to memory constraints, we cannot optimise all 900,000 edges at the same time.
  # This setting draws a random sample of edge indices, which are optimised. A smaller value
  # is more memory efficient but means the model will take longer to converge.
  # Use the maximum size that will fit your GPU -- around 50,000 for a good GPU, depending also
  # on the size of the neural network and latent space dimension.
  Random_sample_size: 50000
  
  # Perform a gradient descent step after every batch. A batch is a single five-year interval, 
  # corresponding to one stock data interval. For the period from 1990--2023, there are seven batches.
  # We recommend using the full training period as a batch (batch gradient descent)
  Batch_size: 7
  
  # Store the neural network after this many steps
  write_every: 100

  # If you want to mask a certain fraction of flow corridors for testing, increase this
  # to a number in [0, 1]. The mask is randomly generated but stored alongside the neural network, 
  # meaning that you can reproduce the test data later on. If you interrupt training and then 
  # pick up where you left off using the load_from_dir argument, the stored flow mask will be 
  # loaded, so that the test and training data is always the same for each model.
  flow_test_frac: 0.2

  # Gradient norm clipping. Set to False to turn off, or pass a gradient norm to clip to.
  clip_grad_norm: 1.0

  # Rescaling lambdas for the Yeo-Johnson transforms of the target data.
  Rescaling:
    stock:
      lmbda: 0.5
    net_migration:
      lmbda: 0.5
    flow:
      lmbda: 0.5

  # Confidence bands around the target data within which we do not penalise. This can be
  # useful to prevent strong overfitting.
  Confidence_band:
    stock: 0.01
    net_migration: 0.01
    flow: 0.01

  # Balancing of the different terms in the loss function.
  weight_factors:
    stock: 1
    flow: 1
    net_migration: 1
    # An additional regularisation term to ensure outflows do not exceed the total population -- not necessary but can be turned on
    # if required.
    regulariser: 0 

```