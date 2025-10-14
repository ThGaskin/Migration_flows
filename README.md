![Flows](Images/Github/global_flows.png)

Deep learning-based estimates of global migration flows
---
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

> [!NOTE]
> We recommend using Python 3.11: this way progress bars in Jupyter notebooks will be correctly displayed.

This repository contains all code and data needed to train and evaluate a deep neural network 
used to infer annual bilateral migration flows between all countries since 1990. If you have downloaded the datasets (in particular the `T.nc` flow table) from
the [Zenodo repository](https://zenodo.org/records/15623215), you can directly evaluate the data from there. The `Evaluate.ipynb` notebook will guide you through the evaluation 
process step-by-step (see below) and allow you to recreate all the plots from the [publication](https://arxiv.org/abs/2506.22821).

> [!NOTE]
> This repository is work in progress and is still being updated. `git fetch & git pull` regularly for updates. If you encounter any problems, please [file an issue](https://github.com/ThGaskin/Migration_flows/issues/new).

> [!NOTE]
> Smaller datasets are stored in this repository in the `Estimates` folder. 
> See the [Zenodo repository](https://zenodo.org/records/15623215) for all datasets, including the full flow table.

> [!WARNING]
> The full flow table `T.nc` is quite large — around 3GB! Make sure you have enough system memory to load it. 
> If you are only interested in bilateral total flows (without the disaggregation by country of birth), the `flows.nc` file 
> is considerably smaller and easier to handle.


## Estimates
If you only want to open and analyse the data, the `Estimates` folder in this repository contains:
- A file `flows.nc`: total origin-destination flows
- A file `stocks.nc`: migrant stocks, including native-born stocks on the diagonal
- A file `net_migration.nc`: net migration estimates
- A file `T.nc` containing all flows, disaggregated by birth, is provided in the Zenodo repository.

Additionally, we provide two files for user convenience:
- `mig_unilateral.csv`, containing the following unilateral variables:
  - `imm`: total immigration flows
  - `emi`: total emigration flows
  - `net`: net migration
  - `imm_pop`: total immigrant population (i.e. not native-born)
  - `emi_pop`: total emigrant population (i.e. living abroad)
- `mig_bilateral.ncsv`, provided in the Zenodo repository, containing the following variables:
  - `mig_prev`: total origin-destination flows
  - `mig_brth`: bilateral flows by birth, i.e. the `Origin ISO` coordinate reflects the birth place
All files give both a mean estimate and a standard deviation.

## Installation and Evaluation
The `Evaluate.ipynb` Jupyter notebook contained in this folder will guide you through the model evaluation step-by-step. Before you can run
it, you will need to make sure you have installed all the required Python packages. We recommend creating a new virtual
environment and installing packages into it; you can easily install all required packages using the command
```commandline
pip install -r requirements.txt
```
If you are not executing the command from within this folder, adjust the command to point to the `requirements.txt` file
contained in this folder. Make sure you have downloaded the `T.nc` file from the Zenodo repository into the `Estimates` folder.

### Installing GPU-accelerated PyTorch
We use [`PyTorch`](https://pytorch.org) to build and train the model. Given the size of the model, we recommend using 
GPU-accelerated PyTorch for faster evaluation and training — but this is optional. Follow the [correct installation 
guide](https://pytorch.org/get-started/locally/) for your system. In the `Evaluate` notebook, you can then set the default 
device to use:

```python
device = 'cuda' # alternatively: 'cpu' or 'mps'
torch.set_default_device(device)
```
If you are using macOS on Apple Silicon, the GPU device is called `mps`. Note that printing `torch.tensors` from the GPU
to the jupyter console is sometimes not supported; move a tensor to the CPU first by running
```python
print(tensor.cpu())
```
## Training data
All training data (inputs and targets) is provided in the `Data` folder. This folder contains the input data as PyTorch `torch.Tensors` (located in `Data/Training_data`), but we also provide all the original data in seperate subfolders. The `Data/README` file gives a high-level overview of each folder's contents, and each subfolder (where necesssary) contains its own seperate README file detailing the data source and collection methodology.

## Trained neural networks
The trained networks are located in the `Trained_networks` folder. Since we have trained an ensemble of networks, each ensemble member is given in its own folder. These folders contain:
- The model weights (`model_trained.pt`)
- The optimizer state, in case you wish to continue training the model from its current state (`optim.pt`)
- The config file used to run the model (`cfg.yaml`, see below)
- The test edges masked during training and used as a test set. 
  
## Train your own model
The neural network weights are stored in the `Trained_networks` folder, alongside the configuration file used to create it.
The training code is fully configuration-based, meaning you do not need to edit any Python code to configure the training procedure.
Instead, you can adjust the settings in the `Code/cfg.yml` file, and then call
```python
python -m Code.train_model Code/cfg.yaml
```
This will load all the training data, located in `Data/Training_data`, and train a neural network. We recommend training
on a GPU. Below you will find a guide to all the settings provided in the `cfg.yaml` file; the settings shown are
the original settings used to train the network:

```yaml
# Set this to point to this folder
BASE_PATH: "."

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
  Random_sample_size: 1000
  
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
Try it out by running
```python
python -m Code.train_model Code/cfg.yaml
```
with the config settings above. You should see an output in the console that looks like this:
```commandline
Epoch     | Prediction                                                    | Loss                                                          | Test err  | Time [s]
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
          | Stock         | Net migr.     | Flow          | Outflow       | Stock         | Net migr.     | Flow          | Outflow       |           |      
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
1         |4322.1226      | 175733.8906   | 23585.3516    | 988476.9375   | 1.6634316     | 2.2389610     | 23.5002422    | 0.0000000     | 33226.7891| 6.2741
2         |4309.5229      | 175266.1406   | 22712.1484    | 927179.0000   | 1.6527443     | 2.1943829     | 22.5668011    | 0.0000000     | 31988.4902| 5.2951
3         |4299.4482      | 174890.6406   | 21881.2637    | 870300.1250   | 1.6420590     | 2.1545382     | 21.6801319    | 0.0000000     | 30809.6914| 5.2632
4         |4291.4243      | 174589.0469   | 21092.2031    | 819864.5625   | 1.6354619     | 2.1165998     | 20.8394680    | 0.0000000     | 29689.7305| 5.4498
5         |4285.0933      | 174351.3750   | 20342.2871    | 772784.6875   | 1.6300865     | 2.0847149     | 20.0419292    | 0.0000000     | 28625.2090| 5.5193
6         |4280.1152      | 174166.6719   | 19617.3242    | 728620.7500   | 1.6255611     | 2.0587058     | 19.2723675    | 0.0000000     | 27595.8398| 5.4035
7         |4276.5591      | 174021.5312   | 18919.6934    | 687027.1250   | 1.6215575     | 2.0369611     | 18.5332489    | 0.0000000     | 26604.9766| 5.5283
8         |4274.7324      | 173906.0781   | 18246.1699    | 648351.6875   | 1.6181552     | 2.0172634     | 17.8210449    | 0.0000000     | 25648.0781| 5.5880
9         |4274.0884      | 173811.5469   | 17605.2793    | 613443.9375   | 1.6154636     | 1.9999654     | 17.1445656    | 0.0000000     | 24737.0996| 5.4904
10        |4274.1289      | 173726.2500   | 16985.8691    | 579894.7500   | 1.6135281     | 1.9819719     | 16.4921188    | 0.0000000     | 23856.4414| 5.7321
```
The `Prediction` and `Test err` columns simply list L1 errors on the various target datasets — this way you can compare the model performance for different loss functions. The `Loss` columns
actually indicate the training loss – what the model is being trained on. 
