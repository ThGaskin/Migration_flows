"""
    Migration Flow Model with Recurrent Neural Network

    This module implements a recurrent neural network model for estimating international
    migration flows. The model integrates stock data, net migration, and known flow data
    to predict migration patterns while maintaining physical consistency constraints.

    Key features:
    - Recursive stock-flow modeling with temporal dependencies
    - Custom RNN with latent state for capturing migration dynamics
    - Physical constraints (population bounds, stock conservation)
    - Multi-objective optimization with configurable loss weights

    The model processes country-pair data over time, using neural networks to estimate
    flows that reconcile observed stock differences and net migration patterns.
"""

# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import datetime
import numpy as np
import os
import pickle
import sys
import time
import torch

from ruamel.yaml import YAML

from Code import NeuralNet
from Code import yeo_johnson_transform, build_input

yaml = YAML(typ="safe")

# Load the configuration, with the option of passing a config path
if len(sys.argv) > 1:
    with open(sys.argv[1], "r") as file:
        cfg = yaml.load(file)
else:
    from pathlib import Path
    config_path = Path(__file__).parent / "cfg.yaml"
    with open(config_path, "r") as file:
        cfg = yaml.load(file)

# Set default device
device = cfg["device"]

# Load the base paths
BASE_PATH = cfg["BASE_PATH"]

# Set seed for reproducibility, if given
if cfg.get('seed', None) is not None:
    torch.manual_seed(cfg['seed'])

# Load or create the save path, if not running in dry run setting (no data saving)
dry_run = cfg.get("dry_run", False)
if not dry_run:
    save_to_path = cfg["Data_loading"].get("load_from_dir", None)
    if save_to_path is None:
        _date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        note = cfg.get("path_note", None)
        if note:
            _date_time += f"_{note}"
        save_to_path = os.path.expanduser(
            os.path.join(BASE_PATH, f"{cfg.get('OUT_DIR', 'Results')}/{_date_time}")
        )
        os.makedirs(save_to_path)

        # Write the cfg
        with open(f"{save_to_path}/cfg.yaml", "w") as file:
            yaml.dump(cfg, file)
    else:
        save_to_path = os.path.expanduser(save_to_path)

# Update neural network config and covariates paths from any pre-trained configurations
if cfg["Data_loading"].get("load_from_dir", None) is not None:
    with open(f"{cfg['Data_loading']['load_from_dir']}/cfg.yaml", "r") as file:
        nn_cfg = yaml.load(file)
    cfg['NeuralNet'] = nn_cfg['NeuralNet']
    cfg['Data_loading']['covariates'] = nn_cfg['Data_loading']['covariates']

# ----------------------------------------------------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------------------------------------------------
# Path to training data, relative to base path
data_path = cfg["Data_loading"]["data_path"]
target_data_cfg = cfg["Data_loading"].get('targets', {})

# Load args, passed to ``torch.load``
load_args = cfg["Data_loading"].get("load_args", {})

# Net migration data and mask of entries excluded from the optimisation; weights quantifying the uncertainty on each
# entry (currently all 1)
print('Loading data ... ')
NetMigration = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/{target_data_cfg.get('net_migration', 'net_migration')}.pt"), **load_args
    ).to(device).float()
)
NetMigrationMask = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/{target_data_cfg.get('net_migration_mask', 'net_migration_mask')}.pt"),
        **load_args,
    ).to(device).bool()
)
NetMigrationWeights = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/{target_data_cfg.get('net_migration_weights', 'net_migration_weights')}.pt"), **load_args
    ).to(device).float()
)

# Stock, stock differences, and mask of entries excluded from the optimisation;
# weights quantifying the uncertainty on each entry
Stock = (
    torch.load(os.path.expanduser(f"{BASE_PATH}/{data_path}/{target_data_cfg.get('stock', 'stock')}.pt"), **load_args).to(device).float()
)
StockDifferences = (
    torch.load(os.path.expanduser(f"{BASE_PATH}/{data_path}/{target_data_cfg.get('stock_diff', 'stock_diff')}.pt"), **load_args).to(device).float()
)
StockDifferenceMask = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/{target_data_cfg.get('stock_diff_mask', 'stock_diff_mask')}.pt"), **load_args
    ).to(device).bool()
)
StockDifferenceWeights = (
    torch.load(os.path.expanduser(f"{BASE_PATH}/{data_path}/{target_data_cfg.get('stock_diff_weights', 'stock_diff_weights')}.pt"), **load_args).to(device).float()
)
# Array of the year indices to which the stock data correspond (typically every 5 years)
StockRange = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/{target_data_cfg.get('stock_range', 'stock_range')}.pt"), **load_args
    ).to(device).int().tolist()
)

# Total population of each country, acting as an upper bound for outflows, as well the total number of births
# and death rates for each country over time
Population = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/{target_data_cfg.get('total_population', 'total_population')}.pt"), **load_args
    ).to(device).float()
)
Births = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/{target_data_cfg.get('total_births', 'total_births')}.pt"), **load_args
    ).to(device).float()
)
# Reshape into a diagonal matrix for each year
Births = torch.stack([torch.diag(Births[i]) for i in range(len(Births))])

# Death rates for each destination country and year
DeathRates = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/{target_data_cfg.get('death_rate', 'death_rate')}.pt"), **load_args
    ).to(device).float()
).reshape(-1, 1, Births.shape[-1])

# Known flow data and mask; weights quantifying uncertainty on each entry (currently only not 1 for Quantmig
# entries)
Flow = (
    torch.load(os.path.expanduser(f"{BASE_PATH}/{data_path}/{target_data_cfg.get('flow', 'flow')}.pt"), **load_args).to(device).float()
)
FlowMask = (
    torch.load(os.path.expanduser(f"{BASE_PATH}/{data_path}/{target_data_cfg.get('flow_mask', 'flow_mask')}.pt"), **load_args).to(device).bool()
)
FlowWeights = (
    torch.load(os.path.expanduser(f"{BASE_PATH}/{data_path}/{target_data_cfg.get('flow_weights', 'flow_weights')}.pt"), **load_args).to(device).float()
)

# Network edges to train on
EdgeIndices = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/{target_data_cfg.get('edge_indices', 'edge_indices')}.pt"), **load_args
    ).to(device).long()
)

print('Constructing training data ... ')
# Build the input data to the neural network (covariates for each year and edge)
TrainingData = build_input(
    cfg, EdgeIndices, Y = NetMigration.shape[0], device=device
)

# Transformation parameters for the stock, required to adjust the input to the neural network
with open(os.path.expanduser(f"{BASE_PATH}/{data_path}/transformation_parameters.pickle"), "rb") as file:
    StockTransfParams = pickle.load(file)['Stock']

# Standard deviation of the total population and the stock, used to scale the additional errors
inv_pop_std = 1 / Population.flatten().std()
# inv_stock_std = 1 / torch.masked_select(Stock, StockMask).std()

# Number of countries
N = NetMigration.shape[1]

# ----------------------------------------------------------------------------------------------------------------------
# Transform the training data
# ----------------------------------------------------------------------------------------------------------------------
# Bundle the years into batches
BatchYears = [
    torch.arange(
        StockRange[i],
        StockRange[min(i + 1, len(StockRange) - 1)],
        dtype=int,
    ).tolist()
    for i in range(0, len(StockRange) - 1)
]

# The stock year indices are just the indices of the individual batches
StockYears = list(range(0, len(StockDifferences)))

# Move the transformation parameters to the device
for key in list(cfg["Training"]["Rescaling"].keys()):
    cfg["Training"]["Rescaling"][key]["lmbda"] = torch.tensor(
        cfg["Training"]["Rescaling"][key]["lmbda"]
    ).to(device)

def transform_data(_data_dict) -> dict:
    """
    Transform dataset using Yeo-Johnson transformation and standardize.

    Applies Yeo-Johnson power transformation to normalize the data distribution,
    then centers and scales the transformed data to zero mean and unit variance.
    This is particularly useful for handling skewed distributions and improving
    neural network training stability.

    The transformation is applied batch-wise, with separate normalization
    parameters computed for each batch to handle potential distribution shifts
    over time.

    Parameters
    ----------
    _data_dict : dict
        Dictionary containing the data and transformation parameters with keys:
        - 'data': torch.Tensor
            Raw data to transform, typically of shape [batch_size, ...]
        - 'mask': torch.Tensor (bool)
            Mask indicating which elements to include in transformation
        - 'batch_indices': list
            Indices specifying how to group data into batches
        - 'transformation_parameters': dict
            Parameters for Yeo-Johnson transformation, including 'lmbda'

    Returns
    -------
    dict
        Enhanced dictionary with transformation results, containing all original
        keys plus:
        - 'transformed_data': list of torch.Tensor
            Transformed, centered, and scaled data for each batch
        - 'mean': list of torch.Tensor
            Mean of transformed (but not centered) data for each batch
        - 'std': list of torch.Tensor
            Standard deviation of transformed data for each batch

    Notes
    -----
    The transformation pipeline is:
    1. Apply Yeo-Johnson transform to masked data
    2. Compute mean and std of transformed data per batch
    3. Center and scale: (transformed - mean) / std

    This ensures the final transformed data has zero mean and unit variance
    while preserving the distribution-shaping benefits of Yeo-Johnson.
    """
    _data_transformed = [
        yeo_johnson_transform(
            torch.masked_select(_data_dict["data"][b], _data_dict["mask"][b]),
            **_data_dict["transformation_parameters"],
        )
        for b in _data_dict["batch_indices"]
    ]
    _data_dict["mean"] = [_d.mean() for _d in _data_transformed]
    _data_dict["std"] = [_d.std() for _d in _data_transformed]
    _data_dict["transformed_data"] = [
        (_data_transformed[i] - _data_dict["mean"][i]) / _data_dict["std"][i]
        for i in range(len(_data_transformed))
    ]
    return _data_dict

# Mask of test values used to track the testing error. These are randomly generated or re-loaded from previous runs
# (mask must be consistent for a single train)
if cfg["Data_loading"].get("load_from_dir", None) is not None:
    FlowTestMask = (
        torch.load(f"{cfg['Data_loading']['load_from_dir']}/FlowTestMask.pt", **load_args).to(device).bool()
    )
    FlowMask = torch.logical_and(FlowMask, ~FlowTestMask)
else:
    # Origin-Destination pairs with at least one non-NaN flow value
    od_indices = FlowMask.any(dim=0).nonzero(as_tuple=False)

    # Randomly sample a subset of them
    n_samples = int(cfg['Training'].get('flow_test_frac', 0) * len(od_indices))
    perm = torch.randperm(len(od_indices), device=device)
    test_idx = od_indices[perm[:n_samples]]
    train_idx = od_indices[perm[n_samples:]]  # remaining pairs

    # Make a test and train mask and populate
    test_mask = torch.zeros((N, N), dtype=torch.bool, device=device)
    train_mask = torch.zeros((N, N), dtype=torch.bool, device=device)
    test_mask[test_idx[:, 0], test_idx[:, 1]] = True
    train_mask[train_idx[:, 0], train_idx[:, 1]] = True

    # Combine the masks
    FlowTestMask = torch.logical_and(FlowMask, test_mask.unsqueeze(0))
    FlowMask = torch.logical_and(FlowMask, train_mask.unsqueeze(0))

    # Save for future training (to ensure we always use the same test and train sets)
    if not dry_run:
        torch.save(FlowTestMask.cpu(), f"{save_to_path}/FlowTestMask.pt")

    del perm, test_idx, train_idx, test_mask, train_mask, od_indices

print('Transforming the training data ... ')
# Transform the targets and store them in a dictionary together with the weights, masks, and batch indices
TrainingDataDict = dict(
    net_migration=transform_data(
        dict(
            data=NetMigration,
            mask=NetMigrationMask,
            weights=NetMigrationWeights,
            batch_indices=BatchYears,
            transformation_parameters=cfg["Training"]["Rescaling"]["net_migration"]
        )
    ),
    stock=transform_data(
        dict(
            data=StockDifferences,
            mask=StockDifferenceMask,
            weights=StockDifferenceWeights,
            batch_indices=StockYears,
            transformation_parameters=cfg["Training"]["Rescaling"]["stock"]
        )
    ),
    flow=transform_data(
        dict(
            data=Flow,
            mask=FlowMask,
            weights=FlowWeights,
            batch_indices=BatchYears,
            transformation_parameters=cfg["Training"]["Rescaling"]["flow"]
        )
    ),
)

# ----------------------------------------------------------------------------------------------------------------------
# Set up neural network
# ----------------------------------------------------------------------------------------------------------------------
# Scaling factor for the Neural Net output
Scale = torch.tensor(cfg["Data_loading"].get("data_rescale", 1.0))

print('Initialising neural network ... ')
NN = NeuralNet(
    input_size=TrainingData.shape[2] + 2 + cfg['NeuralNet'].get('latent_space_dim', 0),
    output_size=1 + cfg['NeuralNet'].get('latent_space_dim', 0), **cfg["NeuralNet"]
).to(device)

# If using a pretrained model, load model and loss time series
if cfg["Data_loading"].get("load_from_dir", None) is not None:
    NN.load_state_dict(
        torch.load(
            f"{cfg['Data_loading']['load_from_dir']}/model_trained.pt",
            map_location=torch.device(device),
            **load_args,
        )
    )
    NN.eval()
    NN.optimizer.load_state_dict(
        torch.load(
            f"{cfg['Data_loading']['load_from_dir']}/optim.pt",
            map_location=torch.device(device),
            **load_args,
        )
    )
    with open(f"{cfg['Data_loading']['load_from_dir']}/loss_dict.pickle", "rb") as file:
        LossDict = pickle.load(file)
else:
    LossDict = dict(
        (k, {"stock": [], "net_migration": [], "flow": [], "outflow": []})
        for k in ["prediction", "loss"]
    )
    LossDict['epoch'] = []
    LossDict['test'] = {"flow": []}

# ----------------------------------------------------------------------------------------------------------------------
# Training settings
# ----------------------------------------------------------------------------------------------------------------------
N_EPOCHS = cfg["Training"]["N_epochs"]
BATCH_SIZE = cfg["Training"]["Batch_size"]
RANDOM_SAMPLE_SIZE = cfg["Training"]["Random_sample_size"]
WRITE_EVERY = cfg["Training"].get("write_every", N_EPOCHS - 1)
for key in cfg['Training']['weight_factors'].keys():
    cfg['Training']['weight_factors'][key] = torch.tensor(cfg['Training']['weight_factors'][key], device=device)

# ----------------------------------------------------------------------------------------------------------------------
# Single training batch
# ----------------------------------------------------------------------------------------------------------------------
def batch(batch_idx, batch_stock_init, epoch_loss_dict, h_t=None):

    """
    Process a single time batch through the recurrent migration model.

    A batch represents a contiguous time period between stock measurement points.
    This function processes all years within the batch, updating stocks recursively
    and accumulating flows while tracking the hidden state.

    The function handles:
    - Random sampling of edges for gradient computation (detached/undetached)
    - Recursive stock prediction using birth, death, and migration flows
    - Hidden state propagation through time
    - Loss computation for stock differences, net migration, and flows

    Parameters
    ----------
    batch_idx : int
        Index of the current batch in BatchYears list
    batch_stock_init : torch.Tensor
        Initial stock matrix [N, N] for the first year of the batch
    epoch_loss_dict : dict
        Dictionary to accumulate loss statistics across the batch.
        Structure: {'prediction': {metric: list}, 'loss': {metric: list}}
    h_t : torch.Tensor, optional
        Initial hidden state [num_edges, latent_dim] for the batch.
        If None, no latent state is used.

    Returns
    -------
    tuple
        batch_loss : torch.Tensor
            Total loss for the batch, suitable for backpropagation
        stock_prediction : torch.Tensor
            Final stock matrix [N, N] at the end of the batch
        flow_predictions : torch.Tensor
            Stacked flow predictions for all years in the batch
        epoch_loss_dict : dict
            Updated loss dictionary with batch statistics appended
        h_t : torch.Tensor
            Final hidden state at the end of the batch
    """

    # Store the flow predictions to allow for testing after each epoch
    batch_flow_predictions = []

    # Store the net migration predictions to calculate the batch loss
    batch_net_migration_predictions = []

    # Track the current stock prediction, used as input to the neural network
    stock_prediction = batch_stock_init.clone()
    stock_input = yeo_johnson_transform(stock_prediction, **StockTransfParams) # Input to NN

    # Gather the batch loss
    batch_loss = torch.tensor(0.0, requires_grad=True)

    # Run the model forward in time and optimise the neural network parameters
    for t in BatchYears[batch_idx]:

        # Randomly select RANDOM_SAMPLE_SIZE many indices from the edge list
        shuffled_edge_indices = torch.randperm(EdgeIndices.shape[1])
        undetached = shuffled_edge_indices[
            : min(EdgeIndices.shape[1] - 1, RANDOM_SAMPLE_SIZE)
        ]
        detached = shuffled_edge_indices[
            min(EdgeIndices.shape[1] - 1, RANDOM_SAMPLE_SIZE):
        ]

        # Initialize aggregation tensors
        net_stock_flow = torch.zeros(N, N, device=device)  # inflow - outflow for stock[i,k]
        flow = torch.zeros(N, N, device=device)  # total flow[j,k]

        # Prediction in the case of a latent dimension
        if h_t is not None:
            # Make a prediction on the undetached edges
            idx_i, idx_j, idx_k = EdgeIndices[:, undetached]
            res = NN(torch.cat([TrainingData[t][undetached],
                                stock_input[idx_i, idx_j].unsqueeze(1),
                                stock_input[idx_i, idx_k].unsqueeze(1),
                                h_t[undetached]],
                               dim=1))
            T_values = Scale * torch.exp(res[:, 0])
            h_t[undetached, :] = res[:, 1:].tanh()

            # Accumulate into aggregation tensors
            net_stock_flow.index_put_((idx_i, idx_k), T_values, accumulate=True)  # inflow
            net_stock_flow.index_put_((idx_i, idx_j), -T_values, accumulate=True)  # outflow
            flow.index_put_((idx_j, idx_k), T_values, accumulate=True)

            # Make a prediction on the detached edges
            idx_i, idx_j, idx_k = EdgeIndices[:, detached]
            with torch.no_grad():
                res = NN(torch.cat([TrainingData[t][detached],
                                    stock_input[idx_i, idx_j].unsqueeze(1),
                                    stock_input[idx_i, idx_k].unsqueeze(1),
                                    h_t[detached]],
                                   dim=1))
                T_values = Scale * torch.exp(res[:, 0])
                h_t[detached, :] = res[:, 1:].tanh()

                # Accumulate into aggregation tensors
                net_stock_flow.index_put_((idx_i, idx_k), T_values, accumulate=True)
                net_stock_flow.index_put_((idx_i, idx_j), -T_values, accumulate=True)
                flow.index_put_((idx_j, idx_k), T_values, accumulate=True)
        else:
            # Make a prediction on the undetached edges
            idx_i, idx_j, idx_k = EdgeIndices[:, undetached]
            T_values = Scale * torch.exp(NN(
                torch.cat([TrainingData[t][undetached],
                           stock_input[idx_i, idx_j].unsqueeze(1),
                           stock_input[idx_i, idx_k].unsqueeze(1)],
                          dim=1))).flatten()

            # Accumulate into aggregation tensors
            net_stock_flow.index_put_((idx_i, idx_k), T_values, accumulate=True)
            net_stock_flow.index_put_((idx_i, idx_j), -T_values, accumulate=True)
            flow.index_put_((idx_j, idx_k), T_values, accumulate=True)

            # Make a prediction on the detached edges
            idx_i, idx_j, idx_k = EdgeIndices[:, detached]
            with torch.no_grad():
                T_values = Scale * torch.exp(NN(
                    torch.cat([TrainingData[t][detached],
                               stock_input[idx_i, idx_j].unsqueeze(1),
                               stock_input[idx_i, idx_k].unsqueeze(1)],
                              dim=1))).flatten()

                # Accumulate into aggregation tensors
                net_stock_flow.index_put_((idx_i, idx_k), T_values, accumulate=True)
                net_stock_flow.index_put_((idx_i, idx_j), -T_values, accumulate=True)
                flow.index_put_((idx_j, idx_k), T_values, accumulate=True)

        # Predict the stock of next year
        stock_prediction = (
                Births[t]
                + (1 - DeathRates[t]) * stock_prediction
                + net_stock_flow
        )

        batch_flow_predictions.append(flow)

        # Predict the net migration
        batch_net_migration_predictions.append(flow.sum(dim=0) - flow.sum(dim=1))

        # The total outflow also cannot exceed the total population. This can be used as a regulariser, but is
        # usually not necessary
        outflow_error_population = torch.relu(
            flow.sum(dim=1) - Population[t]
        ).mean()

        # Add to loss. The components are each scaled with the std of the stock and population to balance
        # their contribution to the loss with the stock and net migration errors. Without this balancing they will
        # dominate the loss and decrease training performance
        total_additional_err = cfg["Training"]["weight_factors"]["regulariser"] * (
                inv_pop_std * outflow_error_population
        )
        epoch_loss_dict["prediction"]["outflow"].append(
            outflow_error_population.detach()
        )
        epoch_loss_dict["loss"]["outflow"].append(total_additional_err.clone().detach())
        if cfg["Training"]["weight_factors"].get("regulariser", 0) > 0:
            batch_loss = batch_loss + total_additional_err

        # Prepare stock covariate for next year
        # Ensure stocks are positive
        stock_input = torch.maximum(stock_prediction.clone().detach(), torch.tensor(0.0))

        # Estimate the native-born population
        stock_input.fill_diagonal_(0)
        stock_input[range(stock_input.shape[0]), range(stock_input.shape[1])] = torch.maximum(
            Population[t] - stock_input.sum(dim=0), torch.tensor(0.0, device=device)
        )
        # Transform
        stock_input = yeo_johnson_transform(stock_input, **StockTransfParams)

    # Calculate the error on the transformed predictions
    predictions = dict(
        net_migration=torch.stack(batch_net_migration_predictions),
        stock=stock_prediction - batch_stock_init, # Difference in stocks
        flow=torch.stack(batch_flow_predictions),
    )

    for key in TrainingDataDict.keys():

        # If no values are present, continue
        if (~(TrainingDataDict[key]["mask"][TrainingDataDict[key]["batch_indices"][batch_idx]])).all():
            continue

        # Transform the prediction. Centralise using the mean and standard deviation from
        # the transformed target data
        predictions_transformed = (
            yeo_johnson_transform(
                torch.masked_select(
                    predictions[key],
                    TrainingDataDict[key]["mask"][TrainingDataDict[key]["batch_indices"][batch_idx]],
                ),
                **TrainingDataDict[key]["transformation_parameters"],
            )
            - TrainingDataDict[key]["mean"][batch_idx]
        ) / TrainingDataDict[key]["std"][batch_idx]

        # Get the weights
        weights = cfg['Training']['weight_factors'][key] * torch.masked_select(
            TrainingDataDict[key]["weights"][TrainingDataDict[key]["batch_indices"][batch_idx]],
            TrainingDataDict[key]["mask"][TrainingDataDict[key]["batch_indices"][batch_idx]]
        )

        # Do not penalise within a small band around the values to prevent overfitting.
        # This is because the initial value is uncertain, and its error will propagate
        # forwards onto all the other predictions. We use a band of around 5% (~0.01 when transformed)
        pred_loss = (weights * torch.relu(
            (predictions_transformed - TrainingDataDict[key]["transformed_data"][batch_idx])**2
            - cfg['Training']['Confidence_band'].get(key, 0) * TrainingDataDict[key]["transformed_data"][batch_idx]**2
        )).mean()

        # Add to loss
        batch_loss = batch_loss + pred_loss

        # Store the training loss and prediction error
        epoch_loss_dict["loss"][key].append(pred_loss.clone().detach())
        epoch_loss_dict["prediction"][key] += [
            *(
                abs(
                    predictions[key] -
                    torch.masked_fill(TrainingDataDict[key]["data"][
                              TrainingDataDict[key]["batch_indices"][batch_idx]],
                          ~TrainingDataDict[key]["mask"][
                              TrainingDataDict[key]["batch_indices"][batch_idx]],
                          torch.nan)
                ).detach()
            )
        ]

    return batch_loss, stock_prediction, predictions['flow'].detach(), epoch_loss_dict, h_t

# ----------------------------------------------------------------------------------------------------------------------
# Single training epoch
# ----------------------------------------------------------------------------------------------------------------------
def epoch(epoch_init_stock) -> dict:

    """
    Execute a complete training epoch over all time batches.

    An epoch processes the entire temporal sequence, divided into batches by
    stock measurement intervals. Each batch is processed sequentially, with
    gradient accumulation and periodic optimization steps.

    The epoch:
    - Processes all batches in temporal order
    - Maintains hidden state continuity across batches
    - Accumulates gradients and performs optimization every BATCH_SIZE batches
    - Tracks prediction errors and test performance

    Parameters
    ----------
    epoch_init_stock : torch.Tensor
        Initial stock matrix [N, N] for the first year of the training period

    Returns
    -------
    dict
        Dictionary containing epoch-averaged metrics with structure:
        {
            'prediction': {
                'stock': average stock prediction error,
                'net_migration': average net migration error,
                'flow': average flow error,
                'outflow': average outflow constraint violation
            },
            'loss': {
                'stock': average stock loss,
                'net_migration': average net migration loss,
                'flow': average flow loss,
                'outflow': average outflow constraint loss
            },
            'test': {
                'flow': average test set flow error
            }
        }
    """


    # We collect the loss values in a dictionary. ``prediction`` contains the L2 prediction error, ``loss`` contains
    # actual training loss. This is useful to compare the performance of different neural network settings and for
    # hyperparameter optimisation. We also track the test loss on the test set
    epoch_loss_dict = dict(
        (k, {"net_migration": [], "stock": [], "flow": [], "outflow": []})
        for k in ["prediction", "loss"]
    )

    # Collect the flow predictions and stock predictions from each batch
    epoch_flow_predictions = []
    epoch_stock_predictions = [epoch_init_stock]

    # Training loss
    epoch_loss = torch.tensor(0.0, requires_grad=True)

    # Hidden state
    if cfg['NeuralNet'].get('latent_space_dim', 0) > 0:
        h_t = torch.zeros((EdgeIndices.shape[1], cfg['NeuralNet']['latent_space_dim']), device=device, dtype=torch.float)
    else:
        h_t = None

    # Pass over the batches and perform a gradient descent step after batch_size many steps.
    for batch_idx in range(len(BatchYears)):
        batch_loss, batch_stock_prediction, batch_flow_prediction, epoch_loss_dict, h_t = batch(
            batch_idx,
            epoch_stock_predictions[-1],
            dict(
                (k, {"net_migration": [], "stock": [], "flow": [], "outflow": []})
                for k in ["prediction", "loss"]
            ),
            h_t
        )

        # Add loss to epoch and perform a gradient descent step
        epoch_loss = epoch_loss + batch_loss
        if batch_idx > 0 and ((batch_idx % BATCH_SIZE == 0) or (batch_idx == BATCH_SIZE-1)):

            # Gradient descent step
            epoch_loss.backward()

            # Gradient clipping: this can prevent the optimizer from taking larger steps and stabilise training performance,
            # but usually also slows training down
            if cfg["Training"].get("clip_grad_norm", False):
                torch.nn.utils.clip_grad_norm_(
                    NN.parameters(), max_norm=cfg["Training"].get("clip_grad_norm", 1.0)
                )

            NN.optimizer.step()
            NN.optimizer.zero_grad()

            del epoch_loss
            epoch_loss = torch.tensor(0.0, requires_grad=True)

        # Track the flow predictions for testing
        epoch_flow_predictions.append(batch_flow_prediction)

        # Update the initial stock value
        epoch_stock_predictions.append(batch_stock_prediction)

    # Return epoch-averaged values
    epoch_loss_dict = dict(
        (k, dict((v, torch.nanmean(torch.stack(epoch_loss_dict[k][v]))) for v in epoch_loss_dict[k].keys())) for k in epoch_loss_dict.keys()
    )

    # Calculate the test error
    epoch_loss_dict['test'] = dict(flow=
        abs((torch.masked_select(
            torch.cat(epoch_flow_predictions), FlowTestMask
        ) -
        torch.masked_select(
            Flow, FlowTestMask
        )
        )).mean()
    )

    return epoch_loss_dict

# Perform a Yeo-Johnson transform on the batched stock data and net migration data, calculate means and
# standard deviations, and centre and normalise the transformed stock values.
# We need to store the means and standard deviations in order to be able to
# apply them to the transformed predictions later on.

# ----------------------------------------------------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------------------------------------------------
# Print table header
print('Commencing training.')
print(
    "{:<10}| {:<60}  | {:<60}  | {:<8}  | {:<5}".format(
        "Epoch", "Prediction", "Loss", "Test err", "Time [s]"
    )
)
print("—" * 165)
print(
    "{:<10}| {:<14}| {:<14}| {:<14}| {:<14}| {:<14}| {:<14}| {:<14}| {:<14}| {:<10}| {:<5}".format(
        "",
        "Stock",
        "Net migr.",
        "Flow",
        "Outflow",
        "Stock",
        "Net migr.",
        "Flow",
        "Outflow",
        "",
        "",
    )
)
print("—" * 165)
# Train for n epochs
e0 = LossDict['epoch'][-1]+1 if LossDict['epoch'] else 1

# Initial value of stocks
InitStock = Stock[0]

for ep in range(e0, N_EPOCHS + e0):

    # Run the epoch and track the compute time
    t0 = time.time()
    _l = epoch(InitStock)
    dt = time.time() - t0

    # Track the epoch-averaged (training) loss and prediction (test) error
    if not LossDict['epoch']:
        LossDict['epoch'] = [1]
    else:
        LossDict['epoch'].append(ep)
    for key in _l.keys():
        for sub_key in _l[key].keys():
            LossDict[key][sub_key].append(_l[key][sub_key].cpu().numpy())

    # Print the table
    _ep_str = f"{LossDict['epoch'][-1]:<10d}"
    print(
        f"{_ep_str}|"
        f"{LossDict['prediction']['stock'][-1]:<14.4f} | "
        f"{LossDict['prediction']['net_migration'][-1]:<14.4f}| "
        f"{LossDict['prediction']['flow'][-1]:<14.4f}| "
        f"{LossDict['prediction']['outflow'][-1]:<14.4f}| "
        f"{LossDict['loss']['stock'][-1]:<14.7f}| "
        f"{LossDict['loss']['net_migration'][-1]:<14.7f}| "
        f"{LossDict['loss']['flow'][-1]:<14.7f}| "
        f"{LossDict['loss']['outflow'][-1]:<14.7f}| "
        f"{LossDict['test']['flow'][-1]:<10.4f}| "
        f"{dt:<5.4f}"
    )

    # Save trained model, initial hidden state (stock), and loss by components
    if not dry_run and (ep % WRITE_EVERY == 0 or (ep-e0) == N_EPOCHS - 1):
        torch.save(NN.state_dict(), f"{save_to_path}/model_trained.pt")
        torch.save(NN.optimizer.state_dict(), f"{save_to_path}/optim.pt")
        with open(f"{save_to_path}/loss_dict.pickle", "wb") as file:
            pickle.dump(
                dict(
                    (
                        k,
                        dict(
                            (kk, np.array(vv).flatten().tolist())
                            for kk, vv in LossDict[k].items()
                        ),
                    ) if k !='epoch' else (k, v)
                    for k, v in LossDict.items()
                ),
                file,
            )
