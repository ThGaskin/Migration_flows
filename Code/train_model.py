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

from neural_net import NeuralNet
from utils import yeo_johnson_transform, build_input

yaml = YAML(typ="safe")

# Load the configuration
with open(sys.argv[1], "r") as file:
    cfg = yaml.load(file)

# Set default device
device = cfg["device"]

# Load the base paths
BASE_PATH = cfg["BASE_PATH"]

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

# Load args, passed to ``torch.load``
load_args = cfg["Data_loading"].get("load_args", {})

# Net migration data and mask of entries excluded from the optimisation; weights quantifying the uncertainty on each
# entry (currently all 1)
NetMigration = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/net_migration.pt"), **load_args
    ).to(device).float()
)
NetMigrationMask = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/net_migration_mask.pt"),
        **load_args,
    ).to(device).bool()
)
NetMigrationWeights = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/net_migration_weights.pt"), **load_args
    ).to(device).float()
)

# Stock, stock differences, and mask of entries excluded from the optimisation;
# weights quantifying the uncertainty on each entry
Stock = (
    torch.load(os.path.expanduser(f"{BASE_PATH}/{data_path}/stock.pt"), **load_args).to(device).float()
)
# StockMask = (
#     torch.load(os.path.expanduser(f"{BASE_PATH}/{data_path}/stock_mask.pt"), **load_args).to(device).bool()
# )
# StockWeights = (
#     torch.load(os.path.expanduser(f"{BASE_PATH}/{data_path}/stock_weights.pt"), **load_args).to(device).float()
# )
StockDifferences = (
    torch.load(os.path.expanduser(f"{BASE_PATH}/{data_path}/stock_diff.pt"), **load_args).to(device).float()
)
StockDifferenceMask = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/stock_diff_mask.pt"), **load_args
    ).to(device).bool()
)
StockDifferenceWeights = (
    torch.load(os.path.expanduser(f"{BASE_PATH}/{data_path}/stock_diff_weights.pt"), **load_args).to(device).float()
)
# Array of the year indices to which the stock data correspond (typically every 5 years)
StockRange = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/stock_range.pt"), **load_args
    ).to(device).int().tolist()
)

# Total population of each country, acting as an upper bound for outflows, as well the total number of births
# and death rates for each country over time
Population = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/total_population.pt"), **load_args
    ).to(device).float()
)
Births = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/total_births.pt"), **load_args
    ).to(device).float()
)
# Reshape into a diagonal matrix for each year
Births = torch.stack([torch.diag(Births[i]) for i in range(len(Births))])

# Death rates for each destination country and year
DeathRates = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/death_rate.pt"), **load_args
    ).to(device).float()
).reshape(-1, 1, Births.shape[-1])

# Known flow data and mask; weights quantifying uncertainty on each entry (currently only not 1 for Quantmig
# entries)
Flow = (
    torch.load(os.path.expanduser(f"{BASE_PATH}/{data_path}/flow.pt"), **load_args).to(device).float()
)
FlowMask = (
    torch.load(os.path.expanduser(f"{BASE_PATH}/{data_path}/flow_mask.pt"), **load_args).to(device).bool()
)
FlowWeights = (
    torch.load(os.path.expanduser(f"{BASE_PATH}/{data_path}/flow_weights.pt"), **load_args).to(device).float()
)

# Network edges to train on
EdgeIndices = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/edge_indices.pt"), **load_args
    ).to(device).long()
)

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
    """Transforms a dataset using a transformation dictionary, and returns all the information needed to recreate
    the transform.

    :param _data_dict: a dictionary containing the following keys:
        - data: the data to transform
        - mask: a mask to select entries to transform
        - batch_indices: batch indices used to group time series slices together
        - cfg: dictionary of transformation parameters, passed to `yeo_johnson_transform`
    :return: a dictionary containing the transformed dataset, the means and standard deviations of the (uncentralised)
        transformed data, and the dictionary of transformation parameters, as well the original untransformed data
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

    :param h_t: the hidden state
    :param batch_idx:
    :param batch_stock_init:
    :param epoch_loss_dict:
    :return:
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

        # Create an empty flow table and fill most of the edges with detached (gradient-free) predictions
        T = torch.zeros(N, N, N, device=device)

        # Randomly select RANDOM_SAMPLE_SIZE many indices from the edge list and populate the flow table
        shuffled_edge_indices = torch.randperm(EdgeIndices.shape[1])
        undetached = shuffled_edge_indices[
                     : min(EdgeIndices.shape[1] - 1, RANDOM_SAMPLE_SIZE)
                     ]
        detached = shuffled_edge_indices[
                   min(EdgeIndices.shape[1] - 1, RANDOM_SAMPLE_SIZE):
                   ]
        # Prediction in the case of a latent dimension
        if h_t is not None:

            # Make a prediction on the undetached edges
            idx_i, idx_j, idx_k = EdgeIndices[:, undetached]
            res = NN(torch.cat([TrainingData[t][undetached],
                                   stock_input[idx_i, idx_j].unsqueeze(1),
                                   stock_input[idx_i, idx_k].unsqueeze(1),
                                   h_t[undetached]],
                        dim=1))
            T[idx_i, idx_j, idx_k] = Scale * torch.exp(res[:, 0])
            h_t[undetached, :] = res[:, 1:]

            # Make a prediction on the detached edges
            idx_i, idx_j, idx_k = EdgeIndices[:, detached]
            res = NN(torch.cat([TrainingData[t][detached],
                                stock_input[idx_i, idx_j].unsqueeze(1),
                                stock_input[idx_i, idx_k].unsqueeze(1),
                                h_t[detached]],
                               dim=1)).detach()
            T[idx_i, idx_j, idx_k] = Scale * torch.exp(res[:, 0])
            h_t[detached, :] = res[:, 1:]
        else:

            # Make a prediction on the undetached edges
            idx_i, idx_j, idx_k = EdgeIndices[:, undetached]
            T[idx_i, idx_j, idx_k] = (
                    Scale * torch.exp(NN(
                        torch.cat([TrainingData[t][undetached],
                                   stock_input[idx_i, idx_j].unsqueeze(1),
                                   stock_input[idx_i, idx_k].unsqueeze(1)],
                                  dim=1))).flatten()
            )

            # Make a prediction on the detached edges
            idx_i, idx_j, idx_k = EdgeIndices[:, detached]
            T[idx_i, idx_j, idx_k] = (
                    Scale * torch.exp(NN(
                        torch.cat([TrainingData[t][detached],
                                   stock_input[idx_i, idx_j].unsqueeze(1),
                                   stock_input[idx_i, idx_k].unsqueeze(1)],
                                  dim=1)).detach()).flatten()
            )

        # Predict the stock of next year, taking demographics into account. Births in a given country increase the
        # native-born stock.
        stock_prediction = (
                Births[t]
                + (1 - DeathRates[t]) * stock_prediction
                + T.sum(dim=1) - T.sum(dim=2)
        )

        # Calculate the total flow by summing over all birthplaces
        flow = T.sum(dim=0)
        batch_flow_predictions.append(flow)

        # Predict the net migration
        batch_net_migration_predictions.append(flow.sum(dim=0) - flow.sum(dim=1))

        # The total outflow also cannot exceed the total population
        outflow_error_population = torch.relu(
            flow.sum(dim=1) - Population[t]
        ).mean()

        # Add to loss. The components are each scaled with the std of the stock and population to balance
        # their contribution to the loss with the stock and net migration errors. Without this balancing they will
        # dominate the loss and decrease training performance
        total_additional_err = cfg["Training"]["weight_factors"]["regulariser"] * (
                inv_pop_std * outflow_error_population
              #+ inv_stock_std * outflow_error_stock
        )
        batch_loss = batch_loss + total_additional_err
        epoch_loss_dict["prediction"]["outflow"].append(
            outflow_error_population.detach()
            # + outflow_error_stock.detach()
        )
        epoch_loss_dict["loss"]["outflow"].append(total_additional_err.clone().detach())

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
    """An epoch is a single pass over the entire time interval covered by the data. One interval between stock data
    points (``stock_range``) is defined as a batch.

    Each batch is processed in the following way: given an initial and final stock data point, the model learns to
    interpolate flows between the two in such a way that stock and net migration values match. After a certain number of
    steps (``batch_size``), a gradient descent step is made on the neural network parameters. Performing a step only
    after a full pass over the entire dataset is called 'batch gradient descent', performing a step after every
    batch is called 'stochastic gradient descent'.

    Since the flow table is very large, memory might not be sufficient to optimise the predictions on all the edges
    at the same time. For this reason, we compute gradients on a random subsample of edges (``sample_size`` ) --
    the random subsample changes for every time frame t, so over the entire batch and many epochs all edges
    will be optimised.

    :param: training_data_dict: a dictionary containing the (transformed) target net migration, stock, and flow data,
        as well as their means, stds, and transformation parameters. The structure of each entry is:

        ```
        stock:
            - transformed_data: ...
            - mean: ...
            - std: ...
            - transformation_params:
                - lmbda: ...
                - other kwargs, passed to `yeo_johnson_transform`
        ```
    :param epoch_init_stock: the initial value of the stocks. This is used to start the recurrent process and is
        adjusted over time
    :return: dictionary of losses and prediction errors
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
    # TODO: Should we calculate a median instead?
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
