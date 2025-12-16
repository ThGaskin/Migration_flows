import copy
import pickle
import os
import numpy as np
import torch
import tqdm
import xarray as xr
from ruamel.yaml import YAML
from typing import Any, Tuple, Union

yaml = YAML(typ='safe')

import sys
from os.path import dirname as up
from dantro._import_tools import import_module_from_path
sys.path.append(up(__file__))
Code = import_module_from_path(mod_path=up(__file__), mod_str="Code")

from Code import NeuralNet

""" Utility functions used to generate predictions using a neural network """

def yeo_johnson_transform(data: Union[torch.Tensor, np.ndarray],
                          lmbda: Union[torch.Tensor, float],
                          *,
                          flip_negative_values: bool = True,
                          mean: Union[torch.Tensor, float, None] = None,
                          std: Union[torch.Tensor, float, None] = None,
                          standardize: bool = False
) -> Union[torch.Tensor, np.ndarray]:

    """ Yeo-Johnson transform with parameter lmbda. By default, the transformation is symmetric. If specified,
    a mean-zero and unit-variance distribution is returned. This function allows handling both torch.Tensors
    and numpy.ndarrays. The function returns the input type.

    :param data: data to transform
    :param lmbda: transformation parameter
    :param flip_negative_values: use symmetrised version
    :param standardize: return a mean-zero unit-variance distribution
    :param __: other kwargs (ignored)
    :return: transformed data
    """
    _conv_to_np = isinstance(data, np.ndarray)
    if _conv_to_np:
        data = torch.from_numpy(data).float()

    if lmbda == 1:
        res = data
    else:
        # Vectorised Yeo-Johnson transform for pytorch tensors
        mask_pos = data >= 0
        mask_neg = ~mask_pos
        res = torch.zeros_like(data)

        # Handle positive values
        if lmbda != 0:
            res[mask_pos] = (torch.pow(data[mask_pos] + 1, lmbda) - 1) / lmbda
        else:
            res[mask_pos] = torch.log(data[mask_pos] + 1)

        # Handle negative values
        if flip_negative_values:
            neg_data = -data[mask_neg]
            if lmbda != 0:
                res[mask_neg] = -(torch.pow(neg_data + 1, lmbda) - 1) / lmbda
            else:
                res[mask_neg] = -torch.log(neg_data + 1)
        else:
            if lmbda != 2:
                res[mask_neg] = -(torch.pow(-data[mask_neg] + 1, 2 - lmbda) - 1) / (2 - lmbda)
            else:
                res[mask_neg] = -torch.log(-data[mask_neg] + 1)

    if mean is None and standardize:
        mean = torch.nanmean(res)
    if std is None and standardize:
        std = torch.std(res)
    if mean is not None:
        res = res - mean
    if std is not None:
        res = res/std
    if _conv_to_np:
        return res.numpy()
    else:
        return res

def inv_yeo_johnson(data: Union[torch.Tensor, np.ndarray],
                    lmbda: Union[torch.Tensor, float],
                    mean: Union[torch.Tensor, float] = 0.0,
                    std: Union[torch.Tensor, float] = 1.0, *,
                    flip_neg_values=True
                    ) -> Union[torch.Tensor, np.ndarray]:
    """ Inverse Yeo-Johnson transform. Reverses standardisations, if passed.

    :param data: data to transform
    :param lmbda: transformation parameter
    :param mean: mean of the original distribution; the standardization is undone first
    :param std: standard deviation of the original distribution
    :param flip_neg_values: whether to use the symmetrised version
    :return: inversely transformed data
    """

    _conv_to_np = isinstance(data, np.ndarray)
    if _conv_to_np:
        data = torch.from_numpy(data).float()

    # Undo standardization if applied
    data = data * std + mean

    mask_pos = data >= 0
    mask_neg = data < 0
    res = torch.zeros_like(data)

    # Handle positive values
    if not (mask_pos == False).all():
        if lmbda != 0:
            res[mask_pos] = torch.pow(data[mask_pos] * lmbda + 1, 1 / lmbda) - 1
        else:
            res[mask_pos] = torch.exp(data[mask_pos]) - 1

    # Handle negative values
    if not (mask_neg == False).all():
        if flip_neg_values:
            res[mask_neg] = -inv_yeo_johnson(-data[mask_neg], lmbda, mean=0, std=1, flip_neg_values=False)
        else:
            if lmbda != 2:
                res[mask_neg] = 1 - torch.pow(-data[mask_neg] * (2 - lmbda) + 1, 1 / (2 - lmbda))
            else:
                res[mask_neg] = 1 - torch.exp(-data[mask_neg])

    if _conv_to_np:
        return res.numpy()
    else:
        return res


def build_input(cfg, edges, Y, device: str = 'cpu', *, transformation_params: dict = {}) -> torch.Tensor:
    """ Builds an input tensor from a list of edge indices and a configuration of paths to covariates. Inputs are
    loaded sequentially in the order given by the configuration, and stacked into a single tensor. Time-independent
    covariates are repeated Y times (number of years). By default, the `NATIVE` covariate (i == j) and `RETURN'
    covariate (i == k) are added to the end of the input data.

    If a dictionary of transformation parameters is passed, the unscaled covariates are loaded, the inputs are sampled
    from a normal distribution, and the samples transformed using the given transformation parameters.

    :param cfg: configuration of data paths to covariates
    :param edges: list of edge indices, of shape (3, N_edges)
    :param Y: number of years, used to reshape time-independent covariates
    :param device: training device (cpu by default)
    :param transformation_params: a dictionary of transformation parameters. If None, parameters are sampled from a
    :return: torch.Tensor of inputs
    """

    # Collect covariates in a single input tensor
    input = []

    # Map index keys to numbers
    idx_dict = {'i': 0, 'j': 1, 'k': 2}

    # Path to data
    for p in cfg['Data_loading']['covariates']:

        name = list(p.keys())[0]
        info = list(p.values())[0]

        # If parameter is to be sampled, load unscaled parameter
        load_unscaled = name in transformation_params.keys() and transformation_params.get(name, {}).get(
            'transformation_parameters', None) is not None
        if load_unscaled:
            tensor = torch.load(
                os.path.expanduser(
                    f"{cfg['BASE_PATH']}/{cfg['Data_loading']['data_path']}/{info['path'].replace('input_covariates', 'unscaled_covariates')}.pt"),
                **cfg['Data_loading']['load_args']
            ).float().to(device)
        else:
            tensor = torch.load(
                os.path.expanduser(f"{cfg['BASE_PATH']}/{cfg['Data_loading']['data_path']}/{info['path']}.pt"),
                **cfg['Data_loading']['load_args']
            ).float().to(device)

        # Sample, if given
        sample = name in transformation_params.keys()
        if sample:
            tensor = torch.normal(tensor, std=tensor.abs() * transformation_params[name]['sample_std'])
            if name not in ['GDP_growth', 'Refugees_diff']:
                tensor = torch.maximum(tensor, torch.tensor(0.0))

        # Transform using a Yeo-Johnson scaling, if required
        if load_unscaled:
            tensor = yeo_johnson_transform(tensor, **transformation_params[name]['transformation_parameters'])

        for _idx in info['idx']:

            # Unilateral
            if isinstance(_idx, str):
                if tensor.dim() == 2:
                    input.append(tensor[:, edges[idx_dict[_idx]]])
                else:
                    input.append(tensor[edges[idx_dict[_idx]]].expand(Y, -1))

            # Bilateral covariates
            else:
                # Time-dependent bilateral covariate
                if tensor.dim() == 3:
                    input.append(tensor[:, edges[idx_dict[_idx[0]]], edges[idx_dict[_idx[1]]]])
                # Time-independent bilateral covariate
                else:
                    input.append(
                        (tensor[edges[idx_dict[_idx[0]]], edges[idx_dict[_idx[1]]]]).unsqueeze(dim=0).expand(Y, -1))

    # Native of origin
    input.append(
        (edges[0] == edges[1]).float().expand(Y, -1)
    )

    # Native of destination
    input.append(
        (edges[0] == edges[2]).float().expand(Y, -1)
    )
    return torch.cat([x.unsqueeze(2) for x in input], dim=2)


def load_training_data(dir, cfg, *, device: str = 'cpu') -> dict:
    """ Loads data from a directory into a dictionary, and constructs the input data. If available, also loads ground truth datasets.

    :param dir: directory containing the training data and input covariates
    :param cfg: load configuration, specifying which covariates to load
    :param device: storage device for torch.Tensors; 'cpu' by default
    :return: dictionary containing all data required for running the model.
    """
    res: dict = {'S': torch.load(f"{dir}/stock.pt", weights_only=True, map_location=torch.device(device)),
                 'F': torch.load(f"{dir}/flow.pt", weights_only=True, map_location=torch.device(device)),
                 'mu': torch.load(f"{dir}/net_migration.pt", weights_only=True, map_location=torch.device(device)),
                 'S_mask': torch.load(f"{dir}/stock_mask.pt", weights_only=True, map_location=torch.device(device)),
                 'F_mask': torch.load(f"{dir}/flow_mask.pt", weights_only=True, map_location=torch.device(device)),
                 'mu_mask': torch.load(f"{dir}/net_migration_mask.pt", weights_only=True,
                                       map_location=torch.device(device)),
                 'death_rate': torch.load(f"{dir}/death_rate.pt", weights_only=True, map_location=torch.device(device)),
                 'total_births': torch.load(f"{dir}/total_births.pt", weights_only=True,
                                            map_location=torch.device(device)),
                 'stock_range': torch.load(f"{dir}/stock_range.pt", weights_only=True,
                                           map_location='cpu').int().numpy().tolist()}

    # Number of years, countries, and initial stocks
    res['Y'] = res['mu'].shape[0]
    res['N'] = res['mu'].shape[1]
    res['S_0'] = res['S'][0]

    # Tensors of edge indices
    res['edge_indices'] = torch.load(f"{dir}/edge_indices.pt", weights_only=True,
                                     map_location=torch.device(device)).long().to(device)

    # Transformation parameters used to scale the stocks
    with open(f'{dir}/transformation_parameters.pickle', 'rb') as handle:
        transformation_parameters = pickle.load(handle)
    res['transformation_parameters'] = transformation_parameters

    # Also load true values, if given
    for label, item in [("T_true", "true_flow_table"), ("S_true", "true_stock"), ("F_true", "true_flow"),
                        ("mu_true", "true_net_migration")]:
        try:
            res[label] = torch.load(f"{dir}/{item}.pt", weights_only=True, map_location=torch.device(device))
            if label == 'S_true':
                res['S_0'] = res[label][0]
        except:
            print(f"INFO: No ground truth values {item} found.")
            continue

    # Build the input data
    res['input_data'] = build_input(cfg, res['edge_indices'], res['Y'], device=device)

    return res


def get_NN(dir: str, *, device: str = 'cpu') -> Tuple[NeuralNet, dict]:

    """ Builds and loads a neural network from a directory.
    :param dir: directory from which to load the neural network
    :param device: torch.device on to which to load the neural network
    :return: neural network and configuration tuple
    """
    with open(f"{dir}/cfg.yaml", "r") as file:
        nn_cfg = yaml.load(file)

    # Load the neural network weights
    weights = torch.load(f"{dir}/model_trained.pt", weights_only=True, map_location=torch.device(device))

    # Set up the neural network.
    NN = NeuralNet(
        input_size=weights['layers.0.weight'].shape[1],
        output_size=weights[
            max([s for s in list(weights.keys()) if s.endswith('.weight')], key=lambda x: int(x.split('.')[1]))].shape[
            0],
        **nn_cfg["NeuralNet"]
    ).to(device)

    # Set the weights
    NN.load_state_dict(weights)
    NN.eval()

    return NN, nn_cfg

def generate_predictions(*,
                         NN: NeuralNet = None,
                         dir: str = None,
                         edge_indices: torch.Tensor,
                         input_data: torch.Tensor,
                         S_0: torch.Tensor,
                         total_births: torch.Tensor,
                         death_rate: torch.Tensor,
                         transformation_parameters: dict,
                         device: str = 'cpu',
                         show_pbar: bool = True,
                         scaling_factor: Union[torch.Tensor, float] = 1000.,
                         apply_tanh_to_latent_space: bool = False,
                         generate_full_T: bool = True,
                         **__
                         ) -> dict:
    """Generates predictions using a neural network. The neural network can be passed directly or loaded from a
    directory.

    :param NN: NeuralNetwork to use. If None, a directory containing a network to load must be passed
    :param dir: path from which to load the neural network, if none is passed
    :param edge_indices: list of edge indices
    :param input_data: torch.Tensor of input data
    :param S_0: initial stocks
    :param total_births: total number of births in each year, by country
    :param death_rate: death rate for each country
    :param transformation_parameters: dictionary of Yeo-Johnson transformation values
    :param device: training device
    :param show_pbar: show the progress bar during evaluation
    :param scaling_factor: scale to use for the neural network output
    :param apply_tanh_to_latent_space: whether to apply the tanh function to the latent space output of an RNN
        (for backwards compatibility purposes only)
    :param generate_full_T: whether to generate the full flow table. If False, the compressed version is returned.
    :param __: other parameters (ignored)
    :return: dictionary containing the predictions
    """

    # Load the neural network from the directory, if provided
    if NN is None:
        if dir is None:
            raise ValueError("Must supply one of 'NN', 'dir'!")
        NN, _ = get_NN(dir, device=device)

    # Edge indices
    idx_i, idx_j, idx_k = edge_indices

    # Initialise the hidden state
    h_t = torch.zeros((idx_i.shape[0], NN.output_dim - 1), device=device)

    Y = input_data.shape[0]
    N = S_0.shape[0]
    T_pred = torch.zeros((Y, edge_indices.shape[-1]), device=device)
    stock_predictions = [S_0]
    flow_predictions = torch.zeros(Y, N, N, device=device)  # total flow[j,k]

    for y in tqdm.tqdm(range(Y), leave=False) if show_pbar else range(Y):

        # Make a prediction and fill the flow table. Recurrent architectures include the hidden state as an input, which is recursively updated.
        _input_data = torch.cat([
            input_data[y],
            yeo_johnson_transform(
                stock_predictions[-1][idx_i, idx_j], **transformation_parameters['Stock']
            ).unsqueeze(1),
            yeo_johnson_transform(
                stock_predictions[-1][idx_i, idx_k], **transformation_parameters['Stock']
            ).unsqueeze(1)], dim=1
        )

        # Initialize aggregation tensors
        net_stock_flow = torch.zeros(N, N, device=device)  # inflow - outflow for stock[i,k]

        # Append latent state to neural network input
        if NN.output_dim > 1:
            _input_data = torch.cat([_input_data, h_t], dim=1)
            with torch.no_grad():
                res = NN(_input_data)
            log_flow, h_t = res[:, 0], res[:, 1:]
            if apply_tanh_to_latent_space:
                h_t = h_t.tanh()
        else:
            with torch.no_grad():
                log_flow = NN(_input_data).flatten()
        T_pred[y, :] = scaling_factor * torch.exp(log_flow)

        net_stock_flow.index_put_((idx_i, idx_k), T_pred[y], accumulate=True)  # inflow
        net_stock_flow.index_put_((idx_i, idx_j), -T_pred[y], accumulate=True)  # outflow
        flow_predictions[y].index_put_((idx_j, idx_k), T_pred[y], accumulate=True)

        # Update the stock predictions
        stock_predictions.append(torch.maximum(
            torch.tensor(0.0),
            (1 - death_rate[y]).reshape(1, N) * stock_predictions[-1] + torch.diag(total_births[y]) + net_stock_flow)
        )

    # Combine predictions and return a dictionary
    stock_predictions = torch.stack(stock_predictions)
    net_migration_predictions = flow_predictions.sum(dim=1) - flow_predictions.sum(dim=2)

    # Generate the full flow tensor if required
    if generate_full_T:
        n = idx_i.shape[0]
        _T_pred = torch.zeros(Y, N, N, N, device=T_pred.device, dtype=T_pred.dtype)
        _T_pred[torch.arange(Y, device=T_pred.device)[:, None].expand(Y, n),
                idx_i[None, :].expand(Y, n),
                idx_j[None, :].expand(Y, n),
                idx_k[None, :].expand(Y, n)
        ] = T_pred
        T_pred = _T_pred

    return dict(T_pred=T_pred,
                S_pred=stock_predictions,
                mu_pred=net_migration_predictions,
                F_pred=flow_predictions,
                NN=NN)

def convert_tensor_predictions_to_xarray(*,
                                         T_pred: torch.Tensor = None,
                                         S_pred: torch.Tensor,
                                         mu_pred: torch.Tensor,
                                         F_pred: torch.Tensor,
                                         years: Union[np.ndarray, None] = None,
                                         countries: Union[np.ndarray, None] = None,
                                         **__
                                         ) -> dict:
    """ Converts torch.Tensors into xarray items for easier indexing and plotting.

    :param T_pred: torch.Tensor of total flow predictions, of shape (Y, N, N, N) (optional)
    :param S_pred: torch.Tensor of migrant stocks, of shape (Y, N, N)
    :param mu_pred: torch.Tensor of net migration, of shape (Y, N)
    :param F_pred: torch.Tensor of flows, of shape (Y, N, N)
    :param years: (optional) coordinates to use for the year dimension. If None, an array of indices is created.
    :param countries: (optional) coordinates to use for the countries. If None, an array of indices is created.
    :param __: other kwargs (ignored)
    :return: dictionary of converted xr.DataArray objects
    """

    if years is None:
        years = np.arange(F_pred.shape[0])
    if countries is None:
        countries = np.arange(F_pred.shape[1])

    res = dict(S_pred=xr.DataArray(
        data=S_pred.cpu(),
        dims=["Year", "Origin ISO", "Destination ISO"],
        coords={"Year": np.append(years, years[-1] + 1), "Origin ISO": countries, "Destination ISO": countries},
        name="Migrant stocks"
    ), F_pred=xr.DataArray(
        data=F_pred.cpu(),
        dims=["Year", "Origin ISO", "Destination ISO"],
        coords={"Year": years, "Origin ISO": countries, "Destination ISO": countries},
        name="Total flows"
    ), mu_pred=xr.DataArray(
        data=mu_pred.cpu(),
        dims=["Year", "Country ISO"],
        coords={"Year": years, "Country ISO": countries},
        name="Net migration"
    ))

    # Add the full flow table if possible; else
    if T_pred is not None and T_pred.dim() == 4:
        res['T_pred'] = xr.DataArray(
            data=T_pred.cpu(),
            dims=["Year", "Birth ISO", "Origin ISO", "Destination ISO"],
            coords={"Year": years, "Birth ISO": countries, "Origin ISO": countries, "Destination ISO": countries},
            name="Flow table"
        )
    elif T_pred is not None:
        res['T_pred'] = T_pred

    return res


def aggregate(arr: xr.DataArray, years, *, label: str = 'lower', dim_name: str = 'Year0'):
    """ Calculates total values over interval periods, useful for e.g. calculating 5-year totals or nets.

    :param arr: the xr.DataArray to aggregate over the 'Year' dimension
    :param years: intervals of years
    :param label: which interval bound to use for the coordinate index
    :param dim_name: name to use for the new dimension of aggregated years
    :return: xr.DataArray of aggregated flows
    """

    return xr.concat([
        arr.sel({"Year": range(years[i], years[i + 1])}).sum('Year').expand_dims(
            {dim_name: [years[i] if label == 'lower' else [years[i + 1]]]}
        ) for i in range(len(years) - 1)
    ], dim=dim_name)


def aggregate_T(T, years):
    """ Calculates total flows (T) over intervals of years, calculating both flows by residence and by birth.

    :param T: flow table of dimensions (Year, Birth ISO, Origin ISO, Deestination ISO)
    :param years: intervals over which to cumulate
    :return: table of aggregated flows
    """

    # Flows by residence
    cor = T.sum("Birth ISO").expand_dims({"Definition": ['residence']})

    # Flows by birth, coverage: total
    cob = T.sum("Origin ISO").rename({"Birth ISO": "Origin ISO"}).expand_dims({"Definition": ['birth']})

    # Sum over five-year periods
    _f = xr.concat([cor, cob], dim='Definition').expand_dims({"Coverage": ['Total']})
    _f = aggregate(_f, years).expand_dims({"Method": ['NN']})

    return _f.transpose('Method', 'Definition', 'Coverage', 'Year0', 'Origin ISO', 'Destination ISO')


def expand_T(T_pred: torch.Tensor, data: dict, *, countries: np.ndarray = None, years: np.ndarray = None,
             return_xarray: bool = True) -> Union[xr.Dataset, torch.Tensor]:
    """ Expands the compressed full flow table T samples (containing means and standard deviations)
    to the full (Y, N, N, N)-dimensional flow table.

    :param T_pred: compressed flow table of mean and standard deviations
    :param data: dictionary containing N, Y, and the edge indices
    :param countries: array of country coordinates to use (optional)
    :param years: array of year coordinates to use (optional)
    :param return_xarray: return an xr.Dataset; if False, returns a stacked torch.Tensor
    :return: xr.Dataset or torch.Tensor of mean and std T values
    """

    # Shapes of the dataset
    Y, N = data['Y'], data['N']
    idx_i, idx_j, idx_k = data['edge_indices']
    n = idx_i.shape[0]

    # Expand compressed flow table, containing mean and standard deviations
    T_full = torch.zeros(2, Y, N, N, N)
    T_full[0, torch.arange(Y)[:, None].expand(Y, n),
                    idx_i[None, :].expand(Y, n),
                    idx_j[None, :].expand(Y, n),
                    idx_k[None, :].expand(Y, n)] = T_pred[0]
    T_full[1, torch.arange(Y)[:, None].expand(Y, n),
                    idx_i[None, :].expand(Y, n),
                    idx_j[None, :].expand(Y, n),
                    idx_k[None, :].expand(Y, n)] = T_pred[1]

    if return_xarray:
        if years is None:
            years = np.arange(Y)
        if countries is None:
            countries = np.arange(N)
        return xr.Dataset(dict(
            mean=xr.DataArray(
                    data=T_full[0].cpu(),
                    dims=["Year", "Birth ISO", "Origin ISO", "Destination ISO"],
                    coords={"Year": years, "Birth ISO": countries, "Origin ISO": countries, "Destination ISO": countries}),
            std=xr.DataArray(
                    data=T_full[1].cpu(),
                    dims=["Year", "Birth ISO", "Origin ISO", "Destination ISO"],
                    coords={"Year": years, "Birth ISO": countries, "Origin ISO": countries, "Destination ISO": countries}))
        )
    else:
        return T_full

def get_stock_offsets(*, stock_predictions: xr.DataArray, stock_data: xr.DataArray, weights: xr.DataArray,
                      gamma: xr.DataArray) -> xr.DataArray:
    """ Calculates an offset value for each edge (origin, destination), such that the L2-error
    on each corridor is minimised. Each error is weighted using the 'weights' table.
    The offset is calculated by including the death rates, to ensure the stocks remain demographically balanced.
    Stocks cannot be negative, so an iterative procedure is employed to find the minimum offset to ensure non-negativity
    of the stocks.

    :param stock_predictions: xr.DataArray of predictions
    :param stock_data: xr.DataArray of true values
    :param stock_weights: xr.DataArray of weights
    :param gamma: xr.DataArray of fraction of population alive at start of stock data (Jan 1990) still alive at start of each year.
    :return: xr.DataArray of stock offsets
    """

    # Initial offset
    offset = ((stock_data - stock_predictions) * weights * gamma).sum('Year')
    offset /= (weights * gamma ** 2).sum('Year')
    offset = offset.fillna(0)

    # Iteratively adjust until no more stocks are below 0
    negative_stocks = (stock_predictions + gamma * offset).where(lambda x: x < 0, 0)
    while negative_stocks.sum() != 0:
        offset += xr.where((stock_predictions + gamma * offset) == negative_stocks.min('Year'),
                           - (stock_predictions + gamma * offset) / gamma, 0).sum('Year')
        negative_stocks = (stock_predictions + gamma * offset).where(lambda x: x < 0, 0)

    offset = gamma * offset

    return offset.transpose(*list(stock_data.dims))

def get_elasticities(data: dict,
                     predictions: dict,
                     cfg: dict,
                     *,
                     n_edges: int = 20,
                     n_years: int = 5, device: str) -> tuple[torch.Tensor, list]:
    """Calculates a matrix of elasticities over a batch. A random number of edges and years are selected on which to calculate the elasticities. This is to save memory and also speed up the computation. The batch_size is n_edges * n_years

    :param data: dictionary containing the scaled input data and edge indices
    :param predictions: dictionary containing the neural network and stock predictions
    :param cfg: config pointing to the scaled covariate locations
    :param n_edges: number of random edges to select
    :param n_years: numer of random years to select
    :param device: device to use
    :return: (elasticities, labels): elasticity for each continuous entry of shape (batch_size, p), and the associated labels
    """

    # Collect the transformation parameters for each covariate
    transformation_params = []

    # Also collect covariate labels and their positions in the input vector
    covariate_indices = []
    covariate_names = []
    idx_0 = 0

    # Amend the configuration
    cfg_unscaled = copy.deepcopy(cfg)
    for idx, item in enumerate(cfg_unscaled['Data_loading']['covariates']):
        name = list(item.keys())[0]
        cfg_unscaled['Data_loading']['covariates'][idx][name]['path'] = cfg_unscaled['Data_loading']['covariates'][idx][name]['path'].replace('input_', 'unscaled_')
        for k in range(len(item[name]['idx'])):
            if name in data['transformation_parameters'].keys():
                transformation_params.append((data['transformation_parameters'][name]['lmbda'], data['transformation_parameters'][name]['std']))
                covariate_indices.append(idx_0)
                covariate_names.append(name.replace('_', ' ')+'$_{'+''.join(item[name]['idx'][k])+'}$')
            else:
                transformation_params.append((1.0, 1.0))

                # These two are not transformed but still continuous
                if name in ['Linguistic_similarity', 'Religious_similarity']:
                    covariate_indices.append(idx_0)
                    covariate_names.append(name.replace('_', ' ') + '$_{' + ''.join(item[name]['idx'][k]) + '}$')

            idx_0 += 1

    # Native and return covariates are not scaled
    for _ in range(2):
        idx_0 += 1
        transformation_params.append((1.0, 1.0))

    # Also need the transformation parameters for the stocks
    for _ in range(2):
        transformation_params.append((data['transformation_parameters']['Stock']['lmbda'], data['transformation_parameters']['Stock']['std']))
        covariate_indices.append(idx_0)
        covariate_names.append('Stock$_{ij}$' if _ == 0 else 'Stock$_{ik}$')
        idx_0 += 1

    # Select a random number of edges and years on which to calculate gradients
    edges = data['edge_indices'][:, torch.randperm(data['edge_indices'].shape[1])[:n_edges]]
    years = torch.randperm(data['Y'])[:n_years]

    # Build the scaled and unscaled inputs
    unscaled_input = build_input(cfg_unscaled, edges, data['Y'], device)[years, :].flatten(end_dim=1)
    scaled_input = build_input(cfg, edges, data['Y'], device)[years, :].flatten(end_dim=1)
    stock_input = torch.cat([
         torch.from_numpy(predictions['S_pred'].data).to(device)[:-1, edges[0, :], edges[1, :]].unsqueeze(-1),
         torch.from_numpy(predictions['S_pred'].data).to(device)[:-1, edges[0, :], edges[2, :]].unsqueeze(-1)
    ], dim=-1)[years, :].flatten(end_dim=1)

    # Append stocks to the unscaled input and scaled input
    unscaled_input = torch.cat([unscaled_input, stock_input], dim=-1)
    scaled_input = torch.cat([scaled_input,
                              yeo_johnson_transform(stock_input, **data['transformation_parameters']['Stock']),
                              torch.zeros(scaled_input.shape[0], predictions['NN'].input_dim-(scaled_input.shape[1] + 2),
                                          device=device)], dim=-1)
    scaled_input.requires_grad_(True)

    # Calculate the YJ derivatives
    YJ_derivatives = torch.cat([
        (((torch.abs(unscaled_input[:, idx]) + 1) ** (params[0] - 1)) / (params[1])).unsqueeze(-1)
        for idx, params in enumerate(transformation_params)
    ], dim=-1)

    # Get the neural network outputs
    outputs = predictions['NN'](scaled_input)

    # Extract only the log-flow predictions of shape (batch_size, )
    log_flow = outputs[:, 0]

    # Create identity-like grad_outputs to get per-sample gradients
    grad_outputs = torch.ones_like(log_flow)

    # Calculate elasticities
    grads = torch.autograd.grad(
        outputs=log_flow,
        inputs=scaled_input,
        grad_outputs=grad_outputs,
        create_graph=False,
        retain_graph=True
    )[0][:, :unscaled_input.shape[1]]

    return (grads * YJ_derivatives * unscaled_input)[:, covariate_indices], covariate_names


def get_samples(*,
                dir: Union[str, list] = None,
                NN: NeuralNet = None,
                data: dict,
                device: str = 'cpu',
                cfg: dict = None,
                input_transformation: dict = None,
                show_pbar: bool = True,
                stock_data: xr.Dataset,
                stock_std: float = None,
                gamma: xr.DataArray,
                generate_full_T: bool = True,
                apply_tanh_to_latent_space: bool = False,
                n_samples: int = 0
                ) -> tuple[dict, Any]:

    """ Generate ensemble predictions from a family of trained networks.
    Mean estimates and uncertainties are calculated from the ensemble. This can optionally be done in a memory-efficient
    way, avoiding generating of the full flow table T and instead returning the neural network statistics

    :param dirs: list of directories from which to source the estimates
    :param data: data dictionary containing the information needed to run the neural network
    :param device: device to use
    :param show_pbar: show a progress bar
    :param stock_data: xr.Dataset containing the stock data and weights
    :param gamma: cumulative death rates used to calculate the stock offsets
    :param n_samples: number of samples to draw for the stock data
    :param generate_full_T: avoid generating the full flow table. If False, the mean and std of the neural network
        estimates are returned alongside the dictionary of estimates
    :return: dictionary of estimates and (optionally) the full flow table statistics
    """
    Y, N = data['Y'], data['N']
    n = data['edge_indices'].shape[1]
    samples = dict(
        T_pred=torch.zeros(3, Y, n, device=device),
        F_pred=torch.zeros(3, Y, N, N, device=device),
        S_pred=torch.zeros(3, Y + 1, N, N, device=device),
        mu_pred=torch.zeros(3, Y, N, device=device),
    )

    # Standard deviation on the stocks: taken from the stock dataset or through an explicit standard deviation level,
    # if provided
    S_0 = data['S_0']
    if stock_std is None:
        stock_std = torch.from_numpy(stock_data['Error'].isel({"Year": 0}).fillna(0).data).float().to(device)
    else:
        stock_std = stock_std * S_0
    input_0 = data['input_data']

    # Number of samples seen: needed for Welford's algorithm
    count = 0

    if dir is not None and isinstance(dir, list):
        outer = dir
    elif dir is not None and isinstance(dir, str):
        outer = [dir]
    elif NN is not None:
        outer = [NN]
    else:
        raise ValueError("Missing either dir or NN!")

    with tqdm.tqdm(desc='Sampling', total=len(outer) * (n_samples+1), disable=not show_pbar) as pbar:
        for item in outer:
            for i in range(n_samples+1):

                # Sample the initial stock and input data. The first sample (i=0) is always the central estimate.
                if i > 0:
                    data['S_0'] = torch.maximum(torch.tensor(0.0), torch.normal(S_0, stock_std))
                    if cfg is not None and input_transformation is not None:
                        data['input_data'] = build_input(cfg,
                                                         data['edge_indices'],
                                                         data['Y'],
                                                         device=device,
                                                         transformation_params=input_transformation)
                else:
                    data['S_0'] = S_0
                    data['input_data'] = input_0

                # Generate a prediction using the sample as input
                pred = generate_predictions(NN=item if NN is not None else None,
                                            dir=item if dir is not None else None,
                                            device=device, show_pbar=False,
                                            generate_full_T = False,
                                            apply_tanh_to_latent_space=apply_tanh_to_latent_space,
                                            **data
                )

                # Add the offset to the stock sample
                pred['S_pred'] += torch.from_numpy(get_stock_offsets(
                    stock_predictions=xr.DataArray(
                        pred['S_pred'].cpu(), dims=['Year', 'Origin ISO', 'Destination ISO'],
                        coords=dict((key, gamma.coords['Destination ISO'].data if key != 'Year' else gamma.coords['Year'].data)
                                    for key in stock_data.dims)),
                    stock_data=stock_data['Start of year estimate'],
                    weights=stock_data['Weight'],
                    gamma=gamma
                ).data).float().to(device)

                # Calculate the mean only from the central estimate
                # Calculate the variance using Welford's algorithm
                for key in samples:
                    x = pred[key]
                    if i == 0:
                        samples[key][0] += x / len(outer)
                    delta = x - samples[key][1]
                    samples[key][1] += delta / (count + 1)
                    samples[key][2] += delta * (x - samples[key][1])
                count += 1
                pbar.update(1)

    # Calculate standard deviation
    for key in samples:
        samples[key][2] = torch.sqrt(samples[key][2] / count)
        samples[key] = samples[key][[0, 2]]

    # Generate full flow table
    if generate_full_T:
        samples['T_pred'] = expand_T(samples['T_pred'], data, return_xarray=False)

    _keys = list(samples.keys())
    if not generate_full_T:
        _keys.remove('T_pred')
    means = convert_tensor_predictions_to_xarray(
        **dict((k, samples[k][0].cpu()) for k in _keys), years=gamma.coords['Year'].data[:-1],
        countries=gamma.coords['Destination ISO'].data
    )
    std = convert_tensor_predictions_to_xarray(
        **dict((k, samples[k][1].cpu()) for k in _keys), years=gamma.coords['Year'].data[:-1],
        countries=gamma.coords['Destination ISO'].data
    )
    ensemble_predictions = dict(
        (k, xr.Dataset(dict(mean=means[k], std=std[k]))) for k in _keys
    )

    # Add stock offset again
    ensemble_predictions['S_pred']['mean'] += get_stock_offsets(
        stock_predictions=ensemble_predictions['S_pred']['mean'],
        stock_data=stock_data['Start of year estimate'],
        weights=stock_data['Weight'], gamma=gamma)

    if not generate_full_T:
        return ensemble_predictions, samples['T_pred'].cpu()
    else:
        return ensemble_predictions, None