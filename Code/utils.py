import copy
import pickle
import os
import numpy as np
import torch
import tqdm
import xarray as xr
from ruamel.yaml import YAML

yaml = YAML(typ='safe')

from .neural_net import NeuralNet

""" Utility functions used to generate predictions using a neural network """

def yeo_johnson_transform(data: torch.Tensor | np.ndarray,
                          lmbda: torch.Tensor | float,
                          *,
                          flip_negative_values: bool = True,
                          mean: torch.Tensor | float | None = None,
                          std: torch.Tensor | float | None = None,
                          standardize: bool = False
) -> torch.Tensor | np.ndarray:

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
        res = torch.nan * torch.zeros_like(data)

        # Handle positive values
        if lmbda != 0:
            res[mask_pos] = (torch.pow(data[mask_pos] + 1, lmbda) - 1) / lmbda
        else:
            res[mask_pos] = torch.log(data[mask_pos] + 1)

        # Handle negative values
        if flip_negative_values:
            res[mask_neg] = -yeo_johnson_transform(-data[mask_neg], lmbda, flip_negative_values=False)
        else:
            if lmbda != 2:
                res[mask_neg] = - (torch.pow(-data[mask_neg] + 1, 2 - lmbda) - 1) / (2 - lmbda)
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

def inv_yeo_johnson(data: torch.Tensor | np.ndarray,
                    lmbda: torch.Tensor | float,
                    mean: torch.Tensor | float = 0.0,
                    std: torch.Tensor | float = 1.0, *, flip_neg_values=True
                    ) -> torch.Tensor | np.ndarray:
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


def generate_predictions(NN: NeuralNet, *,
                         edge_indices: torch.Tensor,
                         input_data: torch.Tensor,
                         S_0: torch.Tensor,
                         show_pbar: bool = True,
                         device: str = 'cpu',
                         transformation_parameters: dict,
                         scaling_factor: torch.Tensor | float = 1000.,
                         death_rate: torch.Tensor,
                         total_births: torch.Tensor,
                         **__
                         ) -> dict:
    """Generates predictions using a neural network

    :param edge_indices: list of edge indices
    :param input_data: torch.Tensor of input data
    :param S_0: initial stocks
    :param total_births: total number of births in each year, by country
    :param death_rate: death rate for each country
    :param transformation_parameters: dictionary of Yeo-Johnson transformation values
    :param device: training device
    :param show_pbar: show the progress bar during evaluation
    :param __: other parameters (ignored)
    :return: dictionary containing the predictions
    """

    # Edge indicies
    idx_i, idx_j, idx_k = edge_indices

    # Initialise the hidden state
    h_t = torch.zeros((idx_i.shape[0], NN.output_dim - 1), device=device)

    Y = input_data.shape[0]
    N = S_0.shape[0]
    T_pred = torch.zeros((Y, N, N, N), device=device)
    stock_predictions = [S_0]

    for y in tqdm.trange(Y) if show_pbar else range(Y):

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

        # Append latent state to neural network input
        if NN.output_dim > 1:
            _input_data = torch.cat([_input_data, h_t], dim=1)
            res = NN(_input_data).detach()
            log_flow, h_t = res[:, 0], res[:, 1:]
        else:
            log_flow = NN(_input_data).detach().flatten()
        T_pred[y, idx_i, idx_j, idx_k] = scaling_factor * torch.exp(log_flow)

        # Update the stock predictions
        stock_predictions.append(torch.maximum(
            torch.tensor(0.0),
            (1 - death_rate[y]).reshape(1, N) * stock_predictions[-1] + torch.diag(total_births[y]) + T_pred[y].sum(
                dim=1) - T_pred[y].sum(dim=2))
        )

    # Combine predictions and return a dictionary
    stock_predictions = torch.stack(stock_predictions)
    flow_predictions = T_pred.sum(dim=1)
    net_migration_predictions = flow_predictions.sum(dim=1) - flow_predictions.sum(dim=2)

    return dict(T_pred=T_pred, S_pred=stock_predictions, mu_pred=net_migration_predictions,
                F_pred=flow_predictions, NN=NN)


def get_predictions(dir: str, *,
                    edge_indices: torch.Tensor,
                    input_data: torch.Tensor,
                    S_0: torch.Tensor,
                    total_births: torch.Tensor,
                    death_rate: torch.Tensor,
                    transformation_parameters: dict,
                    device: str = 'cpu',
                    show_pbar: bool = True,
                    **__
                    ) -> dict:
    """ Loads a neural network and uses it to generate predictions.

    :param dir: directory containing the trained neural network
    :param edge_indices: list of edge indices
    :param input_data: torch.Tensor of input data
    :param S_0: initial stocks
    :param total_births: total number of births in each year, by country
    :param death_rate: death rate for each country
    :param transformation_parameters: dictionary of Yeo-Johnson transformation values
    :param device: training device
    :param show_pbar: show the progress bar during evaluation
    :param __: other parameters (ignored)
    :return: dictionary containing the predictions
    """
    with open(f"{dir}/cfg.yaml", "r") as file:
        nn_cfg = yaml.load(file)

    # Set up the neural network
    NN = NeuralNet(
        input_size=input_data.shape[2] + 2 + nn_cfg['NeuralNet'].get('latent_space_dim', 0),
        output_size=1 + nn_cfg['NeuralNet'].get('latent_space_dim', 0),
        **nn_cfg["NeuralNet"]
    ).to(device)
    NN.load_state_dict(torch.load(f"{dir}/model_trained.pt", weights_only=True, map_location=torch.device(device)))
    NN.eval()

    return generate_predictions(NN, input_data=input_data, edge_indices=edge_indices, S_0=S_0,
                                total_births=total_births,
                                death_rate=death_rate, transformation_parameters=transformation_parameters,
                                device=device,
                                show_pbar=show_pbar, scaling_factor=nn_cfg['Data_loading']['data_rescale'])


def convert_tensor_predictions_to_xarray(*,
                                         T_pred: torch.Tensor,
                                         S_pred: torch.Tensor,
                                         mu_pred: torch.Tensor,
                                         F_pred: torch.Tensor,
                                         years: np.ndarray | None = None,
                                         countries: np.ndarray | None = None,
                                         **__
                                         ) -> dict:
    """ Converts torch.Tensors into xarray items for easier indexing and plotting.

    :param T_pred: torch.Tensor of total flow predictions, of shape (Y, N, N, N)
    :param S_pred: torch.Tensor of migrant stocks, of shape (Y, N, N)
    :param mu_pred: torch.Tensor of net migration, of shape (Y, N)
    :param F_pred: torch.Tensor of flows, of shape (Y, N, N)
    :param years: (optional) coordinates to use for the year dimension. If None, an array of indices is created.
    :param countries: (optional) coordinates to use for the countries. If None, an array of indices is created.
    :param __: other kwargs (ignored)
    :return: dictionary of converted xr.DataArray objects
    """

    if years is None:
        years = np.arange(T_pred.shape[0])
    if countries is None:
        countries = np.arange(T_pred.shape[1])

    return dict(T_pred=xr.DataArray(
        data=T_pred.cpu(),
        dims=["Year", "Birth ISO", "Origin ISO", "Destination ISO"],
        coords={"Year": years, "Birth ISO": countries, "Origin ISO": countries, "Destination ISO": countries},
        name="Flow table"
    ), S_pred=xr.DataArray(
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
    )
    )


def generate_samples(data: dict, predictions: dict, *, cfg, n_samples: int, transformation_parameters: dict,
                     stock_std: float | torch.Tensor = 0.1, device: str = 'cpu', show_pbar: bool = False) -> xr.Dataset:
    """ Generate samples

    :param data: dictionary containing the training data
    :param predictions: dictionary containing the neural network
    :param cfg: configuration used to point to the covariates
    :param n_samples: number of samples to draw
    :param transformation_parameters: transformation parameters containing the standard deviation for each continuous
        covariate
    :param stock_std: standard deviation to use to sample the stocks
    :param device: device to use
    :param show_pbar: whether to show the progress bar during sampling
    :return: xr.Dataset of sample mean and sums of squares
    """

    # Prepare a dictionary for the samples
    samples = dict(
        (k.replace('pred', 'sample'), torch.zeros(predictions[k].shape).unsqueeze(0).to(device).float()) for k in
        ['T_pred', 'F_pred', 'S_pred', 'mu_pred'])
    for k, v in samples.items():
        dim = [1] * v.dim()
        dim[0] = 2
        samples[k] = samples[k].repeat(dim)

    # Get the original initial stock to sample
    S_0 = data['S_0']

    # Draw samples
    for _ in tqdm.trange(n_samples) if show_pbar else range(n_samples):

        data['input_data'] = build_input(cfg, data['edge_indices'], data['Y'], device=device,
                                               transformation_params=transformation_parameters)

        # Sample the initial stock
        data['S_0'] = torch.maximum(torch.tensor(0.0), torch.normal(S_0, stock_std * S_0))

        # Generate predictions
        sample_predictions = generate_predictions(predictions['NN'],
                                                  scaling_factor=cfg['Data_loading']['data_rescale'],
                                                  device=device, show_pbar=False, **data)

        for k in samples.keys():
            samples[k][0, :] += 1 / n_samples * sample_predictions[k.replace('sample', 'pred')]
            samples[k][1, :] += 1 / n_samples * sample_predictions[k.replace('sample', 'pred')] ** 2

    # Convert to an xarray Dataset
    for key in samples.keys():
        samples[key] = xr.Dataset({
            'mean': (predictions[key.replace('sample', 'pred')].dims, samples[key][0].cpu()),
            'std': (predictions[key.replace('sample', 'pred')].dims, torch.sqrt(samples[key][1] - samples[key][0]**2).cpu())
        }, coords=predictions[key.replace('sample', 'pred')].coords)

    return samples


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

    # Iteratively adjust unitl no more stocks are below 0
    negative_stocks = (stock_predictions + gamma * offset).where(lambda x: x < 0, 0)
    while negative_stocks.sum() != 0:
        offset += xr.where((stock_predictions + gamma * offset) == negative_stocks.min('Year'),
                           - (stock_predictions + gamma * offset) / gamma, 0).sum('Year')
        negative_stocks = (stock_predictions + gamma * offset).where(lambda x: x < 0, 0)

    offset = gamma * offset

    return offset

def get_elasticities(data: dict, predictions: dict, cfg: dict, *, n_edges: int = 20,
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


def generate_ensemble_predictions(dirs: list, data: dict, *, device: str = 'cpu', show_pbar: bool = True,
                                  stock_data: xr.Dataset, gamma: xr.DataArray,
                                  n_samples: int = 0) -> dict:

    """ Generate ensemble predictions from a family of trained networks.
    Mean estimates and uncertainties are calculated from the ensemble.

    :param dirs: list of directories from which to source the estimates
    :param data: data dictionary containing the information needed to run the neural network
    :param device: device to use
    :param show_pbar: show a progress bar
    :param stock_data: xr.Dataset containing the stock data and weights
    :param gamma: cumulative death rates used to calculate the stock offsets
    :param n_samples: number of samples to draw for the stock data
    :return: dictionary of estimates
    """
    samples = dict(
        T_pred=torch.zeros(2, data['Y'], data['N'], data['N'], data['N']),
        F_pred=torch.zeros(2, data['Y'], data['N'], data['N']),
        S_pred=torch.zeros(2, data['Y'] + 1, data['N'], data['N']),
        mu_pred=torch.zeros(2, data['Y'], data['N']),
    )

    S_0 = data['S_0']
    S_std = torch.from_numpy(stock_data['Error'].isel({"Year": 0}).fillna(0).data).float().to(device)

    for dir in tqdm.tqdm(dirs) if show_pbar else dirs:

        for i in range(n_samples + 1):

            # Sample the initial stock
            data['S_0'] = torch.maximum(torch.tensor(0.0), torch.normal(S_0, S_std if i > 0 else 0.0))
            pred = get_predictions(
                dir, device=device, show_pbar=False, **data
            )

            # Add the offset to the stock sample
            pred['S_pred'] += torch.from_numpy(get_stock_offsets(
                stock_predictions=xr.DataArray(
                    pred['S_pred'].cpu(), dims=['Year', 'Origin ISO', 'Destination ISO'],
                    coords=dict((key, gamma.coords['Destination ISO'].data if key != 'Year' else gamma.coords['Year'].data)
                                for key in stock_data.dims)).sel({"Year": stock_data.coords['Year']}),
                stock_data=stock_data['Start of year estimate'],
                weights=stock_data['Weight'],
                gamma=gamma
            ).data).float().to(device)

            # Calculate the mean only from the central estimate
            for key in samples:
                if i ==0:
                    samples[key][0] += pred[key].cpu() / len(dirs)
                samples[key][1] += pred[key].cpu()**2 / (len(dirs)*(n_samples + 1))

    for key in samples:
        samples[key][1] = torch.sqrt(samples[key][1] - samples[key][0]**2)

    means = convert_tensor_predictions_to_xarray(
        **dict((k, samples[k][0]) for k in samples.keys()), years=gamma.coords['Year'].data[:-1],
        countries=gamma.coords['Destination ISO'].data
    )
    std = convert_tensor_predictions_to_xarray(
        **dict((k, samples[k][1]) for k in samples.keys()), years=gamma.coords['Year'].data[:-1],
        countries=gamma.coords['Destination ISO'].data
    )
    ensemble_predictions = dict(
        (k, xr.Dataset(dict(mean=means[k], std=std[k]))) for k in samples.keys()
    )

    # Add stock offset again
    ensemble_predictions['S_pred']['mean'] += get_stock_offsets(
        stock_predictions=ensemble_predictions['S_pred']['mean'].sel({"Year": stock_data.coords['Year'].data}),
        stock_data=stock_data['Start of year estimate'], weights=stock_data['Weight'], gamma=gamma)

    return ensemble_predictions