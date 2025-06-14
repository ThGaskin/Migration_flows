from typing import Any, List, Sequence, Union

import torch
from torch import nn


def random_tensor(
    cfg: dict | list, *, size: tuple = None, device: str, **__
) -> torch.Tensor:
    """Generates a multi-dimensional random tensor. Each entry can be initialised separately, or a common
    initialisation configuration is used for each entry. For instance, the configuration

    .. code-block::

        cfg:
            distribution: uniform
            parameters:
                lower: 0
                upper: 1

    together with `size: (2, 2)` will initialise a 2x2 matrix with entries drawn from a uniform distribution on
    [0, 1]. The configuration

    .. code-block::

        cfg:
            - distribution: uniform
              parameters:
                 lower: 0
                 upper: 1
            - distribution: normal
              parameters:
                mean: 0.5
                std: 0.1

    will initialise a (2, 1) tensor with entries drawn from different distributions.

    :param cfg: the configuration entry containing the initialisation data
    :param size (optional): the size of the tensor, in case the configuration is not a list
    :param device: the device onto which to load the data
    :param __: additional kwargs (ignored)
    :return: the tensor of random variables
    """

    def _random_tensor_1d(
        *, distribution: str, parameters: dict, s: tuple = (1,), **__
    ) -> torch.Tensor:
        """Generates a random tensor according to a distribution.

        :param distribution: the type of distribution. Can be 'uniform' or 'normal'.
        :param parameters: the parameters relevant to the respective distribution
        :param s: the size of the random tensor
        """

        # Uniform distribution in an interval
        if distribution == "uniform":
            l, u = parameters.get("lower"), parameters.get("upper")
            if l > u:
                raise ValueError(
                    f"Upper bound must be greater or equal to lower bound; got {l} and {u}!"
                )

            return torch.tensor((u - l), dtype=torch.float) * torch.rand(
                s, dtype=torch.float
            ) + torch.tensor(l, dtype=torch.float)

        # Normal distribution
        elif distribution == "normal":
            return torch.normal(
                parameters.get("mean"),
                parameters.get("std"),
                size=s,
                dtype=torch.float,
            )

        else:
            raise ValueError(f"Unrecognised distribution type {distribution}!")

    if isinstance(cfg, list):
        return torch.tensor([_random_tensor_1d(**entry) for entry in cfg]).to(device)
    else:
        return _random_tensor_1d(**cfg, s=size).to(device)


# ----------------------------------------------------------------------------------------------------------------------
# -- NN utility functions ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def sigmoid(alpha=torch.tensor(1.0), beta=torch.tensor(1.0), gamma=torch.tensor(0.0), delta=torch.tensor(0.0)):
    """Extends the torch.nn.sigmoid activation function by allowing for a scale and slope parameter."""

    return lambda x: alpha * torch.sigmoid(beta * (x-delta)) + gamma


# Pytorch activation functions.
# Pairs of activation functions and whether they are part of the torch.nn module, in which case they must be called
# via func(*args, **kwargs)(x).


ACTIVATION_FUNCS = {
    "abs": [torch.abs, False],
    "celu": [torch.nn.CELU, True],
    "cos": [torch.cos, False],
    "cosine": [torch.cos, False],
    "elu": [torch.nn.ELU, True],
    "gelu": [torch.nn.GELU, True],
    "hardshrink": [torch.nn.Hardshrink, True],
    "hardsigmoid": [torch.nn.Hardsigmoid, True],
    "hardswish": [torch.nn.Hardswish, True],
    "hardtanh": [torch.nn.Hardtanh, True],
    "leakyrelu": [torch.nn.LeakyReLU, True],
    "linear": [None, False],
    "logsigmoid": [torch.nn.LogSigmoid, True],
    "mish": [torch.nn.Mish, True],
    "prelu": [torch.nn.PReLU, True],
    "relu": [torch.nn.ReLU, True],
    "rrelu": [torch.nn.RReLU, True],
    "selu": [torch.nn.SELU, True],
    "sigmoid": [sigmoid, True],
    "silu": [torch.nn.SiLU, True],
    "sin": [torch.sin, False],
    "sine": [torch.sin, False],
    "softplus": [torch.nn.Softplus, True],
    "softshrink": [torch.nn.Softshrink, True],
    "swish": [torch.nn.SiLU, True],
    "tanh": [torch.nn.Tanh, True],
    "tanhshrink": [torch.nn.Tanhshrink, True],
    "threshold": [torch.nn.Threshold, True],
}


def get_architecture(
    input_size: int, output_size: int, n_layers: int, cfg: dict
) -> List[int]:
    # Apply default to all hidden layers
    _nodes = [cfg.get("default")] * n_layers

    # Update layer-specific settings
    _layer_specific = cfg.get("layer_specific", {})
    for layer_id, layer_size in _layer_specific.items():
        _nodes[layer_id] = layer_size

    return [input_size] + _nodes + [output_size]


def get_activation_funcs(n_layers: int, cfg: dict) -> List[callable]:
    """Extracts the activation functions from the config. The config is a dictionary containing the
    default activation function, and a layer-specific entry detailing exceptions from the default. 'None' entries
    are interpreted as linear layers.

    .. Example:
        activation_funcs:
          default: relu
          layer_specific:
            0: ~
            2: tanh
            3:
              name: HardTanh
              args:
                - -2  # min_value
                - +2  # max_value
    """

    def _single_layer_func(layer_cfg: Union[str, dict, callable]) -> callable:
        """Return the activation function from an entry for a single layer"""

        # Entry is a single string
        if isinstance(layer_cfg, str):
            _f = ACTIVATION_FUNCS[layer_cfg.lower()]
            if _f[1]:
                return _f[0]()
            else:
                return _f[0]

        # Entry is a dictionary containing args and kwargs
        elif isinstance(layer_cfg, dict):
            _f = ACTIVATION_FUNCS[layer_cfg.get("name").lower()]
            if _f[1]:
                return _f[0](*layer_cfg.get("args", ()), **layer_cfg.get("kwargs", {}))
            else:
                return _f[0]

        # Callable passed
        elif callable(layer_cfg):
            return layer_cfg

        # Empty cfg: linear activation function
        elif layer_cfg is None:
            _f = ACTIVATION_FUNCS["linear"][0]

        else:
            raise ValueError(f"Unrecognized activation function {cfg}!")

    # Use default activation function on all layers
    _funcs = [_single_layer_func(cfg.get("default"))] * (n_layers + 1)

    # Change activation functions on specified layers
    _layer_specific = cfg.get("layer_specific", {})
    for layer_id, layer_cfg in _layer_specific.items():
        _funcs[layer_id] = _single_layer_func(layer_cfg)

    return _funcs


def get_bias(n_layers: int, cfg: dict) -> List[Any]:
    """Extracts the bias initialisation settings from the config. The config is a dictionary containing the
    default, and a layer-specific entry detailing exceptions from the default. 'None' entries
    are interpreted as unbiased layers.

    .. Example:
        biases:
          default: ~
          layer_specific:
            0: [-1, 1]
            3: [2, 3]
    """

    # Use the default value on all layers
    biases = [cfg.get("default")] * (n_layers + 1)

    # Amend bias on specified layers
    _layer_specific = cfg.get("layer_specific", {})
    for layer_id, layer_bias in _layer_specific.items():
        biases[layer_id] = layer_bias

    return biases


# -----------------------------------------------------------------------------
# -- Neural net class ---------------------------------------------------------
# -----------------------------------------------------------------------------


class NeuralNet(nn.Module):
    OPTIMIZERS = {
        "Adagrad": torch.optim.Adagrad,
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SparseAdam": torch.optim.SparseAdam,
        "Adamax": torch.optim.Adamax,
        "ASGD": torch.optim.ASGD,
        "LBFGS": torch.optim.LBFGS,
        "NAdam": torch.optim.NAdam,
        "RAdam": torch.optim.RAdam,
        "RMSprop": torch.optim.RMSprop,
        "Rprop": torch.optim.Rprop,
        "SGD": torch.optim.SGD
    }

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        num_layers: int,
        nodes_per_layer: dict,
        activation_funcs: dict,
        biases: dict,
        prior: Union[list, dict] = None,
        prior_max_iter: int = 500,
        prior_tol: float = 1e-5,
        optimizer: str = "Adam",
        optimizer_kwargs: dict = None,
        device: str = 'cpu',
        **__,
    ):
        """

        :param input_size: the number of input values
        :param output_size: the number of output values
        :param num_layers: the number of hidden layers
        :param nodes_per_layer: a dictionary specifying the number of nodes per layer
        :param activation_funcs: a dictionary specifying the activation functions to use
        :param biases: a dictionary containing the initialisation parameters for the bias
        :param prior (optional): initial prior distribution of the parameters. If given, the neural net will
            initially output a random value within that distribution.
        :param prior_tol (optional): the tolerance with which the prior distribution should be met
        :param prior_max_iter (optional): maximum number of training iterations to hit the prior target
        :param optimizer: the name of the optimizer to use. Default is the torch.optim.Adam optimizer.
        :param optimizer_kwargs: passed to the optimizer
        :param __: Additional model parameters (ignored)
        """

        super().__init__()
        self.flatten = nn.Flatten()

        self.input_dim = input_size
        self.output_dim = output_size
        self.hidden_dim = num_layers

        # Get architecture, activation functions, and layer bias
        self.architecture = get_architecture(
            input_size, output_size, num_layers, nodes_per_layer
        )
        self.activation_funcs = get_activation_funcs(num_layers, activation_funcs)
        self.bias = get_bias(num_layers, biases)

        # Add the neural net layers
        self.layers = nn.ModuleList()
        for i in range(len(self.architecture) - 1):
            layer = nn.Linear(
                self.architecture[i],
                self.architecture[i + 1],
                bias=self.bias[i] is not None,
            )

            # Initialise the biases of the layers with a uniform distribution
            if self.bias[i] is not None:
                # Use the pytorch default if indicated
                if self.bias[i] == "default":
                    torch.nn.init.uniform_(layer.bias)
                # Initialise the bias on explicitly provided intervals
                else:
                    torch.nn.init.uniform_(layer.bias, self.bias[i][0], self.bias[i][1])

            self.layers.append(layer)

        # Get the optimizer
        self.optimizer = self.OPTIMIZERS[optimizer](
            self.parameters(), **(optimizer_kwargs if optimizer_kwargs is not None else {})
        )

        # Get the initial distribution and initialise
        self.prior_distribution = prior
        self.initialise_to_prior(tol=prior_tol, max_iter=prior_max_iter, device=device)

    def initialise_to_prior(self, *, tol: float = 1e-5, max_iter: int = 500, device: str) -> None:
        """Initialises the neural net to output values following a prior distribution. The random tensor is drawn
        following a prior distribution and the neural network trained to output that value. Training is performed
        until the neural network output matches the drawn value (which typically only takes a few seconds), or until
        a maximum iteration count is reached.

        :param tol: the target error on the neural net initial output and drawn value.
        :param max_iter: maximum number of training steps to perform in the while loop
        """

        # If not initial distribution is given, nothing happens
        if self.prior_distribution is None:
            return

        # Draw a target tensor following the given prior distribution
        target = random_tensor(self.prior_distribution, size=(self.output_dim,), device=device)

        # Generate a prediction and train the net to output the given target
        prediction = self.forward(torch.rand(self.input_dim))
        iter = 0

        # Use a separate optimizer for the training
        optim = torch.optim.Adam(self.parameters(), lr=0.002)
        while torch.norm(prediction - target) > tol and iter < max_iter:
            prediction = self.forward(torch.rand(self.input_dim))
            loss = torch.nn.functional.mse_loss(target, prediction, reduction="sum")
            loss.backward()
            optim.step()
            optim.zero_grad()
            iter += 1

    # ... Evaluation functions .........................................................................................

    # The model forward pass
    def forward(self, x):
        for i in range(len(self.layers)):
            if self.activation_funcs[i] is None:
                x = self.layers[i](x)
            else:
                x = self.activation_funcs[i](self.layers[i](x))
        return x
