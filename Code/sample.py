import argparse
import os
from ruamel.yaml import YAML
import Code.utils as utils
from Data import stock_data, gamma

# Load the config
yaml = YAML(typ='safe')
from pathlib import Path
config_path = Path(__file__).parent / "cfg.yaml"
with open(config_path, "r") as file:
    cfg = yaml.load(file)

parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', help='Number of samples')
parser.add_argument('--device', help='Device to use')
parser.add_argument('--full_T', action="store_true", help='Generate the full flow table')
args = parser.parse_args()

# Number of samples
if args.n_samples:
    N_samples = int(args.n_samples)
else:
    N_samples = 10

# Device to use (defaults to the config value)
if args.device:
    cfg["device"] = args.device

if args.full_T:
    generate_full_T = True
else:
    generate_full_T = False

BASE_PATH = cfg["BASE_PATH"]
device = cfg["device"]

# Load the data
data = utils.load_training_data(os.path.expanduser(os.path.join(BASE_PATH, cfg["Data_loading"]["data_path"])), cfg, device=cfg['device'])

# Generate the ensemble
dir_list = [os.path.expanduser(os.path.join(BASE_PATH, "Trained_networks_new", dir)) for dir in os.listdir(os.path.expanduser(os.path.join(BASE_PATH, "Trained_networks_new"))) if not dir.startswith('.')]
ensemble_predictions, T_pred = utils.get_samples(
    dir=dir_list,
    data=data,
    device=cfg['device'],
    stock_data=stock_data,
    gamma=gamma,
    n_samples=N_samples,
    generate_full_T=generate_full_T,
)

for key, item in ensemble_predictions.items():
    item.to_netcdf(f"Estimates/{key}_new.nc")
if T_pred is not None:
    import torch
    torch.save(T_pred.cpu(), f"Estimates/T_pred.pt")