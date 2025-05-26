Data used as input, training targets, and validation
---
This folder contains all the data used to train, evaluate, and test the neural network.
It is stored thematically in different folders, and most folders again contains its own `README` file to further
explain the specific sources and imputation methods. All data is given *both* as a `.csv` file and a `.nc` file, and 
follows the ISO3-naming convention outlined in the main README.

> [!NOTE] 
> This is a reminder that all data is stored in this repository using git LFS (large file
> storage); if you wish to clone the repository with the data, you should follow the instructions
> from the main README. You can still download the files manually from the webpage.

## Training_data
This folder contains all the tensors used to train the neural network. All data is given as a PyTorch
tensor (`.pt`) and can be loaded using `torch.load()`. The folder contains targets, weights, masks, input covariates (scaled
and unscaled), and the edge indices of each input. See the folder README for further details.

## Net migration (`Net_migration`)
This folder contains net migration data, sourced from national statistical offices, together with a list of sources 
and the UN WPP net migration figures.

## GDP indicators (`GDP_data`)
This folder contains data on GDP/capita, GDP growth, nominal GDP, and other GDP-related indicators for all countries and
years included in the training period.

## Gravity covariates (`Gravity_datasets`)

## Demographic indictaors (`UN_WPP_data`)

## Migrant stocks (`UN_stock_data`)

## Refugee figures (`UNHCR_data`)
Total number of refugees, asylum-seekers, and other people in need of international protection, taken from the 
[UNHCR dataset](https://www.unhcr.org/refugee-statistics/download).

## Conflict deaths (`UCDP_data`)
This folder contains data on deaths in conflict provided by
[UCDP Georeferenced Event](https://ucdp.uu.se/downloads/index.html#ged_global) dataset.
NaN values are filled with 0.

## Bilateral flows (`Flow_data`)