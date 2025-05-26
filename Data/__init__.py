import pandas as pd
import os

# We make relevant datasets contained in this folder available across the project, for easier integration and plotting.
# The base directory points to the folder containing this file:
base_dir = os.path.dirname(__file__)

# Make the ISO code lookup table available across the project. Also group countries by region.
lookup_table = pd.read_csv(os.path.join(base_dir, "Iso_code_lookup.csv"))
countries_by_region = pd.read_csv(os.path.join(base_dir, "countries_by_region.csv"))
countries_by_region = countries_by_region.groupby('Region').agg(list).to_dict()['Country ISO']

# Make the world shapefile available for plotting
import geopandas as gpd
world = gpd.read_file(os.path.join(base_dir, "world_shapefile/ne_50m_admin_0_countries.shp"))
world.to_crs("EPSG:3857", inplace=True) # Mercator projection by default

# Make the coordinates available. These are regular latitude/longitude coordinates
coordinates = pd.read_csv(os.path.join(base_dir, "Coordinates.csv"))[['Alpha-3 code', 'Latitude', 'Longitude']].set_index('Alpha-3 code').to_dict()
coordinates = dict((k, (coordinates['Latitude'][k], coordinates['Longitude'][k])) for k in coordinates['Latitude'].keys())

# Load the various comparison datasets into a dictionary for easy plotting, assigning consistent colours
# and markers for each.
from Code.plot_style import colors
import xarray as xr
flow_dsets = {
    "Quantmig": dict(
        data=xr.open_dataset(os.path.join(base_dir, "Flow_data/QuantMig_data/Quantmig_flows.nc")),
        central='flow_50%', lower='flow_2.5%', upper='flow_97.5%',
        primary_color=colors['c_orange'], secondary_color=colors['c_yellow']
    ),
    "Statistics Sweden": dict(
        data=xr.open_dataarray(os.path.join(base_dir, "Flow_data/National_Statistics/SWE_flows.nc")),
        primary_color=colors['c_red'], secondary_color=colors['c_pink'], marker='v', s=10,
    ),
    "Statistics Finland": dict(
        data=xr.open_dataarray(os.path.join(base_dir, "Flow_data/National_Statistics/FIN_flows.nc")),
        primary_color=colors['c_lightblue'], secondary_color=colors['c_darkblue'], marker='s', s=10
    ),
    "StatNZ": dict(
        data=xr.open_dataarray(os.path.join(base_dir, "Flow_data/National_Statistics/NZL_flows.nc")),
        primary_color=colors['c_darkgreen'], secondary_color=colors['c_lightgreen'], marker='^', s=10
    ),
    "Facebook": dict(
        data=xr.open_dataarray(os.path.join(base_dir, "Flow_data/Facebook/facebook_flows.nc")).where(lambda x: x>=25),
        primary_color=colors['c_purple'], secondary_color='#E1C6EC', ec=colors['c_darkgrey'], marker='o', s=10, lw=0.5
    )
}

# UN WPP data
population = 1e3 * xr.open_dataset(os.path.join(base_dir, "UN_WPP_data/UN_WPP_data.nc"))['Total Population, as of 1 January (thousands)']
death_rate = 1e-3 * xr.open_dataset(os.path.join(base_dir, "UN_WPP_data/UN_WPP_data.nc"))['Crude Death Rate (deaths per 1,000 population)']
WPP_net_migration = 1e3 * xr.open_dataset(os.path.join(base_dir, "UN_WPP_data/UN_WPP_data.nc"))['Net Number of Migrants (thousands)']
NatStat_net_migration = xr.open_dataarray(os.path.join(base_dir, "Net_migration/National_statistics.nc"))
WPP_data = xr.open_dataset(os.path.join(base_dir, "UN_WPP_data/UN_WPP_data.nc"))

# Calculate the fraction of population alive at start of 1990 in each destination country still alive at start of year
import numpy as np
gamma = (1-death_rate.sel({"Year": range(1989, 2024)})).assign_coords({"Year": np.arange(1990, 2025, 1)}).rename({
    "Country ISO": "Destination ISO"})
gamma.loc[{"Year": 1990}] = 1.0
gamma = gamma.cumprod('Year')

# Stock data
stock_data = xr.load_dataset(os.path.join(base_dir, "UN_stock_data/stock_data.nc"))