import cartopy
import cartopy.crs as ccrs
import cartopy.mpl.geoaxes as cgeo
import copy
import numpy as np
import pandas as pd
import re
import seaborn as sns
import tqdm
import xarray as xr

from dantro.plot.utils import ColorManager
from matplotlib.colorbar import Colorbar
from ruamel.yaml import YAML

from Code.plot_style import colors
from Data import world, lookup_table, coordinates
from shapely.geometry import Point
from matplotlib.patches import FancyArrowPatch

yaml = YAML(typ='safe')

""" Plot utilities used for evaluation and visualisation """

def iso_to_name(iso3):
    """ Convert an ISO3 code to a compactly formatted country name """
    replacements = {'Congo (the Democratic Republic of the)': 'DR Congo',
                   'United States of America (the)': 'USA',
                   'United Arab Emirates (the)': 'UAE',
                   'Palestine, State of': 'Palestine',
                   'Viet Nam': 'Vietnam',
                   'Venezuela (Bolivarian Republic of)': 'Venezuela',
                   'Saint Helena, Ascension and Tristan da Cunha': 'St Helena',
                   'Russian Federation (the)': 'Russia',
                   'United Kingdom of Great Britain and Northern Ireland (the)': 'United Kingdom',
                   'Netherlands (the)': 'Netherlands',
                   'Taiwan (Province of China)': 'Taiwan',
                   'Korea (the Republic of)': 'South Korea',
                   'Niger (the)': 'Niger',
                   "Lao People's Democratic Republic (the)": 'Lao',
                    'Syrian Arab Republic': 'Syria'
                    }

    _name = lookup_table[lookup_table['Alpha-3 code'] == iso3]['Country'].item()
    # Remove parentheses
    _name = re.sub(r'\s*\([^)]*\)', '', replacements.get(_name, _name))
    return _name



def align_five_year_flows(five_year_flows, pred_5yflow):
    """ Seperate an xr.DataArray of predicted flows into an array of five-year flows, sorted by collection
    and definition, that allows for easy comparison to other methods.

    :param five_year_flows: xr.DataArray of five-year flows, sorted by collection and definition, using other methods
    :param pred_5yflow: xr.DataArray of neural five-year-flow predictions
    :return: xr.DataArray of neural estimates
    """
    NN_ests = []
    for definition in five_year_flows.coords['Definition'].data:
        res = []
        for collection in five_year_flows.coords['Collection'].data:
            res.append(
                xr.concat([
                    xr.Dataset({

                        # Estimated flow
                        'Estimated flow': pred_5yflow.sel({"Method": "NN", "Definition": definition, "Coverage":'Total'},
                                                            drop=True),

                        # Estimated total
                        'Estimated total': pred_5yflow.sel({"Method": "NN", "Definition": definition,
                                                            "Coverage":'Total'}, drop=True).sum('Origin ISO' if _rep == 'Destination' else 'Destination ISO').expand_dims({"Origin ISO": pred_5yflow.coords['Origin ISO'].data} if _rep == 'Destination' else {"Destination ISO": pred_5yflow.coords['Destination ISO'].data}),

                        # Reported flow
                        'Reported flow': five_year_flows.sel({"Coverage": 'total', "Definition": definition, "Reporter": _rep, "Collection": collection}, drop=True)['Reported flow'].mean('Method', skipna=True),

                        # Reported total
                        'Reported total': five_year_flows.sel({"Coverage": 'total', "Definition": definition, "Reporter": _rep, "Collection": collection}, drop=True)['Reported total'].mean('Method', skipna=True)

                    }).expand_dims({"Reporter": [_rep]})
                for _rep in five_year_flows.coords["Reporter"].data], dim='Reporter').expand_dims({"Collection": [collection]})
            )
        NN_ests.append(xr.concat(res, dim='Collection').expand_dims({"Definition": [definition]}))
    NN_ests = xr.concat(NN_ests, dim='Definition')

    NN_ests['Absolute error'] = abs(NN_ests['Estimated flow'] - NN_ests['Reported flow'])
    NN_ests['Relative error'] = NN_ests['Absolute error'] / xr.where(NN_ests['Reported flow'] > 0, NN_ests['Reported flow'], 1)
    NN_ests = NN_ests.expand_dims({"Method": ['NN'], "Coverage": ['total']})
    NN_ests = NN_ests.sel({"Origin ISO": five_year_flows.coords["Origin ISO"].data, "Destination ISO": five_year_flows.coords['Destination ISO'].data}).transpose('Reporter', 'Collection', 'Method', 'Coverage', 'Definition', 'Year0', 'Origin ISO', 'Destination ISO')

    return NN_ests

def comparison_stats(F, population) -> xr.DataArray:
    """ Calculate various comparison statistics on a xr.Dataset of estimated and reported flows.

    :param F: a xr.Dataset of 'Estimated flows' and 'Reported flows', for various collections and
        methods.
    :param population: xr.DataArray of total population for each country
    :return: an xr.DataArray of the calculated statistics, indexed by 'Method', 'Collection', and 'Metric'
    """
    # Calculate stats across all dimensions except the Method and the Collection
    dims = list(F.dims)
    dims.remove('Method')
    dims.remove('Collection')

    # Pearson R
    corr = xr.corr(F['Estimated flow'], F['Reported flow'], dim=dims).expand_dims({"Metric": ['Count']})

    # Pearson R on log flows
    log_corr = xr.corr(np.log(1+F['Estimated flow']), np.log(1+F['Reported flow']), dim=dims).expand_dims({"Metric": ['Log Count']})

    # Pearson R on flow proportions
    prop = xr.corr(F['Estimated flow']/F['Estimated total'], F['Reported flow']/F['Reported total'], dim=dims).expand_dims({"Metric": ["Proportion"]})

    # Pearson R on migration rate
    mig_rate = xr.corr(F['Estimated flow'] / population.rename({"Year": "Year0", "Country ISO": "Origin ISO"}),
                       F['Reported flow'] / population.rename({"Year": "Year0", "Country ISO": "Origin ISO"}),
                       dim=dims).expand_dims({"Metric": ["Migration rate"]})
    return xr.concat([corr, log_corr, prop, mig_rate], dim='Metric')

def plot_grid_to_ax(
        data: np.ndarray, ax, *, cm: ColorManager
):
    """ Plots a single grid to an axis. Labels are added to each cell and the highest value is highlighted.

    :param data: Panel data of evaluation metrics.
    :param ax: axis to use
    :param cm: ColorManager to use
    :return: image data
    """

    # Plot
    im = ax.imshow(data, cmap=cm.cmap, norm=cm.norm, aspect='auto')

    # Display values inside cells, making max values bold
    max_values = data.max(axis=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = f"{data[i, j]:.2f}"
            if data[i, j] == max_values[j]:  # If this is the max in the column
                ax.text(j, i, value, ha='center', va='center', weight='bold')
            else:
                ax.text(j, i, value, ha='center', va='center')
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set(visible=True, color=colors['c_darkgrey'], lw=0.5)
    ax.grid(False)

    return im


def plot_eval_metrics(stats, fig, axs, *, cm: ColorManager):
    """ Plots evaluation metrics from a xr.DataArray of dimensions (Method, Collection, Metric) to a facet_grid.

    :param stats: xr.DataArray of statistical metrics
    :param fig: figure
    :param axs: axes
    :param cm: ColorManager to use for the heatmap
    """

    # Names for the different methods, used as titles
    method_names = {
        'da_pb_closed': 'Demographic Account \n Pseudo Bayesian Closed',
        'mig_rate': 'Migration rate',
        'da_min_closed': 'Demographic Account \n Minimisation Closed',
        'da_min_open': 'Demographic Account \n Minimisation Open',
        'sd_drop_neg': 'Stock Difference \n Drop Negative',
        'sd_rev_neg': 'Stock Difference \n Reverse Negative'
    }

    for idx, collection in enumerate(stats.coords['Collection'].data):

        # Plot grid data to axis
        im = plot_grid_to_ax(stats.sel({"Collection": collection}).data.T, axs[idx], cm=cm)

        # Add title and tick labels
        axs[idx].set_title(collection)
        axs[idx].set_xticks(np.arange(len(stats.coords['Metric'])), stats.coords['Metric'].data, rotation=45, ha='right')

    axs[0].set_yticks(np.arange(len(stats.coords['Method'].data)), [method_names.get(name, name) for name in stats.coords['Method'].data],
                      linespacing=0.8)

    # Add colourbar
    fig.subplots_adjust(right=0.99)
    c_ax = fig.add_axes([1, 0, 0.02, 1])
    cbar = fig.colorbar(im, c_ax, label='Correlation')
    cbar.outline.set_edgecolor(colors['c_darkgrey'])
    cbar.outline.set_linewidth(0.5)
    fig.subplots_adjust(wspace=0.03)

def correlation_violin(ax, *, items, pred, mask, **plot_kwargs):
    """ Violinplot of correlations on the test and training data on various datasets.

    :param ax: axis on which to plot data
    :param items: dictionary datasets, colours, and labels
    :param pred: predicted flows
    :param mask: test mask. Its negation is the train mask
    :param plot_kwargs: passed to `sns.violinplot`
    """

    for label, item in items.items():

        # Compute correlations
        ds = item['data']
        if 'central' in item.keys():
            ds = ds[item['central']]

        corr_train = xr.corr(ds.where(~mask), pred.where(~mask), dim='Year').data.flatten()
        corr_test = xr.corr(ds.where(mask), pred.where(mask), dim='Year').data.flatten()

        # Build a DataFrame
        df = pd.DataFrame({
            'group': [label] * (len(corr_train) + len(corr_test)),
            'subgroup': ['Train'] * len(corr_train) + ['Test'] * len(corr_test),
            'value': np.concatenate([corr_train, corr_test])
        })

        # Plot
        kwargs = dict(split=True, fill=True, inner=None, gap=0., linewidth=0, legend=False)
        kwargs.update(**plot_kwargs)
        sns.violinplot(df, x="group", y="value", hue="subgroup", ax=ax,
                       palette=dict(Train=item['primary_color'], Test=item['secondary_color']), **kwargs)

    ax.set(ylabel='', xlabel='')

def scatter_relative_errs(x: xr.DataArray, y: xr.DataArray, ax, *, cm: ColorManager, **plot_kwargs):
    """ Scatters two datasets x and y against each other, with the hue indicating the relative error

    :param x: first xr.DataArray
    :param y: second xr.DataArray
    :param ax: axis to use
    :param cm: ColorManager
    :param plot_kwargs: passed to ax.scatter
    """
    _msk = (~np.isnan(x.data.flatten())) & (~np.isnan(x.data.flatten()))
    _x_plot = x.data.flatten()[_msk]
    _y_plot = y.data.flatten()[_msk]
    ax.scatter(_x_plot, _y_plot, c=abs(_x_plot-_y_plot)/(_x_plot+1), cmap=cm.cmap, norm=cm.norm, **plot_kwargs)

def plot_flow_data(ax, items: dict, *, o: str, d: str) -> dict:
    """ Plot flow data to an axis for a given origin-destination pair

    :param ax: axis on which to plot data
    :param items: dictionary of datasets, colours, and labels
    :param o: origin ISO
    :param d: destination ISO
    :return: handles: dictionary of labels and handles to use in a legend
    """
    handles = {}
    for label, item in items.items():
        if o not in item['data'].coords['Origin ISO'] or d not in item['data'].coords['Destination ISO']:
            continue
        ds = item['data'].sel({"Origin ISO": o, "Destination ISO": d}, drop=True).dropna('Year')
        if len(ds) == 0:
                continue
        if not 'lower' in item.keys():
            handles[label] = ds.plot.scatter(ax=ax, s=item['s'], marker=item['marker'], color=item['primary_color'], lw=item.get('lw', 0), ec=item.get('ec', None))
        else:
            lower, upper = ds[item['lower']].dropna('Year'), ds[item['upper']].dropna('Year')
            ebar = ax.fill_between(
                ds.coords['Year'].data, lower, upper, alpha=0.2, color=item['primary_color'], lw=0
            )
            line = ds[item['central']].plot(ax=ax, color=item['primary_color'])
            handles[label] = (line[0], ebar)

    ax.set(ylabel='', xlabel='')
    ax.set_title(f'Flow: {iso_to_name(o)} to {iso_to_name(d)}', x=0, ha='left', weight='bold')
    return handles

## ---------------------------------------------------------------------------------------------------------------------
## Map tools
## ---------------------------------------------------------------------------------------------------------------------
def add_country_patches(world, ax, *, country_list: list = None, **kwargs) -> dict:
    """ Plots country patches to an axis and returns the dictionary of patches, indexed by ISO3 code.

    :param world: the geopandas world item to use
    :param ax: the axis to use
    :param country_list: list of specific countries to plot (optional)
    :param kwargs: passed to world.plot
    :return: dictionary of country patches
    """
    country_patches = {}
    for country in tqdm.tqdm(list(world['ISO_A3_EH']) if country_list is None else country_list):
        if country not in world['ISO_A3_EH'].values:
            continue
        p = world[world['ISO_A3_EH'] == country].plot(ax=ax, **kwargs)
        country_patches[country] = p._children[-1]
    return country_patches

def add_map(ax, *, add_sea: bool = True, sea_color: str = 'lightblue'):
    """ Prepares an axis for plotting geospatial information

    :param ax: axis to use
    :param add_sea: whether to add the cartopy Sea feature as a background
    :param sea_color: colour to use for the sea
    """
    ax.axis('off')
    if add_sea:
        sea = cartopy.feature.NaturalEarthFeature('physical', 'ocean', '50m',
                                                  facecolor=sea_color, alpha=0.2)
        ax.add_feature(sea, zorder=-3, lw=0)
    ax.set_extent([-180, 180, -60, 73])

def add_cbar_to_map(fig, cm: ColorManager, *, cbar_loc = list | None, **kwargs) -> Colorbar:

    """Adds a colorbar to a map.

    :param fig: figure to use
    :param cm: dantro.ColorManager containing all the relevant information.
    :param cbar_loc: location of the colorbar
    :param kwargs: passed to colorbar.Colorbar
    :return the colorbar
    """
    # Add a colourbar.

    cbar_loc = [0.15, 0.14, 0.01, 0.1] if cbar_loc is None else cbar_loc
    cax = fig.add_axes(cbar_loc)
    cbar = Colorbar(
        cax, cmap=cm.cmap, orientation='vertical', location='right', drawedges=False, norm=cm.norm, **kwargs
    )
    cbar.outline.set_linewidth(0.2)
    cax.tick_params(width=0.2)
    return cbar

def plot_to_map(data, fig, patches, *, highlight_country: str = None, cbar_loc = None, cm: ColorManager = None,
                add_cbar: bool = True, remove_old_cbars: bool = True, **cbar_kwargs) -> tuple[ColorManager, Colorbar | None]:
    """ Plots a dataset to the map. The dataset must be one-dimensional. An individual country can be highlighted.

    :param data: xr.DataArray of data to be plotted to the map. It must be indexed by a single coordinate of the country ISO.
    :param fig: figure to use
    :param patches: dictionary of country patches
    :param highlight_country: specify country to highlight. It is marked in grey and given a hatched fill.
    :param cbar_loc: location of the colorbar
    :param cm: ColorManager to use
    :param remove_old_cbars: whether to remove previously added cbars (true by default, but can be turned off when
        plotting to multiple axes).
    :param cbar_kwargs: kwargs passed to the colormanager, such as a cmap.
    :return the ColorManager and Colorbar used
    """

    # Coordinate to use
    coord = data.squeeze(drop=True).dims[0]

    # Initialise a ColorManager, if none is pased
    if cm is None:
        cbar_kwargs['vmin'] = cbar_kwargs.get('vmin', data.min())
        cbar_kwargs['vmax'] = cbar_kwargs.get('vmax', data.max())

        # Default colourmap to use, if none is provided
        cbar_kwargs['cmap'] = cbar_kwargs.get('cmap', {'continuous': True, 'from_values': {0: colors['c_orange'], 0.5: colors['c_yellow'], 0.75: colors['c_lightblue'], 1: colors['c_darkblue']}})
        cbar_kwargs['norm'] = cbar_kwargs.get('norm')

        # Initialise the ColorManager
        cm = ColorManager(
            **cbar_kwargs
        )

    # Set the patch colour
    for country, patch in patches.items():
        if country not in data.coords[coord].data:
            continue
        else:
            patch.set(lw=0, color=cm.map_to_color(data.sel({coord: country}).data))
        if country == highlight_country:
            patch.set(lw=0.1, color=colors['c_darkgrey'], edgecolor=colors['c_lightgrey'], hatch='//////')

    # Since data may have been previously plotted to the same figure, remove any colourbars that have previously
    # been added to the figure.
    if remove_old_cbars:
        _to_remove = []
        for ax in fig._localaxes:
            if not isinstance(ax, cgeo.GeoAxes):
                _to_remove.append(ax)
        for ax in _to_remove:
            ax.remove()

    # Add a colourbar
    if add_cbar:
        cbar = add_cbar_to_map(fig, cm, cbar_loc=cbar_loc)
        return cm, cbar
    else:
        return cm, None

def coordinate_to_proj(ax, lat, lon) -> tuple[float, float]:
    """ Projects a coordinate point (lat-lon) to the axis projection, in preparation for plotting. """
    return ax.projection.transform_point(lon, lat, src_crs=ccrs.PlateCarree())

def coordinate_to_figure_pos(fig, ax, lat, lon) -> tuple[float, float]:
    """ Converts a coordinate point (lat-lon) to a relative figure position"""
    return fig.transFigure.inverted().transform(ax.transData.transform(
        coordinate_to_proj(ax, lat, lon)
    ))

def is_point_in_extent(ax, lat, lon):
    """
    Checks if a lat-lon point is within the visible extent of the given axes.
    """
    extent = ax.get_extent(crs=ccrs.PlateCarree())  # [xmin, xmax, ymin, ymax] in the CRS
    lon_min, lon_max, lat_min, lat_max = extent

    # Handle crossing the dateline
    if lon_min > lon_max:
        in_lon = lon >= lon_min or lon <= lon_max
    else:
        in_lon = lon_min <= lon <= lon_max

    in_lat = lat_min <= lat <= lat_max

    return in_lon and in_lat

def highlight_countries(ax, patch_dict,
                        countries: str | list,
                        coordinates: dict = None,
                        patch_kwargs: dict = {}, text_kwargs: dict = {}):

    """ Marks countries on the map and adds a country label, if specified.

    :param ax: axis to use
    :param patch_dict: dictionary of patches
    :param countries: country or list of countries to highlight
    :param coordinates: dictionary of coordinates for the text labels (optional)
    """

    default_text_kwargs = dict(fontsize=7, color=colors['c_darkgrey'], alpha=0.5, ha='center', va='center')
    default_text_kwargs.update(**text_kwargs)
    default_patch_kwargs = dict(lw=0.7, ec='white', facecolor=colors['c_lightbrown'], zorder=2)
    default_patch_kwargs.update(**patch_kwargs)
    if isinstance(countries, str):
        countries = [countries]

    # Remove any previous texts
    for text in ax.texts:
        text.remove()

    # Highlight countries in list
    for c in countries:
        if c not in patch_dict.keys():
            continue
        if coordinates is not None and is_point_in_extent(ax, *coordinates[c]):
            ax.text(*coordinate_to_proj(ax, *coordinates[c]), iso_to_name(c).title(), **default_text_kwargs)
        patch_dict[c].set(**default_patch_kwargs)

def add_borders(ax, **kwargs):
    """ Adds borders to an axis.

    :param ax: axis to use
    :param kwargs: kwargs, passed to ax.add_feature
    """
    default_kwargs = dict(linewidth=0.2, edgecolor=colors['c_darkgrey'], linestyle='-', alpha=0.8)
    default_kwargs.update(**kwargs)
    ax.add_feature(cartopy.feature.BORDERS, **default_kwargs)

def add_ocean(ax, **kwargs):
    """ Adds the ocean to an axis.

    :param ax: axis to use
    :param kwargs: kwargs, passed to ax.add_feature
    """
    default_kwargs = dict(linewidth=0., edgecolor=colors['c_lightblue'], alpha=0.2)
    default_kwargs.update(**kwargs)
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'ocean', '50m'), **default_kwargs)

def add_gridlines(ax, **kwargs):
    """ Adds gridlines to an axis.

    :param ax: axis to use
    :param kwargs: kwargs, passed to ax.gridlines()
    """
    default_kwargs = dict(linewidth=0.2, color=colors['c_darkgrey'], linestyle='-', alpha=0.5, draw_labels=False, zorder=max([_.zorder for _ in ax.get_children()]))
    default_kwargs.update(**kwargs)
    ax.gridlines(crs=ccrs.PlateCarree(), **default_kwargs)


def point_in_axis(ax, lat, lon):
    """ Checks whether a point lies in a given axis."""
    return all([~np.isnan(c) for c in coordinate_to_proj(ax, lat, lon)])

def get_coord_ax(fig, coord):
    for ax in fig.axes:
        if point_in_axis(ax, *coord):
            return ax

def plot_arrow(fig, ax_idx: int = None, *, coord1, coord2, **kwargs):

    """ Plots an arrow from one point to another. If the figure is composed of two hemispheres,
    the function determines the hemispheres of each point."""

    def plot_arrow_to_ax(ax, *, posA: tuple[float, float], posB: tuple[float, float], **kwargs):
        """ Plots a FancyArrowPatch to an axis."""
        if 'zorder' not in kwargs:
            kwargs['zorder'] = zorder=max([_.zorder for _ in ax.get_children()])
        ax.add_artist(
            FancyArrowPatch(
                transform=ax.transData,
                posA=posA, posB=posB, **kwargs
        ))

    def plot_arrow_to_fig(fig, *, posA: tuple[float, float], posB: tuple[float, float], **kwargs):
        """ Plots a FancyArrowPatch to a figure"""
        fig.add_artist(
            FancyArrowPatch(
                transform=fig.transFigure,
                posA=posA, posB=posB, **kwargs
        ))

    # If axis was specified:
    if len(fig.axes) == 1 or ax_idx is not None:
        if ax_idx is None:
            ax_idx = 0
        if is_point_in_extent(fig.axes[ax_idx], *coord1) and is_point_in_extent(fig.axes[ax_idx], *coord2):
            plot_arrow_to_ax(fig.axes[ax_idx],
                            posA=coordinate_to_proj(fig.axes[ax_idx], *coord1),
                            posB=coordinate_to_proj(fig.axes[ax_idx], *coord2),
                            **kwargs)
        else:
            return False
    # Need to determine the axis of each point if multiple axes present
    else:
        posA = coordinate_to_figure_pos(fig, get_coord_ax(fig, coord1), *coord1)
        posB = coordinate_to_figure_pos(fig, get_coord_ax(fig, coord2), *coord2)
        plot_arrow_to_fig(fig, posA=posA, posB=posB, **kwargs)

def get_flow_items(T: xr.DataArray) -> list[dict]:
    """ Extracts a list of flow items that can be plotted to a map; items are sorted in descending order.

    :param T: flow table
    :return: list of flow items
    """
    stacked = T.stack({"idx": T.dims}).where(lambda x: x>0).dropna('idx')
    stacked = stacked.reset_index('idx').to_dataframe(name='Value').reset_index(drop=True).to_dict(orient='records')
    return sorted(stacked, key=lambda x: x['Value'], reverse=True)

def plot_flows(fig, *,
               ax_idx: int = None,
               flow_items, special_coords: dict = {},
               color_dict: dict = None,
               color_key: str = None,
               cm: ColorManager = None,
               mutation_scale: float = None,
               jiggle: float = None,
               scale: float = 1e-3,
               **kwargs):

    for item in flow_items:
        item_cfg = {'coord1': special_coords.get(item['Origin ISO'], coordinates[item['Origin ISO']]), 'coord2': special_coords.get(item['Destination ISO'], coordinates[item['Destination ISO']]), 'mutation_scale': scale * item['Value'] if mutation_scale is None else mutation_scale}
        if jiggle:
            coords = copy.deepcopy(coordinates)
            coords.update(**special_coords)
            item_cfg['coord1'] = get_random_coord_in_country(fig, item['Origin ISO'], coordinates=coords, jiggle=jiggle)
            item_cfg['coord2'] = get_random_coord_in_country(fig, item['Destination ISO'], coordinates=coords, jiggle=jiggle)
        if color_dict is not None:
            item_cfg['color'] = color_dict.get(item[color_key], 'C0')
        elif cm is not None:
            item_cfg['color'] = cm.map_to_color(item[color_key])
        plot_arrow(fig, ax_idx=ax_idx, **item_cfg, **kwargs)

def get_random_coord_in_country(fig, country, *, coordinates: dict, jiggle: float = 0.1, attempts=5):

    """Returns a random (lon, lat) point within a country's shape"""
    ax_idx = 0
    center_x, center_y = coordinate_to_proj(fig.axes[ax_idx], *coordinates[country])
    attempt = 0
    while (np.isnan(center_x) or np.isnan(center_y)) and attempt < attempts:
        if ax_idx < len(fig.axes)-1:
            ax_idx += 1
        center_x, center_y = coordinate_to_proj(fig.axes[ax_idx], *coordinates[country])
        attempt += 1
    ax = fig.axes[ax_idx]
    if np.isnan(center_x) or np.isnan(center_y):
        return ccrs.PlateCarree().transform_point(center_x, center_y, src_crs=world.to_crs(ax.projection).crs)[::-1]

    # Get the country geometry
    geometry = (world[world['ISO_A3_EH'] == country]).to_crs(ax.projection)
    if geometry.bounds.empty:
        return coordinates[country]
    minx, miny, maxx, maxy = geometry.bounds.values.tolist()[0]

    # Catch infinities due to projection distortions
    if any([np.isinf(val) for val in geometry.bounds.values.tolist()[0]]):
        return coordinates[country]

    for _ in range(attempts):
        # Generate random point within bounding box
        random_point = Point(np.random.normal(center_x, jiggle*abs(maxx-minx)), np.random.normal(center_y, jiggle*abs(maxy-miny)))
        if all(geometry.contains(random_point)):
            return ccrs.PlateCarree().transform_point(random_point.x, random_point.y, src_crs=world.to_crs(ax.projection).crs)[::-1]
    return ccrs.PlateCarree().transform_point(center_x, center_y, src_crs=world.to_crs(ax.projection).crs)[::-1]

## ---------------------------------------------------------------------------------------------------------------------
## Other plots
## ---------------------------------------------------------------------------------------------------------------------
def errorbar(ds: xr.Dataset, ax, *, x: str = 'Year', y: str, yerr: str, **kwargs):
    """ Plots an xr.Dataset to an axis as an errorbar plot.

    :param ds: dataset, containing a variable for the y-value and one for the y-error.
    :param ax: axis to use
    :param x: x-coordinate
    :param y: variable for the mean
    :param yerr: variable for the error
    :param kwargs: plot kwargs, passed to ax.errorbar
    :return: plot handles
    """
    return ax.errorbar(ds.coords[x], ds[y], ds[yerr], **kwargs)


def errorband(ds: xr.Dataset, ax, *, x: str = 'Year', mean: str = 'mean', std: str = 'std', **kwargs) -> tuple:
    """ Plots an xr.Dataset to an axis, showing standard deviations as an errorband.

    :param ds: xr.Dataset to use; must contain at least two variables, one for the mean and one for the std
    :param ax: axis to use
    :param x: coordinate for the x-axis. 'Year' by default.
    :param mean: variable to use for the mean.
    :param std: variable to use for the standard deviation
    :param kwargs: passed to ax.fill_between and ax.plot (e.g. color)
    :return: plot handles
    """
    x = ds.coords[x]
    mean = ds[mean]
    std = ds[std]
    ebar = ax.fill_between(x, mean -std, mean +std, alpha=0.2, lw=0, **kwargs)
    l = ax.plot(x, mean, **kwargs)

    return (l[0], ebar)