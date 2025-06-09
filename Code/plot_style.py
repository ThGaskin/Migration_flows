from cycler import cycler
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

""" Plot style settings for uniform Latex integration. """

# Palette of pretty colours
colors = {
    'c_darkblue': '#2F7194',
    'c_red': '#ec7070',
    'c_darkgreen': '#48675A',
    'c_lightblue': '#97c3d0',
    'c_lightgreen': '#AFD8BC',
    'c_lightbrown': '#C6BFA2',
    'c_orange': '#EC9F7E',
    'c_yellow': '#F5DDA9',
    'c_darkgrey': '#3D4244',
    'c_pink': '#F8A6A6',
    'c_purple': '#A07CB0',
    'c_lightgrey': '#AFC1B9',
}

# Plots style using the matplotlib rc parameters keys. Integrate this into the plots by updating the matplotlib.rcParams
# dictionary: ``rcParams.update(**style)``.
fs=9
style = {"text.usetex": False, "mathtext.fontset": "cm", "font.family": ["Garamond Premier Pro", "serif"],
         "text.latex.preamble": r"\usepackage{amssymb} \usepackage{amsmath}", "font.size": fs, "axes.titlesize": fs,
         "axes.labelsize": fs, "xtick.labelsize": fs, "ytick.labelsize": fs, "legend.fontsize": fs,
         "grid.linewidth": 0.5, "savefig.bbox": "tight", "savefig.transparent": True, "axes.grid": True, "axes.spines.top": False,
         "axes.spines.right": False, "figure.subplot.left": 0, "figure.subplot.right": 1, "figure.subplot.bottom": 0,
         "figure.subplot.top": 1, "legend.facecolor": (1, 1, 1, 0.1), "figure.frameon": False,
         'axes.prop_cycle': cycler(color=list(colors.values())), 'patch.edgecolor': 'none', 'savefig.dpi': 400,
         "legend.handlelength": 1, "legend.framealpha": 0,}

