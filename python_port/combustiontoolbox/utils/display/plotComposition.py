import numpy as np
import matplotlib.pyplot as plt
from combustiontoolbox.utils.display.PlotConfig import PlotConfig
from combustiontoolbox.utils.display.setFigure import setFigure
from combustiontoolbox.utils.display.getTitle import getTitle
from combustiontoolbox.utils.display.interpreterLabel import interpreterLabel
from combustiontoolbox.utils.display.species2latex import species2latex
from combustiontoolbox.utils.findIndex import findIndex

def plotComposition(obj, x_var, x_field, y_field, *args, **kwargs):
    """
    Plot molar fractions against any variable
    """
    varargin = list(args)

    # Initialization
    config = PlotConfig()
    mintolDisplay = config.mintolDisplay
    displaySpecies = config.displaySpecies

    # Default values
    ax = None
    results2 = None
    nfrec = 1
    FLAG_PLOT_VALIDATION = False
    config.labelx = interpreterLabel(x_field, config.label_type)
    config.labely = interpreterLabel(y_field, config.label_type)
    config.innerposition = [0.15, 0.05, 0.7, 0.9]
    config.outerposition = [0.15, 0.05, 0.7, 0.9]
    config.yscale = 'log'
    y_var = x_var

    # Unpack args
    kwargs_merged = {}
    i = 0
    while i < len(varargin):
        key = varargin[i]
        if isinstance(key, str) and i + 1 < len(varargin):
            val = varargin[i+1]
            kwargs_merged[key.lower()] = val
            i += 2
        else:
            i += 1

    for k, v in kwargs.items():
        kwargs_merged[k.lower()] = v

    if 'validation' in kwargs_merged or 'results' in kwargs_merged:
        results2 = kwargs_merged.get('validation') or kwargs_merged.get('results')
        FLAG_PLOT_VALIDATION = True
    if 'nfrec' in kwargs_merged:
        nfrec = kwargs_merged['nfrec']
    if any(k in kwargs_merged for k in ['mintol', 'mintol_display', 'toln']):
        mintolDisplay = next(kwargs_merged[k] for k in ['mintol', 'mintol_display', 'toln'] if k in kwargs_merged)
    if any(k in kwargs_merged for k in ['ls', 'species', 'displayspecies', 'display species', 'display_species']):
        displaySpecies = next(kwargs_merged[k] for k in ['ls', 'species', 'displayspecies', 'display species', 'display_species'] if k in kwargs_merged)
    if any(k in kwargs_merged for k in ['y', 'y_var', 'yvar', 'y var', 'y_data', 'ydata', 'y data']):
        y_var = next(kwargs_merged[k] for k in ['y', 'y_var', 'yvar', 'y var', 'y_data', 'ydata', 'y data'] if k in kwargs_merged)
    if 'config' in kwargs_merged:
        config = kwargs_merged['config']
    if 'xscale' in kwargs_merged:
        config.xscale = kwargs_merged['xscale']
    if 'yscale' in kwargs_merged:
        config.yscale = kwargs_merged['yscale']
    if 'xdir' in kwargs_merged:
        config.xdir = kwargs_merged['xdir']
    if 'ydir' in kwargs_merged:
        config.ydir = kwargs_merged['ydir']
    if 'title' in kwargs_merged:
        config.title = kwargs_merged['title']
    if 'ax' in kwargs_merged or 'axes' in kwargs_merged:
        ax = kwargs_merged.get('ax') or kwargs_merged.get('axes')

    # Set species to displaySpecies
    species, listSpecies = get_displaySpecies(obj, displaySpecies)

    # Read data
    FLAG_Y_AXIS = (y_field.lower() == 'xi')
    
    if isinstance(x_var, (list, tuple, np.ndarray)) and len(x_var) > 0 and hasattr(x_var[0], x_field):
        x_data = np.array([getattr(m, x_field) for m in x_var])
        y_data = np.array([getattr(m, y_field) for m in y_var]).T
    else:
        x_data = np.asarray(x_var)
        y_data = np.asarray(y_var)

    # Check lengths
    if len(x_data) < 2:
        return None, None

    # Get index species
    index_species_CT = findIndex(listSpecies, species)
    if not isinstance(index_species_CT, list):
        index_species_CT = [index_species_CT]

    # Remove species that do not appear
    # Here Xi in mix1 is effectively y_data.T if FLAG_Y_AXIS else x_data.T
    species, index_species_CT = clean_displaySpecies(y_data.T if FLAG_Y_AXIS else x_data.T, species, index_species_CT, mintolDisplay, FLAG_PLOT_VALIDATION)

    # Set figure
    if ax is None:
        ax = setFigure(config=config)
        fig = ax.get_figure()
    else:
        fig = ax.get_figure()


    # Set axis limits
    if FLAG_Y_AXIS:
        ax.set_xlim([np.min(x_data), np.max(x_data)])
        ax.set_ylim([mintolDisplay, 1.0])
    else:
        ax.set_xlim([mintolDisplay, 1.0])
        ax.set_ylim([np.min(y_data), np.max(y_data)])

   # Set default style
    NE = len(species)
    
    # Use matplotlib's tab20 for 20 highly distinguishable colors
    colorbw = plt.get_cmap('tab20').colors 
    maxLdisplay = len(colorbw)
    if NE > maxLdisplay:
        NUM_COLORS = maxLdisplay
    else:
        NUM_COLORS = NE
    LINE_STYLES = ['-', '--', ':', '-.']
    SYMBOL_STYLES = ['d', 'o', 's', '<']
    NUM_STYLES = len(LINE_STYLES)

    # Plot main results
    k = 0
    z = 0

    h_lines = []

    for i in range(len(species)):
        sp_idx = index_species_CT[i]

        if FLAG_Y_AXIS:
            line, = ax.plot(x_data, y_data[sp_idx, :], linewidth=config.linewidth, color=colorbw[k % len(colorbw)], linestyle=LINE_STYLES[z % NUM_STYLES])
        else:
            line, = ax.plot(y_data[sp_idx, :], x_data, linewidth=config.linewidth, color=colorbw[k % len(colorbw)], linestyle=LINE_STYLES[z % NUM_STYLES])

        h_lines.append(line)

        k += 1
        if k == maxLdisplay:
            k = 0
            z += 1
            if z >= NUM_STYLES:
                z = 0

    # Plot validations
    h = h_lines
    if FLAG_PLOT_VALIDATION and results2 is not None:
        if isinstance(results2, (list, tuple, np.ndarray)) and len(results2) > 0 and hasattr(results2[0], x_field):
            x_data2 = np.array([getattr(m, x_field) for m in results2])
            y_data2 = np.array([getattr(m, y_field) for m in results2]).T
        else:
            x_data2 = np.asarray(results2.get(x_field, []))
            y_data2 = np.asarray(results2.get(y_field, []))

        k = 0
        z = 0

        for i in range(len(species)):
            sp_idx = index_species_CT[i]

            if FLAG_Y_AXIS:
                ax.plot(x_data2[::nfrec], y_data2[sp_idx, ::nfrec], linestyle='', marker=SYMBOL_STYLES[z % len(SYMBOL_STYLES)], linewidth=config.linewidth, color=colorbw[k % len(colorbw)], markerfacecolor='white', markeredgewidth=config.linewidth)
            else:
                ax.plot(y_data2[sp_idx, ::nfrec], x_data2[::nfrec], linestyle='', marker=SYMBOL_STYLES[z % len(SYMBOL_STYLES)], linewidth=config.linewidth, color=colorbw[k % len(colorbw)], markerfacecolor='white', markeredgewidth=config.linewidth)

            k += 1

            if k == maxLdisplay:
                k = 0
                z += 1
                if z >= NUM_STYLES:
                    z = 0

        # Plot symbols (MATLAB places NaN points here to get legends right, but we passed handles directly to legend)

    # Set legend
    legendname = [species2latex(sp) for sp in species]
    if len(h) > 0:
        ax.legend(h, legendname, fontsize=config.fontsize - 4, loc='upper left', bbox_to_anchor=(1.02, 1))

    # Set title
    t1 = config.title
    t2 = getTitle(obj)
    if not t1:
        if t2:
            ax.set_title(t2, fontsize=config.fontsize + 2)
    else:
        if t2:
            ax.set_title(f"{t1} - {t2}", fontsize=config.fontsize + 2)
        else:
            ax.set_title(t1, fontsize=config.fontsize + 2)
            
    # Remove plt.tight_layout() if setFigure handles it or just keep it
    plt.tight_layout()

    return ax, fig

# SUB-PASS FUNCTIONS
def get_displaySpecies(obj, displaySpecies):
    listSpecies = obj.chemicalSystem.listSpecies

    if not displaySpecies:
        species = obj.chemicalSystem.listProducts
    else:
        species = displaySpecies

    return species, listSpecies

def clean_displaySpecies(molar_fractions, species, index_species, mintolDisplay, FLAG_PLOT_VALIDATION):
    # Remove species that do not appear

    # Checks
    if FLAG_PLOT_VALIDATION:
        mintolDisplay = 0.0

    valid_indices = []
    valid_species = []

    # Molar fractions here is of shape (numPoints, numSpecies)
    # We iterate over index_species
    for sp_idx, sp_name in zip(index_species, species):
        if np.any(molar_fractions[:, sp_idx] > mintolDisplay):
            valid_indices.append(sp_idx)
            valid_species.append(sp_name)

    return valid_species, valid_indices
