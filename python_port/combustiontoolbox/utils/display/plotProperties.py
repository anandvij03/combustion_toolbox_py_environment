import numpy as np
import matplotlib.pyplot as plt
from combustiontoolbox.utils.display.PlotConfig import PlotConfig
from combustiontoolbox.utils.display.setFigure import setFigure
from combustiontoolbox.utils.display.setTitle import setTitle
from combustiontoolbox.utils.display.getTitle import getTitle
from combustiontoolbox.utils.display.plotFigure import plotFigure

def plotProperties(x_field, x_var, y_field, y_var, *args, **kwargs):
    """
    Plot figure with customizable settings.
    """
    varargin = list(args)
    
    nfrec = 1
    FLAG_PLOT_VALIDATION = False
    FLAG_BASIS = False
    FLAG_SAME = False
    main_ax = None
    config = PlotConfig()

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
    if 'config' in kwargs_merged:
        config = kwargs_merged['config']
    if 'leg' in kwargs_merged or 'legend' in kwargs_merged:
        config.legend_name = kwargs_merged.get('leg') or kwargs_merged.get('legend')
    if 'legend_location' in kwargs_merged:
        config.legend_location = kwargs_merged['legend_location']
    if 'ax' in kwargs_merged or 'axes' in kwargs_merged:
        main_ax = kwargs_merged.get('ax') or kwargs_merged.get('axes')
    if 'linestyle' in kwargs_merged:
        config.linestyle = kwargs_merged['linestyle']
    if 'linewidth' in kwargs_merged:
        config.linewidth = kwargs_merged['linewidth']
    if 'fontsize' in kwargs_merged:
        config.fontsize = kwargs_merged['fontsize']
    if 'title' in kwargs_merged:
        config.title = kwargs_merged['title']
    if any(k in kwargs_merged for k in ['labelx', 'xlabel', 'label_x', 'x_label']):
        config.labelx = next(kwargs_merged[k] for k in ['labelx', 'xlabel', 'label_x', 'x_label'] if k in kwargs_merged)
    if any(k in kwargs_merged for k in ['labely', 'ylabel', 'label_y', 'y_label']):
        config.labely = next(kwargs_merged[k] for k in ['labely', 'ylabel', 'label_y', 'y_label'] if k in kwargs_merged)
    if 'label_type' in kwargs_merged:
        config.label_type = kwargs_merged['label_type']
    if 'xscale' in kwargs_merged:
        config.xscale = kwargs_merged['xscale']
    if 'yscale' in kwargs_merged:
        config.yscale = kwargs_merged['yscale']
    if 'xdir' in kwargs_merged:
        config.xdir = kwargs_merged['xdir']
    if 'ydir' in kwargs_merged:
        config.ydir = kwargs_merged['ydir']
    if 'basis' in kwargs_merged:
        basis = kwargs_merged['basis']
        FLAG_BASIS = True
    if 'nfrec' in kwargs_merged:
        nfrec = kwargs_merged['nfrec']

    colorPalette = config.colorlines
    symbolStyles = config.symbolStyles
    
    selectColor = 0
    selectSymbol = 0

    if not isinstance(x_field, list):
        x_field = [x_field]
    if not isinstance(y_field, list):
        y_field = [y_field]

    fig = None
    if main_ax is None:
        N_properties = len(y_field)
        cols = int(np.ceil(np.sqrt(N_properties)))
        rows = int(np.ceil(N_properties / cols))
        fig, axes_grid = plt.subplots(rows, cols, squeeze=False, figsize=(12, 8))
        main_ax = axes_grid.flatten()
        for idx in range(N_properties, len(main_ax)):
            fig.delaxes(main_ax[idx])
        main_ax = main_ax[:N_properties]
    else:
        fig = main_ax[0].get_figure() if isinstance(main_ax, (list, np.ndarray)) else main_ax.get_figure()
        FLAG_SAME = True
        if not isinstance(main_ax, (list, np.ndarray)):
            main_ax = [main_ax]

    x_var_first = x_var[0] if isinstance(x_var, (list, tuple, np.ndarray)) else x_var
    if hasattr(x_var_first, '__class__') and x_var_first.__class__.__name__ == 'Mixture':
        if not config.title:
            config.title = getTitle(x_var_first)
        else:
            config.title = f"{config.title} - {getTitle(x_var_first)}"

    if config.title:
        fig.suptitle(config.title, fontsize=config.fontsize + 2)
    config.title = None

    N_properties = len(y_field)
    if not FLAG_BASIS:
        basis = [None] * N_properties
    elif not isinstance(basis, list):
        basis = [basis] * N_properties

    for i in range(N_properties):
        ax = main_ax[i]
        setFigure(ax, config)

        basis_i = basis[i] if i < len(basis) else None
        
        plotFigure(
            x_field[i], x_var, y_field[i], y_var,
            config=config, ax=ax, basis=basis_i,
            color=colorPalette[selectColor % len(colorPalette)]
        )

        if FLAG_PLOT_VALIDATION and results2:
            val_x = getattr(results2, x_field[i])[::nfrec]
            val_y = getattr(results2, y_field[i])[::nfrec]
            ax.plot(
                val_x, val_y,
                marker=symbolStyles[selectSymbol % len(symbolStyles)],
                linestyle='',
                color=colorPalette[selectColor % len(colorPalette)],
                markerfacecolor='white',
                markeredgewidth=config.linewidth
            )

    plt.tight_layout()
    # Call plt.show() to render the graphs
    plt.show()
    return main_ax, fig
