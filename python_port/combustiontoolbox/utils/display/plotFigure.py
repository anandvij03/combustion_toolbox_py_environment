import numpy as np
import matplotlib.pyplot as plt
from combustiontoolbox.utils.display.PlotConfig import PlotConfig
from combustiontoolbox.utils.display.setFigure import setFigure
from combustiontoolbox.utils.display.interpreterLabel import interpreterLabel
from combustiontoolbox.utils.cell2vector import cell2vector

def plotFigure(x_field, x_var, y_field, y_var, **kwargs):
    """
    Plot figure with customizable settings using matplotlib.
    """
    config = kwargs.get('config', None)
    if config is None:
        config = PlotConfig()
        
    ax = kwargs.get('ax', None)
    basis = kwargs.get('basis', None)

    try:
        x = cell2vector(x_var, x_field)
        y = cell2vector(y_var, y_field)
    except:
        x = getattr(x_var, x_field)
        y = getattr(y_var, y_field)

    if len(x) < 2:
        return ax, None

    label_type = getattr(config, 'label_type', 'short')
    config.labelx = interpreterLabel(x_field, label_type, False)
    config.labely = interpreterLabel(y_field, label_type, False)

    for key, value in kwargs.items():
        if hasattr(config, key) and key not in ['ax', 'config', 'basis']:
            setattr(config, key, value)

    if not config.labelx:
        config.labelx = interpreterLabel(x_field, label_type, False)
    if not config.labely:
        config.labely = interpreterLabel(y_field, label_type, False)

    if ax is None:
        ax = setFigure(None, config)

    if y_field in ['cp', 'cv', 'hf', 'ef', 'h', 'e', 'g', 's']:
        y = np.array(y) * 1e-3

    if basis:
        y_basis = np.array(cell2vector(y_var, basis))
        y = np.array(y) / y_basis
        config.labelx = interpreterLabel(x_field, label_type, True, basis)
        if config.labely != 'Multiple variables':
            config.labely = interpreterLabel(y_field, label_type, True, basis)

    color = kwargs.get('color', config.colorline)
    linestyle = kwargs.get('linestyle', config.linestyle)
    linewidth = kwargs.get('linewidth', config.linewidth)

    dline, = ax.plot(x, y, linestyle=linestyle, linewidth=linewidth, color=color)

    ax = setFigure(ax, config)

    if config.title:
        ax.set_title(config.title, fontsize=config.fontsize + 2)

    if config.legend_name:
        ax.legend(config.legend_name, fontsize=config.fontsize - 4, loc=config.legend_location)

    return ax, dline
