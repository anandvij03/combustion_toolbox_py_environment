import matplotlib.pyplot as plt
from combustiontoolbox.utils.display.PlotConfig import PlotConfig

def setFigure(ax=None, config=None):
    if config is None:
        config = PlotConfig()

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.tick_params(labelsize=config.fontsize - 2)
    ax.set_xscale(config.xscale)
    ax.set_yscale(config.yscale)
    
    if config.grid == 'on':
        ax.grid(True)
    else:
        ax.grid(False)
        
    if config.box == 'on':
        for spine in ax.spines.values():
            spine.set_visible(True)
    elif config.box == 'off':
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if config.labelx:
        ax.set_xlabel(config.labelx, fontsize=config.fontsize)
    if config.labely:
        ax.set_ylabel(config.labely, fontsize=config.fontsize)
        
    return ax
