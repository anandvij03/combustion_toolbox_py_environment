def setTitle(ax, config=None):
    if config is None:
        fontsize = 18
        title_str = ""
    else:
        fontsize = getattr(config, 'fontsize', 18)
        title_str = getattr(config, 'title', "")

    if title_str and ax:
        ax.set_title(title_str, fontsize=fontsize + 2)
