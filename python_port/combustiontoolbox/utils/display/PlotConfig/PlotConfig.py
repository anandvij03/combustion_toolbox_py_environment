import numpy as np

class PlotConfig:
    def __init__(self, **kwargs):
        self.innerposition = [0.15, 0.15, 0.35, 0.45]
        self.outerposition = [0.15, 0.15, 0.35, 0.45]
        self.innerpositionLayout = [0.15, 0.05, 0.7, 0.9]
        self.outerpositionLayout = [0.15, 0.05, 0.7, 0.9]
        self.linestyle = '-'
        self.symbolstyle = 'o'
        self.lineStyles = ['-', '--', ':', '-.']
        self.symbolStyles = ['d', 'o', 's', '<']
        self.linewidth = 1.8
        self.fontsize = 16
        self.colorpalette = 'Seaborn'
        self.colorpaletteLenght = 11
        self.box = 'off'
        self.grid = 'off'
        self.hold = 'on'
        self.axis_x = 'tight'
        self.axis_y = 'auto'
        self.xscale = 'linear'
        self.yscale = 'linear'
        self.xdir = 'normal'
        self.ydir = 'normal'
        self.padding = 'loose'
        self.tilespacing = 'loose'
        self.title = None
        self.label_type = 'short'
        self.labelx = None
        self.labely = None
        self.legend_name = None
        self.legend_location = 'best'
        self.colorline = [44/255, 137/255, 160/255]
        self.colorlines = np.array([
            [135, 205, 222],
            [95, 188, 211],
            [44, 137, 160],
            [22, 68, 80]
        ]) / 255.0
        self.blue = [0.3725, 0.7373, 0.8275]
        self.gray = [0.50, 0.50, 0.50]
        self.red = [0.64, 0.08, 0.18]
        self.orange = [212/255, 85/255, 0]
        self.brown = [200/255, 190/255, 183/255]
        self.brown2 = [72/255, 55/255, 55/255]
        self.id_polar1 = 1001
        self.id_polar2 = 1002
        self.id_polar3 = 1003
        self.displaySpecies = None
        self.mintolDisplay = 1e-6
        self.plotProperties = ['T', 'p', 'rho', 'h', 'e', 'g', 'cp', 's', 'gamma_s', 'sound']
        self.plotPropertiesBasis = [None, None, None, 'mi', 'mi', 'mi', 'mi', 'mi', None, None]

        # Apply overrides
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def numPlotProperties(self):
        return len(self.plotProperties)

    @property
    def position(self):
        return [0, 0, 800, 600]

    @property
    def numStyles(self):
        return len(self.lineStyles)

    @property
    def cmap(self):
        import matplotlib.pyplot as plt
        try:
            return plt.get_cmap('tab10').colors
        except:
            return [[0,0,1]] * self.colorpaletteLenght

    def set(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self
