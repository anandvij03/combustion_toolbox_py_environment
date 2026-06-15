import numpy as np
import matplotlib.pyplot as plt
from combustiontoolbox.utils.display.Canvas import Canvas
from combustiontoolbox.utils.display.PlotConfig import PlotConfig
from combustiontoolbox.utils.display.interpreterLabel import interpreterLabel
from combustiontoolbox.utils.display.species2latex import species2latex
from combustiontoolbox.utils.display.getTitle import getTitle
from combustiontoolbox.utils.findIndex import findIndex

class PlotComposition(Canvas):
    """
    The PlotComposition class extends Canvas to generate species composition plots
    (e.g., molar fractions Xi) against an independent variable.
    """

    def __init__(self, config=None):
        """
        Constructor
        """
        if config is None:
            # Default values
            defaultInnerPosition = [0.15, 0.05, 0.7, 0.9]
            defaultOuterPosition = [0.15, 0.05, 0.7, 0.9]

            # Get default configuration
            config = PlotConfig()
            config.innerposition = defaultInnerPosition
            config.outerposition = defaultOuterPosition

        super().__init__(config)
        self.__FLAG_PLOT_VALIDATION = False

    def plot(self, mixture, x_var, x_field, y_field, *args, **kwargs):
        """
        Plot molar fractions against any variable
        """
        varargin = list(args)
        
        # Initialization
        config = self.config
        mintolDisplay = config.mintolDisplay
        displaySpecies = config.displaySpecies

        # Default values
        ax = None
        results2 = None
        nfrec = 1
        config.labelx = interpreterLabel(x_field, config.label_type)
        config.labely = interpreterLabel(y_field, config.label_type)
        config.yscale = 'log'
        y_var = x_var

        # Unpack kwargs and varargin (simulating MATLAB inputParser/varargin logic)
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

        if any(k in kwargs_merged for k in ['validation', 'results']):
            results2 = next(kwargs_merged[k] for k in ['validation', 'results'] if k in kwargs_merged)
            self.__FLAG_PLOT_VALIDATION = True
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
        if any(k in kwargs_merged for k in ['ax', 'axes']):
            ax = next(kwargs_merged[k] for k in ['ax', 'axes'] if k in kwargs_merged)

        # Set species to displaySpecies
        species, listSpecies = self.getDisplaySpecies(mixture, displaySpecies)

        FLAG_Y_AXIS = (y_field.lower() == 'xi')

        # Read data
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
        species, index_species_CT = self.cleanDisplaySpecies(y_data.T if FLAG_Y_AXIS else x_data.T, listSpecies, index_species_CT, mintolDisplay, self.__FLAG_PLOT_VALIDATION)

        # Set figure
        if ax is None:
            ax = self.getAxis()
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
        maxLdisplay = len(config.colorlines) # using colorlines instead of brewermap directly
        if NE > maxLdisplay:
            numColors = maxLdisplay
        else:
            numColors = NE

        colorbw = config.colorlines
        lineStyles = config.lineStyles

        # Plot main results
        k = 0
        z = 0
        h_lines = []

        for i in range(len(species)):
            sp_idx = index_species_CT[i]

            if FLAG_Y_AXIS:
                line, = ax.plot(x_data, y_data[sp_idx, :], linewidth=config.linewidth, color=colorbw[k % len(colorbw)], linestyle=lineStyles[z % len(lineStyles)])
            else:
                line, = ax.plot(y_data[sp_idx, :], x_data, linewidth=config.linewidth, color=colorbw[k % len(colorbw)], linestyle=lineStyles[z % len(lineStyles)])
            
            h_lines.append(line)
            k += 1

            if k == maxLdisplay:
                k = 0
                z += 1
                if z >= len(lineStyles):
                    z = 0

        # Plot validations
        h = h_lines
        if self.__FLAG_PLOT_VALIDATION and results2 is not None:
            if isinstance(results2, (list, tuple, np.ndarray)) and len(results2) > 0 and hasattr(results2[0], x_field):
                x_data2 = np.array([getattr(m, x_field) for m in results2])
                y_data2 = np.array([getattr(m, y_field) for m in results2]).T
            else:
                x_data2 = np.asarray(results2.get(x_field, []))
                y_data2 = np.asarray(results2.get(y_field, []))

            k = 0
            z = 0
            symbolStyles = config.symbolStyles

            for i in range(len(species)):
                sp_idx = index_species_CT[i]

                if FLAG_Y_AXIS:
                    ax.plot(x_data2[::nfrec], y_data2[sp_idx, ::nfrec], linestyle='', marker=symbolStyles[z % len(symbolStyles)], linewidth=config.linewidth, color=colorbw[k % len(colorbw)], markerfacecolor='white', markeredgewidth=config.linewidth)
                else:
                    ax.plot(y_data2[sp_idx, ::nfrec], x_data2[::nfrec], linestyle='', marker=symbolStyles[z % len(symbolStyles)], linewidth=config.linewidth, color=colorbw[k % len(colorbw)], markerfacecolor='white', markeredgewidth=config.linewidth)

                k += 1

                if k == maxLdisplay:
                    k = 0
                    z += 1
                    if z >= len(symbolStyles):
                        z = 0

            # Symbols logic (empty lines for legend)
            # Just reuse h_lines as they combine color and marker in legend ideally, 
            # or recreate handles if we strictly follow MATLAB
            # But matplotlib handles legend combination automatically if we pass handles.
            pass

        # Set legend
        legendname = [species2latex(sp) for sp in species]
        if len(h) > 0:
            ax.legend(h, legendname, fontsize=config.fontsize - 4, loc='upper left', bbox_to_anchor=(1.02, 1))

        # Set title
        t1 = config.title
        t2 = getTitle(mixture)
        if not t1:
            if t2:
                ax.set_title(t2, fontsize=config.fontsize + 4)
        else:
            if t2:
                ax.set_title(f"{t1} - {t2}", fontsize=config.fontsize + 4)
            else:
                ax.set_title(t1, fontsize=config.fontsize + 4)

        return ax, fig

    @staticmethod
    def getDisplaySpecies(mixture, displaySpecies):
        listSpecies = mixture.chemicalSystem.listSpecies

        if not displaySpecies:
            species = mixture.chemicalSystem.listProducts
        else:
            species = displaySpecies

        return species, listSpecies

    @staticmethod
    def cleanDisplaySpecies(molar_fractions, species, index_species, mintolDisplay, FLAG_PLOT_VALIDATION):
        """
        Remove species that do not appear
        """
        if FLAG_PLOT_VALIDATION:
            mintolDisplay = 0.0

        # Molar fractions shape in MATLAB is (numSpecies, numPoints), but here we passed it as such or transposed.
        # Actually in Python if y_data is passed it is (numSpecies, numPoints). We should just check max over axis 1.
        if molar_fractions.ndim == 2 and molar_fractions.shape[0] == len(index_species):
            pass # if already filtered
        
        # Safe way: iterate and check
        valid_indices = []
        valid_species = []
        
        for sp_idx, sp_name in zip(index_species, species):
            # Check if this species ever exceeds mintolDisplay
            # Depending on how molar_fractions was passed, let's assume molar_fractions is full array (numSpecies, numPoints)
            # wait, MATLAB passed `mix1.Xi` which is all species.
            if np.any(molar_fractions[sp_idx, :] > mintolDisplay):
                valid_indices.append(sp_idx)
                valid_species.append(sp_name)

        return valid_species, valid_indices
