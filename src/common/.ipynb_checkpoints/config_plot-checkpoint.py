from dataclasses import dataclass

@dataclass
class ConfigPlotSize:
    width: int  = 12
    height: int = 8
    xticks: int = 18
    yticks: int = 18
    xlabel: int = 27.5
    ylabel: int = 27.5
    title: int  = 30
    hspace: int = 0.8
    wspace: int = 0.8
    legend: int = 23
    linewidth: int = 2.5
    s: int = 100

plot_config = {
    #'font.size': 12,           
    'axes.titlesize': 30,      
    'axes.labelsize': 29,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 23,
    'figure.figsize': (10, 8),
    'lines.linewidth': 2.5,
    'lines.markersize': 10,
}