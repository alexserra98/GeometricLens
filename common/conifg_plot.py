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