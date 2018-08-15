import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np

import utils.mpl_stylesheet as mplstyle
mplstyle.banskt_presentation()

def draw(df, icol, jcol, ax, xlabel, ylabel, inanlegend, jnanlegend, titletext = None, showfill = 0.0):

    marker_alpha = 0.5
    marker_size = 240

    j_inan = list()
    i_jnan = list()
    xvals = list()
    yvals = list()

    for index, row in df.iterrows():
        if np.isnan(row[icol]) or row[icol] == 0:
            if not np.isnan(row[jcol]) and row[jcol] != 0:
                j_inan.append(row[jcol])
        elif np.isnan(row[jcol]) or row[jcol] == 0:
            if not np.isnan(row[icol]) and row[icol] != 0:
                i_jnan.append(row[icol])
        else:
            xvals.append(row[icol])
            yvals.append(row[jcol])
        
    a = np.log10(min(min(xvals), min(yvals)))
    if a < -3.5:
        xmin = np.round(a)
    else:
        xmin = np.floor(a * 2.0) / 2.0
    xmax = 0
    nan_val = xmin
    delta = 0.5
    xdiag = np.linspace(xmin - delta, xmax, 1000)

    ## Plot properties
    colors = ['#2D69C4', '#CC2529', '#93AA00', '#535154', '#BEBEBE']

    ax.plot (xdiag, xdiag, color = colors[4], lw = 1, zorder = 10)
    if showfill > 0:
        xfill  = np.linspace(xmin, xmax, 1000)
        yfill1 = np.maximum(np.repeat(xmin, len(xfill)), xfill - showfill)
        yfill2 = np.minimum(np.repeat(xmax, len(xfill)), xfill + showfill)
        ax.fill_between(xfill, yfill1, yfill2, color = colors[4], alpha = 0.2)

    thiscolor = colors[0]
    facecolor = clr.ColorConverter().to_rgba(thiscolor, alpha=marker_alpha)
    edgecolor = clr.ColorConverter().to_rgba(thiscolor, alpha=1)
    ax.scatter(np.log10(xvals), np.log10(yvals), zorder = 30, 
               marker = 'o', color = facecolor, s = marker_size, lw = 1, edgecolor = edgecolor,
               label = 'Both')
    
    xjitter = np.random.normal(nan_val, delta / 10, len(j_inan))
    thiscolor = colors[2]
    facecolor = clr.ColorConverter().to_rgba(thiscolor, alpha=marker_alpha)
    edgecolor = clr.ColorConverter().to_rgba(thiscolor, alpha=1)
    ax.scatter(xjitter, np.log10(j_inan), zorder = 20,
               marker = 'D', color = facecolor, s = marker_size * 0.875, lw = 1, edgecolor = edgecolor,
               label = inanlegend)
    
    yjitter = np.random.normal(nan_val, delta / 10, len(i_jnan))
    thiscolor = colors[1]
    facecolor = clr.ColorConverter().to_rgba(thiscolor, alpha=marker_alpha)
    edgecolor = clr.ColorConverter().to_rgba(thiscolor, alpha=1)
    ax.scatter(np.log10(i_jnan), yjitter, zorder = 20,
               marker = 's', color = facecolor, s = marker_size * 0.95, lw = 1, edgecolor = edgecolor,
               label = jnanlegend)

    if titletext is not None:
        plt.title(titletext)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xlim([xmin - delta, xmax])
    ax.set_ylim([xmin - delta, xmax])
    if xmin < -3.0:
        mticks = np.arange(xmin, xmax + 0.25, 1.0)
    else:
        mticks = np.arange(xmin, xmax + 0.25, 0.5)
    ax.set_xticks(mticks)
    ax.set_yticks(mticks)
    
    ax.legend()
    h, l = ax.get_legend_handles_labels()
    legend = ax.legend(title = 'Prediction Success', handles = h, loc = 'lower right', bbox_to_anchor=(1.0, 0.05))
    legend._legend_box.align = "left"
    ax.add_artist(legend)
    
    # for l in legend.legendHandles:
    #     l.set_alpha(1)

    #renderer = fig.canvas.get_renderer()
    #shift = max([t.get_window_extent(renderer).width for t in legend.get_texts()])
    #for fonts in legend.texts:
    #    fonts.set_position((0, -2))
    
    for side, border in ax.spines.items():
        if side == 'left' or side == 'bottom':
            border.set_bounds(xmin, xmax)

    return
