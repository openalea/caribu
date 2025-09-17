import matplotlib as mpl
from matplotlib import pyplot

def colorScale(minval, maxval, label):
    """    Produce a plot of the colorscale used by ViewMapOnCan when gamma == 1
    """
    # Make a figure and axes with dimensions as desired.
    fig = pyplot.figure(figsize=(8,1.5))
    ax1 = fig.add_axes([0.05, 0.4, 0.9, .5])

    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=minval, vmax=maxval)

    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                   norm=norm,
                                   orientation='horizontal')
    cb1.set_label(label)

    fig.show()
 
    return fig,






