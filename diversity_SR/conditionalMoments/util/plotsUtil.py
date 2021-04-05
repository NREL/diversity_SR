import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
import matplotlib
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def prettyLabels(xlabel,ylabel,fontsize,title=None):
    plt.xlabel(xlabel, fontsize=fontsize, fontweight='bold', fontname="Times New Roman")
    plt.ylabel(ylabel, fontsize=fontsize, fontweight='bold', fontname="Times New Roman")
    if not title==None:
        plt.title(title, fontsize=fontsize, fontweight='bold', fontname="Times New Roman")
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("black")
    plt.grid(color='k', linestyle='-', linewidth=0.5)
    plt.tight_layout()


def axprettyLabels(ax,xlabel,ylabel,fontsize,title=None):
    ax.set_xlabel(xlabel, fontsize=fontsize, fontweight='bold', fontname="Times New Roman")
    ax.set_ylabel(ylabel, fontsize=fontsize, fontweight='bold', fontname="Times New Roman")
    if not title==None:
        ax.set_title(title, fontsize=fontsize, fontweight='bold', fontname="Times New Roman")
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("black")
    plt.grid(color='k', linestyle='-', linewidth=0.5)
    plt.tight_layout()

def plotLegend():
    fontsize = 16
    plt.legend()
    leg=plt.legend(prop={'family':'Times New Roman','size': fontsize-3,'weight':'bold' })
    leg.get_frame().set_linewidth(2.0)
    leg.get_frame().set_edgecolor('k')    

def axplotLegend(ax):
    fontsize = 16
    ax.legend()
    leg=ax.legend(prop={'family':'Times New Roman','size': fontsize-3,'weight':'bold' })
    leg.get_frame().set_linewidth(2.0)
    leg.get_frame().set_edgecolor('k')    

def snapVizZslice(field,x,y,figureDir, figureName,title=None):
    fig,ax = plt.subplots(1)
    plt.imshow(np.transpose(field), cmap=cm.jet, interpolation='bicubic', vmin=np.amin(field), vmax=np.amax(field), extent=[np.amin(x),np.amax(x),np.amax(y),np.amin(y)])
    prettyLabels("x [m]","y [m]", 16, title) 
    plt.colorbar()
    fig.savefig(figureDir+'/'+figureName)
    plt.close(fig)
    return 0

def movieVizZslice(field,x,y,itime,movieDir,minVal=None,maxVal=None):
    fig,ax = plt.subplots(1)
    fontsize =  16
    if minVal==None:
        minVal=np.amin(field)
    if maxVal==None:
        maxVal=np.amax(field)
    plt.imshow(np.transpose(field), cmap=cm.jet, interpolation='bicubic', vmin=minVal, vmax=maxVal, extent=[np.amin(x),np.amax(x),np.amax(y),np.amin(y)])
    prettyLabels("x [m]","y [m]", 16, 'Snap Id = ' + str(itime))
    plt.colorbar()
    fig.savefig(movieDir+'/im_'+str(itime)+'.png')
    plt.close(fig)
    return 0

def plotNIm(field,x,y,minval=None,maxval=None,xLab=None,yLab=None,title=None):
    nImages = len(field)
    if minval==None:
        minval=[]
        for i in range(nImages):
            minval.append(np.amin(field[i]))
    if maxval==None:
        maxval=[]
        for i in range(nImages):
            maxval.append(np.amax(field[i]))
    if xLab==None:
        xLab = []
        for i in range(nImages):
            xLab.append('')
    if yLab==None:
        yLab = []
        for i in range(nImages):
            yLab.append('')
    if title==None:
        title = []
        for i in range(nImages):
            title.append('')

    fig, axs = plt.subplots(1, nImages)
    for i in range(nImages):
        im=axs[i].imshow(np.transpose(field[i]), cmap=cm.viridis, interpolation='nearest', vmin=minval[i], vmax=maxval[i], extent=[np.amin(x[i]),np.amax(x[i]),np.amax(y[i]),np.amin(y[i])])
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=fig.colorbar(im, cax=cax)
        #cbar.set_label('U [m/s]')
        ax = cbar.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(family='times new roman', weight='bold', size=14)
        text.set_font_properties(font)
        axprettyLabels(axs[i],xLab[i],yLab[i],14,title[i])
        axs[i].set_xticks([])
        axs[i].set_xticklabels([])
        axs[i].set_yticks([])
        axs[i].set_yticklabels([])
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_family("serif")
            l.set_fontsize(14)

    #plt.subplots_adjust(wspace=0.3, hspace=0)
    return 0

def makeMovie(ntime,movieDir,movieName,fps=24):
    fig = plt.figure()
    # initiate an empty  list of "plotted" images 
    myimages = []
    #loops through available png:s
    for i in range(ntime):
        ## Read in picture
        fname = movieDir+"/im_"+str(i)+".png"
        myimages.append(imageio.imread(fname))
    imageio.mimsave(movieName, myimages,fps=fps)
    return


def plotHist(field,xLabel,folder, filename):
    fig=plt.figure()
    plt.hist(field)
    fontsize = 18
    prettyLabels(xLabel,"bin count", fontsize)
    fig.savefig(folder + '/' + filename)

def plotContour(x,y,z,color):
    ax =plt.gca()
    X,Y = np.meshgrid(x,y)
    CS = ax.contour(X, Y, np.transpose(z), [0.001, 0.005, 0.01 , 0.05], colors=color)
    h,_ = CS.legend_elements()
    return h[0]
