# /------------------------------------------------------------/
# **2D Visualization Helper Function**
# -----
# *pysovist-dev* under MIT License
# -----
# This is a helper function which helps visualize a floor plan
# in addition to the metrics calculated in the Data2D class.
# -----
# Best use case: floor plan pysovist workflows.
# -----
# What's in the file:
# 1. imports
# 2. base method
# /------------------------------------------------------------/

## 1. Imports
import numpy as np
import matplotlib.pyplot as plt

def view_baseplan(plan_lines:np.array,dark:bool=False,show_grid:bool=True,raster:bool=False,**kwargs) -> None:
    clr = '#444433' if not dark else '#ccccbe'
    bg = '#fffffc' if not dark else '#100804'
    if raster == True:
        plt.figure(facecolor=bg,dpi=200)
    else:
        plt.figure(facecolor=bg)
    plt.tick_params(colors=clr) 
    plt.gca().set_facecolor(bg)
    plt.axis('equal')
    for spine in plt.gca().spines.values():
        spine.set_color(clr)
    if show_grid == True:
        plt.grid(color=clr,alpha=0.1)
    lw = kwargs.get('linewidth',2)
    if raster == False:
        for line in plan_lines:
            plt.plot(line[:,0],line[:,1],c=clr,linewidth=lw)
    else:
        res = kwargs.get('res')
        plt.xticks(ticks=np.linspace(plan_lines[:,1].min(),plan_lines[:,1].max(),6,dtype=int),labels=np.linspace(plan_lines[:,1].min()/res,plan_lines[:,1].max()/res,6,dtype=int))
        plt.yticks(ticks=np.linspace(plan_lines[:,0].min(),plan_lines[:,0].max(),6,dtype=int),labels=np.linspace(plan_lines[:,0].min()/res,plan_lines[:,0].max()/res,6,dtype=int))
        plt.scatter(plan_lines[:,1],plan_lines[:,0],s=lw/10,c=clr,marker='s')

    plt.tight_layout()
    return


def view_areas(plan_lines:np.array,grid,dark:bool=False,show_grid:bool=False,raster:bool=False,**kwargs) -> None:
    clr = '#444433' if not dark else '#ccccbe'
    bg = '#fffffc' if not dark else '#100804'
    hl = '#77ddaa' if not dark else '#cc0033'
    if raster == True:
        plt.figure(facecolor=bg,dpi=200)
    else:
        plt.figure(facecolor=bg)
    plt.tick_params(colors=clr) 
    plt.gca().set_facecolor(bg)
    plt.axis('equal')
    for spine in plt.gca().spines.values():
        spine.set_color(clr)
    if show_grid == True:
        plt.grid(show_grid,color=clr,alpha=0.1)
    lw = kwargs.get('linewidth',2)
    xu = np.unique(grid[:,0])
    yu = np.unique(grid[:,1])
    Z = np.zeros((len(yu), len(xu)), dtype=bool)
    ix = np.searchsorted(xu, grid[:,0])
    iy = np.searchsorted(yu, grid[:,1])
    Z[iy, ix] = True
    X, Y = np.meshgrid(xu, yu)
    plt.contourf(X, Y, Z.astype(float), levels=[0.001, 1.5],colors=hl,alpha=0.4,antialiased=True)
    #plt.scatter(X, Y,s=0.2)
    for line in plan_lines:
        plt.plot(line[:,0],line[:,1],c=clr,linewidth=lw)
    plt.tight_layout()
    return

#def view_depth
#def view_integration
#1. Angular Choice
#2. Angular Integration
#3. Connectivity
#4. Intelligibility
#5. Mean Depth