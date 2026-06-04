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

def view_baseplan(plan_lines:np.array,dark:bool=False,show_grid:bool=True,**kwargs) -> None:
    clr = '#444433' if not dark else '#ccccbe'
    bg = '#fffffc' if not dark else '#100804'
    plt.figure(facecolor=bg)
    plt.tick_params(colors=clr) 
    plt.gca().set_facecolor(bg)
    plt.axis('equal')
    for spine in plt.gca().spines.values():
        spine.set_color(clr)
    if show_grid == True:
        plt.grid(show_grid,color=clr,alpha=0.1)
    lw = kwargs.get('linewidth',2)
    for line in plan_lines:
        plt.plot(line[:,0],line[:,1],c=clr,linewidth=lw)
    plt.tight_layout()
    return

def view_areas(plan_lines:np.array,grid,dark:bool=False,show_grid:bool=False,**kwargs) -> None:
    clr = '#444433' if not dark else '#ccccbe'
    bg = '#fffffc' if not dark else '#100804'
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
    plt.contourf(X, Y, Z.astype(float), levels=[0.5, 1.5],colors='#77ddaa',alpha=0.4,antialiased=True)
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