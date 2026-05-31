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

def view_baseplan(plan_lines:np.array,dark:bool=False,show_grid:bool=True) -> None:
    clr = '#444433' if not dark else '#ccccbe'
    bg = '#fffffc' if not dark else '#100804'
    plt.figure(figsize=(10,6),facecolor=bg)
    plt.tick_params(colors=clr) 
    plt.gca().set_facecolor(bg)
    plt.axis('equal')
    for spine in plt.gca().spines.values():
        spine.set_color(clr)
    plt.grid(show_grid,color=clr,alpha=0.1)
    for line in plan_lines:
        plt.plot(line[:,0],line[:,1],c=clr,linewidth=2)
    plt.tight_layout()
    return

def view_areas(plan_lines,grid,dark:bool=False,show_grid:bool=True) -> None:
    clr = '#444433' if not dark else '#ccccbe'
    bg = '#fffffc' if not dark else '#100804'
    plt.figure(figsize=(10,6),facecolor=bg)
    plt.tick_params(colors=clr) 
    plt.gca().set_facecolor(bg)
    plt.axis('equal')
    for spine in plt.gca().spines.values():
        spine.set_color(clr)
    plt.grid(show_grid,color=clr,alpha=0.1)
    for line in plan_lines:
        plt.plot(line[:,0],line[:,1],c=clr,linewidth=2)
    plt.colorbar()
    plt.tight_layout()
    return

#def view_depth
#def view_integration
#1. Angular Choice
#2. Angular Integration
#3. Connectivity
#4. Intelligibility
#5. Mean Depth