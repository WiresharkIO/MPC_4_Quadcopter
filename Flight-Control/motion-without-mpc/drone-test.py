######################################################################################
import numpy as np
from roblib import *
######################################################################################

fig = figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
# print(ax)

lim=30
def draw_quadri(X):
    ax.clear()
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(0, 1*lim)
    l=1
    draw_quadrotor3D(ax, X, A, l*5)

def motion_dynamics(x):
    dx=array([[0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0]]).T
    return dx

"""
    x,
    y,
    z,
    phi(rotation wrt x),
    psi(rotation wrt y),
    theta(rotation wrt z),
    vr(here r- means frame of robot),
    wr(front, right, down)
"""
x = array([[0, 0, -5, 0, 0, 0, 10, 10, 0, 0, 0, 0]]).T

"""
    angles for propellers
"""
A=array([[0, 0, 0, 0]]).T

dt=0.1 # control the speed of visualization with this parameter
for t in arange(0, 5, dt):
    x=x+dt*motion_dynamics(x)
    draw_quadri(x)
    pause(0.001)
pause(1)
