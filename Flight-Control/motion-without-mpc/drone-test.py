######################################################################################
import numpy as np
from numpy import sin as s, cos as c, tan as t
# from drone_sim.sim.parameters import *

from roblib import *
fig = figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
print(ax)

ech=30

def draw_quadri(X):
    ax.clear()
    ax.set_xlim3d(-ech, ech)
    ax.set_ylim3d(-ech, ech)
    ax.set_zlim3d(0, 1*ech)
    l=1
    draw_quadrotor3D(ax, X, A, 5*l)

def f(x):
    dx=array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]).T
    return dx


"""
x, y, z,   
phi(rotation wrt x), psi(rotation wrt y), theta(rotation wrt z),    
vr, wr  (front, right, down)
"""
x = array([[0, 0, -5, 0, 0, 0, 10, 10, 0, 0, 0, 0]]).T

"""
angles for propellers
"""
A=array([[0, 0, 0, 0]]).T

dt=0.1

for t in arange(0, 5, dt):
    x = x + dt*f(x)
    draw_quadri(x)
    pause(0.001)
pause(1)




######################################################################################

# class Drone:
#     def __init__(self, x=0, y=0, z=0.5, enable_death=True):
#         # Position
#         self.x, self.y, self.z = x, y, z
#
#         # Roll Pitch Yaw
#         self.phi, self.theta, self.psi = 0, 0, 0
#
#         # Linear velocities
#         self.vx, self.vy, self.vz = 0, 0, 0
#
#         # Angular Velocities
#         self.p, self.q, self.r = 0, 0, 0
#
#         self.linear_position = lambda: np.array([self.x, self.y, self.z]).reshape(3, 1)
#         self.angular_position = lambda: np.array([self.phi, self.theta, self.psi]).reshape(3, 1)
#         self.linear_velocity = lambda: np.array([self.vx, self.vy, self.vz]).reshape(3, 1)
#         self.angular_velocity = lambda: np.array([self.p, self.q, self.r]).reshape(3, 1)

######################################################################################



