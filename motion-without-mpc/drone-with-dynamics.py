######################################################################################
import numpy as np
from roblib import *
######################################################################################

fig = figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

lim=30
def draw_quadri(X):
    ax.clear()
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(0, 1*lim)
    draw_quadrotor3D(ax, X, A, l*5)

def motion_dynamics(X,W):
    # beta - drag coeff.
    thrust_constant=array([[beta, beta, beta, beta],
                           [-beta*l, 0, beta*l, 0],
                           [0, -beta*l, 0, beta*l],
                           [-delta, delta, -delta, delta]
                           ])
    X=X.flatten()

    # state vector
    x, y, z, phi, theta, psi=list(X[0:6])
    vr=(X[6:9]).reshape(3, 1)
    wr=(X[9:12]).reshape(3, 1)

    W2=W*abs(W)

    # Force and Torque is stored in tow
    tow=thrust_constant@W2.flatten()
    E=eulermat(phi, theta, psi)
    dvr=-adjoint(wr)@vr+inv(E)@array([[0], [0], [g]])+array([[0], [0], [-tow[0]/m]])
    dp=E@vr
    dangles=eulerderivative(phi, theta, psi)@wr
    dwr=inv(I)@(-adjoint(wr)@I@wr+tow[1:4].reshape(3, 1))
    dX=vstack((dp, dangles, dvr, dwr))
    return dX

"""
    Propeller turn rates, 4 propellers so 4 values
"""
def control(X):
    return array([[6], [5], [5], [5]])

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

"""
    Inertial Matrix
"""
I=array([[10, 0, 0], [0, 10, 0], [0, 0, 20]])

"""
    constants
"""
m, g, beta, delta, l=10, 9.81, 2, 1, 1

dt=0.01 # control the speed of visualization with this parameter
for t in arange(0, 5, dt):
    w=control(x)
    x=x+dt*motion_dynamics(x, w)
    A=A+dt*w
    draw_quadri(x)
    pause(0.001)
pause(1)
