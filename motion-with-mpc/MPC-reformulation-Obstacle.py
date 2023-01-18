from rockit import *
from rockit import FreeTime, MultipleShooting, Ocp
import numpy as np
from numpy import pi, cos, sin, tan, square
from casadi import vertcat, horzcat, sumsqr, Function, exp, vcat
import matplotlib.pyplot as plt

dt       = 0.1              # time between steps in seconds (step_horizon)
N        = 10               # number of look ahead steps
Nsim     = 25               # simulation time
nx       = 8                # the system is composed of 8 states
nu       = 4                # the system has 4 control inputs

xf       = 1                # Final coordinate
yf       = 0.8              # Final coordinate
zf       = 1                # Final coordinate

# Logging variables
x_hist         = np.zeros((Nsim+1, N+1))
y_hist         = np.zeros((Nsim+1, N+1))
z_hist         = np.zeros((Nsim+1, N+1))
phi_hist       = np.zeros((Nsim+1, N+1))

vx_hist        = np.zeros((Nsim+1, N+1))
vy_hist        = np.zeros((Nsim+1, N+1))
vz_hist        = np.zeros((Nsim+1, N+1))
vphi_hist      = np.zeros((Nsim+1, N+1))

ux_hist         = np.zeros((Nsim+1, N+1))
uy_hist         = np.zeros((Nsim+1, N+1))
uz_hist         = np.zeros((Nsim+1, N+1))
uphi_hist       = np.zeros((Nsim+1, N+1))

# Drone model from reference paper
# Model constants
# TO-DO change according to Crazy-Flie
k_x      = 1
k_y      = 1
k_z      = 1
k_phi    = pi/180
tau_x    = 0.8355
tau_y    = 0.7701
tau_z    = 0.5013
tau_phi  = 0.5142

# Initializing OCP
ocp = Ocp(T = N*dt)

# Define states - a reduced state vector - x = [x y z phi vx vy vz vphi ]T
# From reference paper
x        = ocp.state()
y        = ocp.state()
z        = ocp.state()
phi      = ocp.state()
vx       = ocp.state()
vy       = ocp.state()
vz       = ocp.state()
vphi     = ocp.state()

# From reference paper - Define controls
ux       = ocp.control()
uy       = ocp.control()
uz       = ocp.control()
uphi     = ocp.control()

# Specification of the ODEs - a nonlinear model dx = f(x, u) defined by the
# set of equations: From reference paper
ocp.set_der(x   ,   vx*cos(phi) - vy*sin(phi))
ocp.set_der(y   ,   vx*sin(phi) + vy*cos(phi))
ocp.set_der(z   ,   vz)
ocp.set_der(phi ,   vphi)
ocp.set_der(vx  ,   (-vx + k_x*ux)/tau_x)
ocp.set_der(vy  ,   (-vy + k_y*uy)/tau_y)
ocp.set_der(vz  ,   (-vz + k_z*uz)/tau_z)
ocp.set_der(vphi,   (-vphi + k_phi*uphi)/tau_phi)

# Control constraints
ocp.subject_to(-1 <= (ux    <= 1))
ocp.subject_to(-1 <= (uy    <= 1))
ocp.subject_to(-1 <= (uz    <= 1))
ocp.subject_to(-1 <= (uphi  <= 1))

# Adding obstacles - Physical structure round - ADDED and alterable
p0 = ocp.parameter(2)
x0, y0 = 0.5, 0.5
p0_coord = vertcat(x0,y0)
ocp.set_value(p0, p0_coord)
r0 = 0.1

p1 = ocp.parameter(2)
x1, y1 = 0.8, 0.2
p1_coord = vertcat(x1,y1)
ocp.set_value(p1, p1_coord)
r1 = 0.1

p2 = ocp.parameter(2)
x2, y2 = 0.3, 0.5
p2_coord = vertcat(x2,y2)
ocp.set_value(p2, p2_coord)
r2 = 0.1

# a point in 3D
p = vertcat(x,y,z)

# Obstacle avoidance hard constraints ADDED and alterable
ocp.subject_to( sumsqr(p[0:2] - p0)  >  (r0)**2 )
ocp.subject_to( sumsqr(p[0:2] - p1)  >  (r1)**2 )
ocp.subject_to( sumsqr(p[0:2] - p2)  >  (r2)**2 )

# Define initial parameter
X_0 = ocp.parameter(nx)
X = vertcat(x, y, z, phi, vx, vy, vz, vphi)

# Initial point
ocp.subject_to(ocp.at_t0(X) == X_0)
ocp.subject_to( 0  <=  (x    <= 1))
ocp.subject_to( 0  <=  (y    <= 1))
ocp.subject_to( 0  <=  (z    <= 1))

# reach end point
pf = ocp.parameter(3)
# end point
p_final = vertcat(xf,yf,zf)

"""
Set a value for a parameter
All variables must be given a value before an optimal control problem can be solved.
"""
ocp.set_value(pf, p_final) # p_final assigned to pf before solving the OCP

#---------------- constraints on velocity ---------------------------------
v_final = vertcat(0,0,0,0)
ocp.subject_to(ocp.at_tf(vx) == 0)
ocp.subject_to(ocp.at_tf(vy) == 0)
ocp.subject_to(ocp.at_tf(vz) == 0)
ocp.subject_to(ocp.at_tf(vphi) == 0)

# Objective Function
# >>> ocp.add_objective( ocp.integral(x) ) # Lagrange term
# sumsqr - used to calculate sum of squares
ocp.add_objective(5*ocp.integral(sumsqr(p-pf)))
ocp.add_objective((1e-6)*ocp.integral(sumsqr(ux + uy + uz + uphi)))

#-------------------------  Pick a solution method: ipopt --------------------
options = {"ipopt": {"print_level": 0}}
options["expand"] = True
options["print_time"] = True
ocp.solver('ipopt', options)

#-------------------------- try other solvers here -------------------
# Multiple Shooting
ocp.method(MultipleShooting(N=N, M=2, intg='rk') )

#-------------------- Set initial-----------------
ux_init     = np.ones(N)
uy_init     = np.ones(N)
uz_init     = np.zeros(N)
uphi_init   = np.zeros(N)

vx_init         = np.empty(N)
vx_init[0]      = 0
vy_init         = np.empty(N)
vy_init[0]      = 0
vz_init         = np.empty(N)
vz_init[0]      = 0
vphi_init       = np.empty(N)
vphi_init[0]    = 0

x_init      = np.empty(N)
x_init[0]   = 0
y_init      = np.empty(N)
y_init[0]   = 0
z_init      = np.empty(N)
z_init[0]   = 0
phi_init    = np.empty(N)
phi_init[0] = 0

for i in range(1,N):
    vx_init[i]   = vx_init[i-1] + ux_init[i-1]*dt
    vy_init[i]   = vy_init[i-1] + uy_init[i-1]*dt
    vz_init[i]   = vz_init[i-1] + uz_init[i-1]*dt
    vphi_init[i] = vphi_init[i-1] + uphi_init[i-1]*dt

    phi_init[i] = phi_init[i-1] + vphi_init[i-1]*dt
    z_init[i]   = z_init[i-1] + vz_init[i-1]*dt
    x_init[i]   = x_init[i-1] + ((vx_init[i-1]*cos(phi_init[i-1])) - (vy_init[i-1]*sin(phi_init[i-1])))*dt
    y_init[i]   = y_init[i-1] + ((vx_init[i-1]*sin(phi_init[i-1])) + (vy_init[i-1]*cos(phi_init[i-1])))*dt

ocp.set_initial(x, x_init)
ocp.set_initial(y, y_init)
ocp.set_initial(z, z_init)
ocp.set_initial(phi, phi_init)
ocp.set_initial(vx, vx_init)
ocp.set_initial(vy, vy_init)
ocp.set_initial(vz, vz_init)
ocp.set_initial(vphi, vphi_init)

ocp.set_initial(ux, ux_init)
ocp.set_initial(uy, uy_init)
ocp.set_initial(uz, uz_init)
ocp.set_initial(uphi, uphi_init)

#---------------- Solve the OCP for the first time step--------------------
# First waypoint is current position
index_closest_point = 0
current_X = vertcat(0,0,0,0,0,0,0,0)
ocp.set_value(X_0, current_X)

# Solve the optimization problem
try:
    sol = ocp.solve()
except:
    ocp.show_infeasibilities(1e-6)
    sol = ocp.non_converged_solution

# Get discretized dynamics as CasADi function to simulate the system
Sim_system_dyn = ocp._method.discrete_system(ocp)

# ----------------------- Log data for post-processing---------------------
t_sol, x_sol    = sol.sample(x, grid='control')
t_sol, y_sol    = sol.sample(y, grid='control')
t_sol, z_sol    = sol.sample(z, grid='control')
t_sol, phi_sol  = sol.sample(phi, grid='control')
t_sol, vx_sol   = sol.sample(vx, grid='control')
t_sol, vy_sol   = sol.sample(vy, grid='control')
t_sol, vz_sol   = sol.sample(vz, grid='control')
t_sol, vphi_sol = sol.sample(vphi, grid='control')

t_sol, ux_sol   = sol.sample(ux, grid='control')
t_sol, uy_sol   = sol.sample(uy, grid='control')
t_sol, uz_sol   = sol.sample(uz, grid='control')
t_sol, uphi_sol = sol.sample(uphi, grid='control')

x_hist[0, :] = x_sol
y_hist[0, :] = y_sol
z_hist[0, :] = z_sol
phi_hist[0, :] = phi_sol
vx_hist[0, :] = vx_sol
vy_hist[0, :] = vy_sol
vz_hist[0, :] = vz_sol
vphi_hist[0, :] = vphi_sol

print(current_X[0])
print(current_X[1])
print(current_X[2])

# plot function
# DISABLED CALLING FOR NOW - ENABLE CALLING FOR PLOT
def plotxy(p0_coord, p1_coord, p2_coord, opt, x_sol, y_sol):
    # x-y plot
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)

    plt.xlabel('x pos [m]')
    plt.ylabel('y pos [m]')
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.title('solution in x,y')
    ax.set_aspect('equal', adjustable='box')

    ts = np.linspace(0, 2 * pi, 1000)
    plt.plot(p0_coord[0] + r0 * cos(ts), p0_coord[1] + r0 * sin(ts), 'r-')
    plt.plot(p0_coord[0] + r0 * cos(ts), p0_coord[1] + r0 * sin(ts), 'r-')
    plt.plot(p1_coord[0] + r1 * cos(ts), p1_coord[1] + r1 * sin(ts), 'b-')
    plt.plot(p2_coord[0] + r2 * cos(ts), p2_coord[1] + r2 * sin(ts), 'g-')
    plt.plot(xf, yf, 'ro', markersize=10)

    if opt == 1:
        plt.plot(x_sol, y_sol, 'go')
        plt.plot(x_hist[:, 0], y_hist[:, 0], 'bo', markersize=3)
    else:
        plt.plot(x_hist[:, 0], y_hist[:, 0], 'bo', markersize=3)
    plt.show(block=True)

# Simulate the MPC solving the OCP
clearance_v         = 1e-5  # should become lower if possible
clearance           = 1e-3
local_min_clearance = 1e-1
i = 0

obs_hist_0  = np.zeros((Nsim+1, 3))
obs_hist_1  = np.zeros((Nsim+1, 3))
obs_hist_2  = np.zeros((Nsim+1, 3))

intermediate_points = []
intermediate_points_required = False
new_path_not_needed = False
intermediate_points_index = 0
is_stuck = False
t_tot = 0

while True:
    print("timestep", i + 1, "of", Nsim)
    plotxy(p0_coord, p1_coord, p2_coord, 1, x_sol, y_sol)
    ux_hist[i, :]   = ux_sol
    uy_hist[i, :]   = uy_sol
    uz_hist[i, :]   = uz_sol
    uphi_hist[i, :] = uphi_sol

    # Combine first control inputs
    current_U = vertcat(ux_sol[0], uy_sol[0], uz_sol[0], uphi_sol[0])

    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_system_dyn(x0=current_X, u=current_U, T=dt)["xf"]
    t_tot = t_tot + dt

    obs_hist_0[i, 0] = x0
    obs_hist_0[i, 1] = y0
    obs_hist_0[i, 2] = r0

    obs_hist_1[i, 0] = x1
    obs_hist_1[i, 1] = y1
    obs_hist_1[i, 2] = r1

    obs_hist_2[i, 0] = x2
    obs_hist_2[i, 1] = y2
    obs_hist_2[i, 2] = r2

    print(f' x: {current_X[0]}')
    print(f' y: {current_X[1]}')
    print(f' z: {current_X[2]}')

    if (sumsqr(current_X[0:2] - p0_coord) - r0 ** 2) > 0:
        print('outside obs 1')
    else:
        print('Problem! inside obs 1')
    if (sumsqr(current_X[0:2] - p1_coord) - r1 ** 2) > 0:
        print('outside obs 2')
    else:
        print('Problem! inside obs 2')
    if (sumsqr(current_X[0:2] - p2_coord) - r2 ** 2) > 0:
        print('outside obs 3')
    else:
        print('Problem! inside obs 3')

    error_v = sumsqr(current_X[4:8] - v_final)

    if intermediate_points_required:
        error = sumsqr(current_X[0:3] - intermediate_points[intermediate_points_index - 1])
    else:
        error = sumsqr(current_X[0:3] - p_final)

    if is_stuck or i == Nsim:
        break

    if intermediate_points_index == len(intermediate_points):  # going to end goal
        clearance = 1e-3
    else:
        clearance = 1e-2

    if error < clearance:
        if intermediate_points_index == len(intermediate_points):
            print('Location reached, now reducing velocity to zero')
            if error_v < clearance_v:
                print('Desired goal reached!')
                break
        else:
            print('Intermediate point reached! Diverting to next point.')
            intermediate_points_index = intermediate_points_index + 1
            ocp.set_value(pf, vcat(intermediate_points[intermediate_points_index - 1]))

    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X)

    # Solve the optimization problem
    try:
        sol = ocp.solve()
    except:
        ocp.show_infeasibilities(1e-6)
        sol = ocp.non_converged_solution
        break

    # Log data for post-processing
    t_sol, x_sol = sol.sample(x, grid='control')
    t_sol, y_sol = sol.sample(y, grid='control')
    t_sol, z_sol = sol.sample(z, grid='control')
    t_sol, phi_sol = sol.sample(phi, grid='control')
    t_sol, vx_sol = sol.sample(vx, grid='control')
    t_sol, vy_sol = sol.sample(vy, grid='control')
    t_sol, vz_sol = sol.sample(vz, grid='control')
    t_sol, vphi_sol = sol.sample(vphi, grid='control')

    t_sol, ux_sol = sol.sample(ux, grid='control')
    t_sol, uy_sol = sol.sample(uy, grid='control')
    t_sol, uz_sol = sol.sample(uz, grid='control')
    t_sol, uphi_sol = sol.sample(uphi, grid='control')

    x_hist[i + 1, :] = x_sol
    y_hist[i + 1, :] = y_sol
    z_hist[i + 1, :] = z_sol
    phi_hist[i + 1, :] = phi_sol
    vx_hist[i + 1, :] = vx_sol
    vy_hist[i + 1, :] = vy_sol
    vz_hist[i + 1, :] = vz_sol
    vphi_hist[i + 1, :] = vphi_sol

    # Initial guess
    ocp.set_initial(x, x_sol)
    ocp.set_initial(y, y_sol)
    ocp.set_initial(z, z_sol)
    ocp.set_initial(phi, phi_sol)
    ocp.set_initial(vx, vx_sol)
    ocp.set_initial(vy, vy_sol)
    ocp.set_initial(vz, vz_sol)
    ocp.set_initial(vphi, vphi_sol)
    i = i + 1

    # Results
    print(f'Total execution time is: {t_tot}')
    timestep = np.linspace(0, t_tot, len(ux_hist[0:i, 0]))