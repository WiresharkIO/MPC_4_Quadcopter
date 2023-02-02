from rockit import MultipleShooting, Ocp
from casadi import vertcat, sumsqr, vcat, vertsplit
from roblib import *

# from cflib.crazyflie.Test_My_Code import position_callback
from cflib.crazyflie.Test_My_Code import sendControl

dt       = 0.1              # time between steps in seconds (step_horizon)
N        = 10               # number of look ahead steps
Nsim     = 30               # simulation time
nx       = 8                # the system is composed of 8 states
nu       = 4                # the system has 4 control inputs
xf       = 1                # Final coordinate
yf       = 0.8              # Final coordinate
zf       = 1                # Final coordinate

k_x      = 1
k_y      = 1
k_z      = 1
k_phi    = pi/180
tau_x    = 0.8355
tau_y    = 0.7701
tau_z    = 0.5013
tau_phi  = 0.5142

ocp      = Ocp(T = N*dt)
x        = ocp.state()
y        = ocp.state()
z        = ocp.state()
phi      = ocp.state()
vx       = ocp.state()
vy       = ocp.state()
vz       = ocp.state()
vphi     = ocp.state()

ux       = ocp.control()
uy       = ocp.control()
uz       = ocp.control()
uphi     = ocp.control()

ocp.set_der(x   ,   vx*cos(phi) - vy*sin(phi))
ocp.set_der(y   ,   vx*sin(phi) + vy*cos(phi))
ocp.set_der(z   ,   vz)
ocp.set_der(phi ,   vphi)
ocp.set_der(vx  ,   (-vx + k_x*ux)/tau_x)
ocp.set_der(vy  ,   (-vy + k_y*uy)/tau_y)
ocp.set_der(vz  ,   (-vz + k_z*uz)/tau_z)
ocp.set_der(vphi,   (-vphi + k_phi*uphi)/tau_phi)

ocp.subject_to(-0.5 <= (ux    <= 0.5))
ocp.subject_to(-0.5 <= (uy    <= 0.5))
ocp.subject_to(-0.5 <= (uz    <= 0.5))
ocp.subject_to(-0.5 <= (uphi  <= 0.5))

# a point in 3D
p = vertcat(x,y,z)

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

ocp.set_value(pf, p_final) # p_final assigned to pf before solving the OCP

v_final = vertcat(0,0,0,0)
ocp.subject_to(ocp.at_tf(vx)    == 0)
ocp.subject_to(ocp.at_tf(vy)    == 0)
ocp.subject_to(ocp.at_tf(vz)    == 0)
ocp.subject_to(ocp.at_tf(vphi)  == 0)

ocp.add_objective(5*ocp.integral(sumsqr(p-pf)))
ocp.add_objective((1e-6)*ocp.integral(sumsqr(ux + uy + uz + uphi)))

# Pick a solution method: ipopt
options = {"ipopt": {"print_level": 0}}
options["expand"] = True
options["print_time"] = True
ocp.solver('ipopt', options)

ocp.method(MultipleShooting(N=N, M=2, intg='rk') )

# Set initial
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

current_X = vertcat(0,0,0,0,0,0,0,0)
ocp.set_value(X_0, current_X)

try:
    sol = ocp.solve()
except:
    ocp.show_infeasibilities(1e-6)
    sol = ocp.non_converged_solution

Sim_system_dyn = ocp._method.discrete_system(ocp)

# Log data for post-processing
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

# Simulate the MPC solving the OCP
clearance_v         = 1e-5  # should become lower if possible
clearance           = 1e-3
i = 0
intermediate_points = []
intermediate_points_index = 0
is_stuck = False
t_tot = 0

def DM2List(U):
    q=[]
    for i in vertsplit(U,1):
        q.append(i)
        # U=np.array(U.full())
    return q

while True:
    print("timestep", i + 1, "of", Nsim)
    # Combine first control inputs
    current_U = vertcat(ux_sol[0], uy_sol[0], uz_sol[0], uphi_sol[0])
    # DM2List(current_U)
    sendControl(DM2List(current_U))
    current_X = Sim_system_dyn(x0=current_X, u=current_U, T=dt)["xf"]

    t_tot = t_tot + dt
    print(f' x: {current_X[0]}')
    print(f' y: {current_X[1]}')
    print(f' z: {current_X[2]}')
    error_v = sumsqr(current_X[4:8] - v_final)
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

    # Initial guess
    ocp.set_initial(x, x_sol)
    ocp.set_initial(y, y_sol)
    ocp.set_initial(z, z_sol)
    ocp.set_initial(phi, phi_sol)
    ocp.set_initial(vx, vx_sol)
    ocp.set_initial(vy, vy_sol)
    ocp.set_initial(vz, vz_sol)
    ocp.set_initial(vphi, vphi_sol)
    print()
    i = i + 1

    print(f'Total execution time is: {t_tot}')

# print(position_callback())