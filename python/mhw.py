"""
Modified Hasegawa-Wakatani
ref: Numata, Ball, and Dewar (2007 Phys Plas)

Usage:
    mhw.py [--alpha=<alpha> --kappa=<kappa> --deltak=<deltak> --D=<D> --restart=<restart_file> --nx=<nx> --ny=<ny> --filter=<filter> --seed=<seed> --ICmode=<ICmode> --stop-time=<stop_time> --use-CFL] 

Options:
    --kappa=<kappa>            Background dx(ln n0)  [default: 1.0]
    --alpha=<alpha>            beta [default: 1.0]
    --deltak=<deltak>          delta k [default: 0.15]
    --D=<D>                    Hyperdiffusion parameter [default: 1e-4]
    --restart=<restart_file>   Restart from checkpoint
    --nx=<nx>                  x (Fourier) resolution [default: 128]
    --ny=<ny>                  y (Sin/Cos) resolution [default: 128]
    --filter=<filter>          fraction of modes to keep in ICs [default: 0.5]
    --seed=<seed>              random seed for ICs [default: None]
    --ICmode=<ICmode>          x mode to initialize [default: None]
    --stop-time=<stop_time>    simulation time to stop [default: 2.]
    --use-CFL                  use CFL condition

"""
import glob
import os
import sys
import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post
import logging
logger = logging.getLogger(__name__)

from docopt import docopt

# parse arguments
args = docopt(__doc__)

# Parameters
nx = int(args['--nx']) # resolution
ny = int(args['--ny']) # resolution
deltak = float(args['--deltak']) # length scale
Lx = 2*np.pi/deltak
Ly = 2*np.pi/deltak

kappa = float(args['--kappa'])
alpha = float(args['--alpha'])
D = float(args['--D'])

filter_frac = float(args['--filter'])

stop_time = float(args['--stop-time'])
restart = args['--restart']
seed = args['--seed']
ICmode = args['--ICmode']
CFL = args['--use-CFL']

if seed == 'None':
    seed = None
else:
    seed = int(seed)

if ICmode == 'None':
    ICmode = None
else:
    ICmode = int(ICmode)

# save data in directory named after script
data_dir = "scratch/" + sys.argv[0].split('.py')[0]
data_dir += "_kappa{0:5.02e}_alpha{1:5.02e}_D{2:5.02e}_filter{3:5.02e}_nx{4:d}_ny{5:d}_dk{6:f}".format(kappa, alpha, D, filter_frac, nx, ny, deltak)

if ICmode:
    data_dir += "_ICmode{0:d}".format(ICmode)

if CFL:
    data_dir += "_CFL"

if restart:
    restart_dirs = glob.glob(data_dir+"restart*")
    if restart_dirs:
        restart_dirs.sort()
        last = int(re.search("_restart(\d+)", restart_dirs[-1]).group(1))
        data_dir += "_restart{}".format(last+1)
    else:
        if os.path.exists(data_dir):
            data_dir += "_restart1"

if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(data_dir)):
        os.mkdir('{:s}/'.format(data_dir))

# Create bases and domain
start_init_time = time.time()

x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
domain = de.Domain([y_basis, x_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['psi','n'], time='t')
problem.parameters['α'] = alpha
problem.parameters['κ'] = kappa
problem.parameters['D'] = D
problem.parameters['Lx'] = Lx
problem.parameters['Ly'] = Ly

# construct the 2D Jacobian
problem.substitutions['J(A,B)'] = "dx(A) * dy(B) - dy(A) * dx(B)"
problem.substitutions["DLap(A)"] = "dx(dx(dx(dx(A)))) + 2*dx(dx(dy(dy(A)))) + dy(dy(dy(dy(A))))"
problem.substitutions['Avg_x(A)'] = "integ(A,'x')/Lx"
problem.substitutions['Avg_y(A)'] = "integ(A,'y')/Ly"
problem.substitutions['psi_fluct'] = "psi - Avg_y(psi)"
problem.substitutions['n_fluct'] = "n - Avg_y(n)"
problem.substitutions['zeta'] = "dx(dx(psi)) + dy(dy(psi))"

problem.add_equation("dt(zeta) - α*(psi_fluct - n_fluct) + D*DLap(zeta) = -J(psi,zeta)", condition="nx != 0")
problem.add_equation("dt(n)    - α*(psi_fluct - n_fluct) + κ*dx(psi) + D*DLap(n) = -J(psi,n)", condition="nx != 0")
problem.add_equation("n = 0", condition="nx ==0")
problem.add_equation("psi = 0", condition="nx ==0")

# Build solver
solver = problem.build_solver(de.timesteppers.MCNAB2)
#solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial conditions
if restart:
    logger.info("Restarting from time t = {0:10.5e}".format(solver.sim_time))
    solver.load_state(restart,-1)
else:
    n = solver.state['n']
    x = domain.grid(axis=1)
    y = domain.grid(axis=0)
    logger.info("initializing electron density")
    if ICmode:
        n['g'] = 1e-3 * np.sin(np.pi*y)*np.sin(ICmode*2*np.pi/Lx*x)
    else:
        shape = domain.local_grid_shape(scales=1)
        rand = np.random.RandomState(seed)
        pert =  1e-3 * rand.standard_normal(shape) #* np.sin(np.pi*y) #* (yt - y) * (y - yb)
        n['g'] = pert

if CFL:
    CFL = flow_tools.CFL(solver, initial_dt=1e-4, cadence=5, safety=0.3,
                         max_change=1.5, min_change=0.5)
    CFL.add_velocities(('dy(psi)', '-dx(psi)'))
    dt = CFL.compute_dt()
else:
    dt = 1e-4

# Integration parameters
solver.stop_sim_time = stop_time
solver.stop_wall_time = 60.*60
solver.stop_iteration = np.inf

# Analysis
analysis_tasks = []
check = solver.evaluator.add_file_handler(os.path.join(data_dir,'checkpoints'), wall_dt=3540, max_writes=50)
check.add_system(solver.state)
analysis_tasks.append(check)

snap = solver.evaluator.add_file_handler(os.path.join(data_dir,'snapshots'), sim_dt=5, max_writes=200)
snap.add_task("dx(psi)", name="u_y")
snap.add_task("-dy(psi)", name="u_x")
snap.add_task("zeta")
snap.add_task("n")
snap.add_task("zeta", name="zeta_kspace", layout='c')
snap.add_task("n", name="n_kspace", layout='c')
analysis_tasks.append(snap)

integ = solver.evaluator.add_file_handler(os.path.join(data_dir,'integrals'), sim_dt=1, max_writes=200)
integ.add_task("Avg_x(dx(psi)**2)", name='<y kin en density>_x', scales=1)
integ.add_task("Avg_x(dy(psi)**2)", name='<x kin en density>_x', scales=1)
integ.add_task("Avg_x(zeta)", name='<vorticity>_x', scales=1)
integ.add_task("Avg_x(n)", name='<n>_x', scales=1)
integ.add_task("Avg_x((n-Avg_x(n)) * (zeta - Avg_x(zeta)))", name="<n_prime zeta_prime>_x",scales=1)
analysis_tasks.append(integ)

timeseries = solver.evaluator.add_file_handler(os.path.join(data_dir,'timeseries'), sim_dt=0.1)
timeseries.add_task("Avg_y(Avg_x(dx(psi)**2 + dy(psi)**2))",name='Ekin')
timeseries.add_task("integ(integ(dy(psi)**2,'x'),'y')/(Lx*Ly)",name='E_zonal')
timeseries.add_task("Avg_y(Avg_x(dx(psi))**2 + Avg_x(dy(psi))**2)",name='E_zonal2')
#timeseries.add_task("2*κ/Pr * Avg_y(Avg_x(psi*dx(n))) - 2*Avg_y(Avg_x((dx(dx(psi)) + dy(dy(psi)))**2))", name="dEdt")
analysis_tasks.append(timeseries)

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("dx(psi)**2 + dy(psi)**2", name='Ekin')

try:
    logger.info('Starting loop')
    start_run_time = time.time()

    while solver.ok:
        solver.step(dt)
        if (solver.iteration-1) % 100 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max E_kin = %f' %flow.max('Ekin'))
        if CFL:
            dt = CFL.compute_dt()
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %f' %(end_run_time-start_run_time))


logger.info('beginning join operation')
for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)

