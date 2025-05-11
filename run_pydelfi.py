import numpy as np
import pydelfi.ndes as ndes
import pydelfi.delfi as delfi
import pydelfi.score as score
import pydelfi.priors as priors
import tensorflow as tf


# set up the simualtor
import subprocess
import os
import uuid

#set up mpi
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
n_procs = size
red_op = MPI.SUM
use_mpi = True


def run_external_simulator(theta, seed, env_name="BNTSmooth"):
    import subprocess
    import os
    from mpi4py import MPI

    theta_str = "[" + ",".join(map(str, theta)) + "]"
    cmd = (
        f"conda run -n {env_name} --no-capture-output "
        f"python simulate_and_save.py '{theta_str}' {seed}"
    )

    print(f"[rank {MPI.COMM_WORLD.Get_rank()}] Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")

    print(f"[rank {MPI.COMM_WORLD.Get_rank()}] stdout: {result.stdout}")
    print(f"[rank {MPI.COMM_WORLD.Get_rank()}] stderr: {result.stderr}")

    outpath = result.stdout.strip().splitlines()[-1]

    if not os.path.exists(outpath):
        raise RuntimeError(f"Simulation failed, output file not found: {outpath}\nstderr:\n{result.stderr}")

    x = np.load(outpath)
    os.remove(outpath)
    return x

#now we return back to the pydelfi script
def mpi_simulator(theta_batch, seed_batch, simulator_args=None, batch=1):
    """
    MPI-distributed simulator that runs only on the assigned rank.
    """
    # Each rank picks out its own theta
    theta = theta_batch[rank]
    seed = seed_batch[rank]

    # Your simulator must be defined such that it returns one data vector
    x = run_external_simulator(theta, seed=seed)

    # Gather results to rank 0
    x_all = comm.gather(x, root=0)

    if rank == 0:
        return np.array(x_all)
    else:
        return None  # Non-root returns nothing

#delfi simulator
def sim_4_delfi(theta, seed):
    return run_external_simulator(theta, seed, batch = 1):
    
simulator_args = None

# Fiducial parameter and setup
theta_fiducial = np.array([1.0, 1.0])
ndata = 10
h = np.abs(theta_fiducial) * 0.01
n_avg = 100  # Number of sims to average over

#make some mock data
if rank == 0:
    if not os.path.exists("data/data.npy"):
        print('computing fiducial data vector')
        data = run_external_simulator(theta_fiducial, seed=1234)
        np.save("data/data.npy", data)
        print('done computing data vector')
    else:
        print('data/data.npy already exists, skipping simulation.')
comm.barrier()

#load the data
data = np.load('data/data.npy')

#set up the prior
lower = np.array([0.5,0.5])
upper = np.array([1.5,1.5])
prior = priors.Uniform(lower, upper)

# Estimate mu (mean at fiducial)
if not os.path.exists("data/mu.npy"):
    print('computing mean data vector at fiducial')
    local_indices = np.array_split(np.arange(n_avg), size)[rank]
    local_mu_sims = [run_external_simulator(theta_fiducial, seed=1000 + i)
                     for i in local_indices]

    # Gather to root
    all_mu_sims = comm.gather(local_mu_sims, root=0)

    if rank == 0:
        # Flatten list and compute mean
        flat_mu_sims = [x for sublist in all_mu_sims for x in sublist]
        mu = np.mean(flat_mu_sims, axis=0)
        np.save("data/mu.npy", mu)
    else:
        mu = None
    comm.barrier()
    print('done computing mean data vector at fiducial')
else:
    mu = np.load("data/mu.npy")
    comm.barrier()

# Estimate dmudt via finite differences, averaged over n_avg sims each
dmudt = np.zeros((ndata, 2)) if rank == 0 else None

# Loop over parameters (alpha and beta)
if os.path.exists("data/dmudt.npy"):
    if rank == 0:
        print("data/dmudt.npy already exists, skipping derivative computation.")
    dmudt = np.load("data/dmudt.npy")
else:
    if rank == 0:
        print("computing derivatives")

    dmudt = np.zeros((ndata, 2)) if rank == 0 else None

    for j in range(2):
        theta_perturbed = theta_fiducial.copy()
        theta_perturbed[j] += h[j]

        local_indices = np.array_split(np.arange(n_avg), size)[rank]
        local_sims = [run_external_simulator(theta_perturbed, seed=2000 + j * n_avg + i)
                      for i in local_indices]

        all_sims = comm.gather(local_sims, root=0)

        if rank == 0:
            flat_sims = [x for sublist in all_sims for x in sublist]
            mu_perturbed = np.mean(flat_sims, axis=0)
            dmudt[:, j] = (mu_perturbed - mu) / h[j]

    dmudt = comm.bcast(dmudt, root=0)
    if rank == 0:
        np.save("data/dmudt.npy", dmudt)
        print("done computing derivatives")

comm.barrier()

# Estimate covariance at fiducial
if not os.path.exists("data/Cinv.npy"):
    print('computing covariance matrix at fiducial')
    n_cov = 200
    local_indices = np.array_split(np.arange(n_cov), size)[rank]
    local_sims = [run_external_simulator(theta_fiducial, seed=3000 + i)
                  for i in local_indices]

    # Gather all simulations on rank 0
    all_sims = comm.gather(local_sims, root=0)

    if rank == 0:
        # Flatten and stack simulations
        flat_sims = np.array([x for sublist in all_sims for x in sublist])
        C = np.cov(flat_sims.T)
        Cinv = np.linalg.inv(C)
        np.save("data/Cinv.npy", Cinv)
    else:
        Cinv = None

    comm.barrier()
    print('done computing covariance matrix at fiducial')
else:
    if rank == 0:
        print('data/Cinv.npy already exists, skipping covariance computation.')
        Cinv = np.load("data/Cinv.npy")
    else:
        Cinv = None
    comm.barrier()

# Broadcast Cinv to all ranks
Cinv = comm.bcast(Cinv, root=0)
comm.barrier()

# Build compressor
Compressor = score.Gaussian(ndata, 
                            theta_fiducial,
                            mu=mu, Cinv=Cinv, 
                            dmudt=dmudt.T, 
                            rank = rank, 
                            n_procs = n_procs, 
                            comm = comm, 
                            red_op = red_op)
Compressor.compute_fisher()
Finv = Compressor.Finv

# Define compressor callable for Delfi
def compressor(d, compressor_args):
    return Compressor.scoreMLE(d)

compressor_args = None

#compressed data
compressed_data = compressor(data, compressor_args)

#define ensemble of ndes
NDEs = [ndes.MixtureDensityNetwork(n_parameters=2, n_data=2, n_components=1, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=0),
       ndes.MixtureDensityNetwork(n_parameters=2, n_data=2, n_components=2, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=1),
       ndes.MixtureDensityNetwork(n_parameters=2, n_data=2, n_components=3, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=2),
       ndes.MixtureDensityNetwork(n_parameters=2, n_data=2, n_components=4, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=3),
       ndes.MixtureDensityNetwork(n_parameters=2, n_data=2, n_components=5, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=4),
       ndes.ConditionalMaskedAutoregressiveFlow(n_parameters=2, n_data=2, n_hiddens=[50,50], n_mades=5, act_fun=tf.tanh, index=5)]

#create delfi object
DelfiEnsemble = delfi.Delfi(compressed_data, prior, NDEs, 
                            Finv = Finv, 
                            theta_fiducial = theta_fiducial, 
                            rank = rank, 
                            n_procs = n_procs, 
                            comm = comm,
                            red_op = red_op,
                            param_limits = [lower, upper],
                            param_names = ['\\alpha', '\\beta'], 
                            results_dir = "delfi/",
                            input_normalization="fisher")

#fisher pretraining
DelfiEnsemble.fisher_pretraining()

#sequential neural likelihood
n_initial = 200
n_batch = 200
n_populations = 5

print ('start the delfi training')

DelfiEnsemble.sequential_training(sim_4_delfi,
                                compressor, 
                                n_initial, 
                                n_batch, 
                                n_populations, 
                                patience=20,
                                save_intermediate_posteriors=True)



#save the training loss etc
r1 = np.array([DelfiMDN.stacked_sequential_training_loss])
r2 = np.array([DelfiMDN.stacked_sequential_validation_loss])
r3 = np.array([DelfiMDN.sequential_nsims])
np.savetxt('delfi/training_loss.txt', r1)
np.savetxt('delfi/validation_loss.txt', r2)
np.savetxt('delfi/nsmis.txt', r3)


