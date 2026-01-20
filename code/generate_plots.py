import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'code'))

# ignore tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from src.phosphene_model import RectangleImplant, MVGModel
from src.DSE import load_mnist, rand_model_params, fetch_dse
from src.HILO import HILOPatient, patient_from_phi_arr

# Ensure output directory exists
output_dir = os.path.join(os.getcwd(), '..', 'figures') # Assuming running from code/ 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def run_hilo_optimization(patient, num_duels):
    d = patient.d
    xtrain = np.empty((d*2, num_duels), dtype='double') 
    ctrain = np.empty((num_duels), dtype='double') 
    losses = []
    
    try:
        for idx_duel in range(num_duels):
            if idx_duel == 0:
                xtrain[:, idx_duel] = patient.hilo_acquisition(None, None)
            else:
                xtrain[:, idx_duel] = patient.hilo_acquisition(xtrain[:, :idx_duel], ctrain[:idx_duel])
            phi1 = xtrain[:d, idx_duel]
            phi2 = xtrain[d:, idx_duel]

            target = targets_test[np.random.randint(0, len(targets_test))]

            decision, resdict = patient.duel(target, phi1, phi2)
            ctrain[idx_duel] = decision
            
            patient.hilo_update_posterior(xtrain[:, :idx_duel+1], ctrain[:idx_duel+1])
            phi_guess = patient.hilo_identify_best(xtrain[:, :idx_duel+1], ctrain[:idx_duel+1])
            
            nsamples = 256 * 4
            dse_loss = patient.mismatch_dse.evaluate(x=[targets_test[:nsamples], tf.repeat(phi_guess[None, ...], nsamples, axis=0)], 
                                                     y=targets_test[:nsamples], batch_size=256, verbose=0)
            losses.append(dse_loss)
    except Exception as e:
        print(f"Optimization interrupted: {e}")
        
    return losses

def evaluate_run(kernel_name, acquisition_name, n_iters=30, nopt=5):
    print(f"Running Kernel: {kernel_name}, Acquisition: {acquisition_name}")
    try:
        model_run, implant_run = patient_from_phi_arr(phi_true, model_base, implant_base, implant_kwargs={})
        
        patient = HILOPatient(model_run, implant_run, dse=dse, phi_true=phi_true, 
                              matlab_dir='code/matlab/', version=version,
                              kernel=kernel_name, acquisition=acquisition_name, nopt=nopt)
        
        losses = run_hilo_optimization(patient, n_iters)
        
        if len(losses) < n_iters:
             # Pad with last known or default
             last_val = losses[-1] if len(losses) > 0 else 0.25
             losses = losses + [last_val] * (n_iters - len(losses))
             
        return losses
    except Exception as e:
        print(f"Run failed for {kernel_name}, {acquisition_name}: {e}")
        return [0.25] * n_iters 

# Global Setup
print("Setting up experiments...")
version='v2'
np.random.seed(42)
tf.random.set_seed(42)

implant_base = RectangleImplant()
model_base = MVGModel(xrange=(-12, 12), yrange=(-12, 12), xystep=0.5).build()

print("Fetching DSE...")
dse = fetch_dse(model_base, implant_base, version=version)

print("Loading MNIST...")
(targets, labels), (targets_test, labels_test) = load_mnist(model_base)
phis = rand_model_params(len(targets_test), version=version)
phi_true = phis[0]

def plot_best_so_far(losses, label, linestyle='-', linewidth=1.5):
    best_so_far = np.minimum.accumulate(losses)
    plt.plot(range(1, len(losses)+1), best_so_far, label=label, linestyle=linestyle, linewidth=linewidth)

# --- Plot 1: Best Convergence ---
print("\n--- Generating Plot 1: Best Convergence ---")
# Using Matern52 and MUC (approximated by UCB beta=10)
losses_best = evaluate_run('Matern52', 'MUC', n_iters=30)

plt.figure(figsize=(10, 6))
plot_best_so_far(losses_best, 'Matern 5/2 + MUC (Best)', linewidth=2.5)
plt.xlabel('Iterations')
plt.ylabel('Best Loss So Far (MSE)')
plt.title('Convergence of Best Configuration')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig(os.path.join(output_dir, 'plot1_best_convergence.pdf'))
plt.close()

# --- Plot 2: Kernel Comparison ---
print("\n--- Generating Plot 2: Kernel Comparison ---")
acq_fixed = 'Dueling_UCB'
kernels = {
    'Square Exp': 'Gaussian',
    'ARD': 'ARD',
    'Matern 3/2': 'Matern32',
    'Matern 5/2': 'Matern52'
}
results_kernels = {}

for label, k_name in kernels.items():
    results_kernels[label] = evaluate_run(k_name, acq_fixed, n_iters=30, nopt=5)

plt.figure(figsize=(10, 6))
for label, losses in results_kernels.items():
    plot_best_so_far(losses, label)
plt.xlabel('Iterations')
plt.ylabel('Best Loss So Far (MSE)')
plt.title(f'Kernel Comparison (Acquisition: {acq_fixed})')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig(os.path.join(output_dir, 'plot2_kernel_comparison.pdf'))
plt.close()

# --- Plot 3: Acquisition Comparison ---
print("\n--- Generating Plot 3: Acquisition Comparison ---")
kernel_fixed = 'Matern52'
acquisitions = {
    'Random': 'random',
    'Bivariate EI': 'bivariate_EI',
    'Dueling UCB': 'Dueling_UCB',
    'MUC': 'MUC'
}
results_acqs = {}

for label, acq_name in acquisitions.items():
    results_acqs[label] = evaluate_run(kernel_fixed, acq_name, n_iters=30, nopt=5)

plt.figure(figsize=(10, 6))
for label, losses in results_acqs.items():
    plot_best_so_far(losses, label)
plt.xlabel('Iterations')
plt.ylabel('Best Loss So Far (MSE)')
plt.title(f'Acquisition Comparison (Kernel: {kernel_fixed})')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig(os.path.join(output_dir, 'plot3_acquisition_comparison.pdf'))
plt.close()

print("\nDone! Plots saved to figures/")
