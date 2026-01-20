import os
# ignore 1 million tensorflow warnings, comment out if you run into errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from src.phosphene_model import RectangleImplant, MVGModel
from src.DSE import load_mnist, rand_model_params, fetch_dse
from src.HILO import HILOPatient, patient_from_phi_arr



def run_hilo(patient, num_duels):
    d = patient.d
    xtrain = np.empty((d * 2, num_duels), dtype='double')  # phi1/phi2 used in duels
    ctrain = np.empty((num_duels), dtype='double')  # responses
    losses = []

    pbar = tqdm(range(num_duels), unit='duels')
    for idx_duel in pbar:
        if idx_duel == 0:
            xtrain[:, idx_duel] = patient.hilo_acquisition(None, None)
        else:
            xtrain[:, idx_duel] = patient.hilo_acquisition(xtrain[:, :idx_duel], ctrain[:idx_duel])
        phi1 = xtrain[:d, idx_duel]
        phi2 = xtrain[d:, idx_duel]

        target = targets_test[np.random.randint(0, len(targets_test))]  # get a random target
        # simulate the duel
        decision, resdict = patient.duel(target, phi1, phi2)
        ctrain[idx_duel] = decision
        # update posterior
        patient.hilo_update_posterior(xtrain[:, :idx_duel + 1], ctrain[:idx_duel + 1])
        # get the current best guess for true phi
        phi_guess = patient.hilo_identify_best(xtrain[:, :idx_duel + 1], ctrain[:idx_duel + 1])

        # for example here, only evaluate on subset of test set to save time
        nsamples = 256 * 4
        dse_loss = patient.mismatch_dse.evaluate(
            x=[targets_test[:nsamples], tf.repeat(phi_guess[None, ...], nsamples, axis=0)],
            y=targets_test[:nsamples], batch_size=256, verbose=0)
        losses.append(dse_loss)
        pbar.set_description(f"loss: {dse_loss : .4f}")
    return phi_guess, losses


if __name__ == '__main__':
    # setup
    version = 'v2'  # version from paper, with bug fixed.
    np.random.seed(42)
    implant = RectangleImplant()
    model = MVGModel(xrange=(-12, 12), yrange=(-12, 12), xystep=0.5).build()
    dse = fetch_dse(model, implant, version=version)
    (targets, labels), (targets_test, labels_test) = load_mnist(model)
    phis = rand_model_params(len(targets_test), version=version)

    # run for the first patient
    phi = phis[0]
    matlab_dir = 'matlab/'
    # get a true model corresponding to the patient specified by phi
    model, implant = patient_from_phi_arr(phi, model, implant, implant_kwargs={})
    patient = HILOPatient(model, implant, dse=dse, phi_true=phi, matlab_dir=matlab_dir, version=version)

    # best_phi, losses = run_hilo(patient, 100)
    best_phi, losses = run_hilo(patient, 100)


    # view the results
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel("Duels")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.yticks([0.05, 0.1, 0.25, 1], labels =[str(i) for i in [0.05, 0.1, 0.25, 1]])
    plt.show()