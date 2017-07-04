import matplotlib.pyplot as plt
import numpy as np


def plot_chains(chain, fileout=None, tracers=0):

    if chain.ndim < 3:
        print("You must include a multiple chains")
        return


    n_chains, length, n_var = chain.shape
    print(n_chains, length, n_var)

    fig, ax = plt.subplots(n_var, 1, figsize=(8, 2*n_var))


    color = np.empty(n_chains, dtype=str)
    color[:] = 'k'
    alpha = 0.01 * np.ones(n_chains)
    if tracers > 0:
        idx = np.random.choice(n_chains, tracers, replace=False)
        color[idx] = 'r'
        alpha[idx] = 0.5

    for i in range(n_var):
        for j in range(n_chains):

            ax[i].plot(chain[j,:,i], color=color[j], alpha=alpha[j], rasterized=True)

        ax[i].set_xlim(0.0, length)

    plt.tight_layout()


    if fileout is None:
        plt.show()
    else:
        plt.savefig(fileout, rasterized=True)


    return



# from dart_board import plotting
# import numpy as np
# import pickle
# chains = pickle.load(open("../data/HMXB_chain.obj", "rb"))
# plotting.plot_chains(chains)
