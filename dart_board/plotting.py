import matplotlib.pyplot as plt
import numpy as np


def plot_chains(chain, fileout=None, tracers=0, downsample=10, labels=None):

    if chain.ndim < 3:
        print("You must include a multiple chains")
        return


    n_chains, length, n_var = chain.shape
    print(n_chains, length, n_var)

    if (labels is not None) and (len(labels) != n_var):
        print("You must provide the correct number of variable labels")
        return

    fig, ax = plt.subplots(int(n_var/2) + n_var%2, 2, figsize=(8, 0.8*n_var))
    plt.subplots_adjust(left=0.09, bottom=0.07, right=0.96, top=0.96, hspace=0)


    color = np.empty(n_chains, dtype=str)
    color[:] = 'k'
    alpha = 0.01 * np.ones(n_chains)
    if tracers > 0:
        idx = np.random.choice(n_chains, tracers, replace=False)
        color[idx] = 'r'
        alpha[idx] = 0.5

    for i in range(n_var):
        ix = int(i/2)  
        iy = i%2

        for j in range(n_chains):

            ax[ix,iy].plot(chain[j,:,i], color=color[j], alpha=alpha[j], rasterized=True)

        ax[ix,iy].set_xlim(0.0, length)
        
        # if (int(i/2) != int(n_var/2) + n_var%2) and (n_var%2!=1 or i%2!=1 or int(i/2)!=int(n_var/2)):  
        #     ax[ix,iy].set_xticklabels([]) 

        ax[ix,iy].set_xticks(np.linspace(0,length,5))
        ax[ix,iy].set_xticklabels([]) 


        # ax[ix,iy].set_xticklabels(np.linspace(0,downsample*length,5).astype('i8').astype('|S6'))

        # Add y-axis labels if provided by use 
        if labels is not None: ax[ix,iy].set_ylabel(labels[i]) 

    # plt.tight_layout()

    ax[-1,0].set_xticklabels(np.linspace(0,downsample*length,5).astype('i8').astype('|S6'))
    ax[-1,1].set_xticklabels(np.linspace(0,downsample*length,5).astype('i8').astype('|S6'))


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
