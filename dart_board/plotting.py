import matplotlib.pyplot as plt
import numpy as np


def plot_chains(chain, fileout=None, tracers=0, downsample=100, labels=None, delay=0):

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

            xvals = (np.arange(length)*downsample - delay) / 1000 
            ax[ix,iy].plot(xvals, chain[j,:,i], color=color[j], alpha=alpha[j], rasterized=True)


        ax[ix,iy].set_xlim(-delay/1000, (length*downsample-delay)/1000)
        
        ax[ix,iy].set_xticks(np.linspace(0,(length*downsample-delay)/1000,5))
        ax[ix,iy].set_xticklabels([]) 


        # Add y-axis labels if provided by use 
        if labels is not None: ax[ix,iy].set_ylabel(labels[i]) 
    
        ax[ix,iy].axvline(0, color='k', linestyle='dashed', linewidth=2.0)     

    # plt.tight_layout()

    ax[-1,0].set_xticklabels(np.linspace(0,(downsample*length-delay)/1000,5).astype('i8').astype('U'))
    ax[-1,1].set_xticklabels(np.linspace(0,(downsample*length-delay)/1000,5).astype('i8').astype('U'))

    ax[-1,0].set_xlabel(r'Steps ($\times$1000)') 
    ax[-1,1].set_xlabel(r'Steps ($\times$1000)') 

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
