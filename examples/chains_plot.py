import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import dart_board


file_root = sys.argv[1]
file_in = "../data/" + file_root + "_chain.npy"
file_out = "../figures/" + file_root + "_chains.pdf"

chains = np.load(file_in)

if chains.ndim == 4: chains = chains[0]

chains[:,:,0] = np.exp(chains[:,:,0])
chains[:,:,1] = np.exp(chains[:,:,1])
chains[:,:,2] = np.exp(chains[:,:,2])
# Use log of orbital separation
chains[:,:,2] = np.log10(chains[:,:,2])
chains[:,:,-1] = np.exp(chains[:,:,-1])


if file_root == 'HMXB' or 'mock_1' in file_root or 'J0513_nosfh' in file_root:
    var = [r'$M_1\ (M_{\odot})$',r'$M_2\ (M_{\odot})$',r'log $a\ (R_{\odot})$',r'$e$',
           r'$v_k\ ({\rm km\ s}^{-1})$', r'$\theta_k$ (rad.)',
           r'$\phi_k$ (rad.)',r'$t_b$ (Myr)']
else:
# elif file_root == 'LMC_HMXB' or 'mock_2' in file_root or 'mock_3' in file_root or file_root == 'J0513' \
#         or file_root == 'J0513_flatsfh' or file_root == 'J0513_flatsfh_PT' or file_root == 'J0513_PT' \
#         or file_root == 'J0513_low':
    var = [r'$M_1\ (M_{\odot})$',r'$M_2\ (M_{\odot})$',r'log $a\ (R_{\odot})$',r'$e$',
           r'$v_k\ ({\rm km\ s}^{-1})$',r'$\theta_k$ (rad.)',r'$\phi_k$ (rad.)',
           r'$\alpha$',r'$\delta$',r'$t_b$ (Myr)']


# delay = 20000
# ymax = 200000
delay = 0
ymax = 40000


if len(sys.argv) > 2:
    delay = int(sys.argv[2])
if len(sys.argv) > 3:
    ymax = int(sys.argv[3])


dart_board.plotting.plot_chains(chains, fileout=file_out, tracers=1, labels=var, delay=delay, ymax=ymax)
