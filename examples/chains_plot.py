import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import dart_board


file_name = sys.argv[1]
file_in = "../data/" + file_name + "_chain.npy"
file_out = "../figures/" + file_name + "_chains.pdf"

chains = np.load(file_in)

if chains.ndim == 4: chains = chains[0]

chains[:,:,0] = np.exp(chains[:,:,0])
chains[:,:,1] = np.exp(chains[:,:,1])
chains[:,:,2] = np.exp(chains[:,:,2])
# Use log of orbital separation
chains[:,:,2] = np.log10(chains[:,:,2])
chains[:,:,-1] = np.exp(chains[:,:,-1])


if file_name == 'HMXB' or 'mock_1' in file_name or 'J0513_nosfh' in file_name:
    var = [r'$M_1\ (M_{\odot})$',r'$M_2\ (M_{\odot})$',r'log $a\ (R_{\odot})$',r'$e$',
           r'$v_k\ ({\rm km\ s}^{-1})$', r'$\theta_k$ (rad.)',
           r'$\phi_k$ (rad.)',r'$t_b$ (Myr)']
else:
# elif file_name == 'LMC_HMXB' or 'mock_2' in file_name or 'mock_3' in file_name or file_name == 'J0513' \
#         or file_name == 'J0513_flatsfh' or file_name == 'J0513_flatsfh_PT' or file_name == 'J0513_PT' \
#         or file_name == 'J0513_low':
    var = [r'$M_1\ (M_{\odot})$',r'$M_2\ (M_{\odot})$',r'log $a\ (R_{\odot})$',r'$e$',
           r'$v_k\ ({\rm km\ s}^{-1})$',r'$\theta_k$ (rad.)',r'$\phi_k$ (rad.)',
           r'$\alpha$',r'$\delta$',r'$t_b$ (Myr)']


truths = None
if 'mock_1' in file_name:
    truths = [11.77, 8.07, np.log10(4850.81), 0.83, 153.04, 2.05, 2.33, 34.74]
elif 'mock_2' in file_name:
    truths = [14.113, 5.094, np.log10(45.12), 0.624, 141.12, 1.6982, 1.6266, 83.2554, -69.939, 21.89]
elif 'mock_3' in file_name:
    truths = [11.01, 7.42, np.log10(744.19), 0.50, 167.69, 1.79, 2.08, 83.2559, -69.9377, 36.59]




# delay = 20000
# ymax = 200000
delay = 0
ymax = 150000
thin = 100
num_xticks = 7

if len(sys.argv) > 2:
    delay = int(sys.argv[2])
if len(sys.argv) > 3:
    ymax = int(sys.argv[3])
if len(sys.argv) > 4:
    thin = int(sys.argv[4])
if len(sys.argv) > 5:
    num_xticks = int(sys.argv[5])


dart_board.plotting.plot_chains(chains, fileout=file_out, tracers=1, labels=var, delay=delay,
                                ymax=ymax, thin=thin, num_xticks=num_xticks, truths=truths)
