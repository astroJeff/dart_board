import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pickle
import dart_board


file_root = sys.argv[1]
file_in = "../data/" + file_root + "_chain.obj"
file_out = "../figures/" + file_root + "_chains.pdf"  

chains = pickle.load(open(file_in, "rb"))

if chains.ndim == 4:
    chains = chains[0,:,:,:]  

chains[:,:,0] = np.exp(chains[:,:,0])  
chains[:,:,1] = np.exp(chains[:,:,1])
chains[:,:,2] = np.exp(chains[:,:,2])
chains[:,:,-1] = np.exp(chains[:,:,-1])


if file_root == 'HMXB' or file_root == 'mock_1' or file_root == 'J0513_nosfh' or file_root == 'J0513_nosfh_PT':
    var = [r'$M_1$',r'$M_2$',r'$a$',r'$e$',r'$v_k$',r'$\theta_k$',r'$\phi_k$',r'$t_b$']
elif file_root == 'LMC_HMXB' or file_root == 'mock_2' or file_root == 'mock_3' or file_root == 'J0513' or file_root == 'J0513_flatsfh' or file_root == 'J0513_flatsfh_PT' or file_root == 'J0513_PT':
    var = [r'$M_1$',r'$M_2$',r'$a$',r'$e$',r'$v_k$',r'$\theta_k$',r'$\phi_k$',r'$\alpha$',r'$\delta$',r'$t_b$']


dart_board.plotting.plot_chains(chains, fileout=file_out, tracers=1, labels=var, delay=20000)  

