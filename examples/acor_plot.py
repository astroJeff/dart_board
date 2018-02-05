import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import dart_board
import pickle



from pandas import Series



file_root = sys.argv[1]
file_in = "../data/" + file_root + "_chain.obj"

if len(sys.argv) == 2:
    delay = 200 
else:
    delay = int(int(sys.argv[2]) / 100)  


chains = pickle.load(open(file_in, "rb"))
if chains.ndim == 3: 
    chains = chains[:,delay:,:]
elif chains.ndim == 4:
    chains = chains[0,:,delay:,:] 
else:
    sys.exit() 

n_chains, length, n_var = chains.shape
#chains = chains.reshape((n_chains*length, n_var))
print(chains.shape)






if file_root == 'HMXB' or file_root == 'mock_1' or file_root == 'J0513_nosfh' or file_root == 'J0513_nosfh_PT':
    var = [r'$M_1$',r'$M_2$',r'$a$',r'$e$',r'$v_k$',r'$\theta_k$',r'$\phi_k$',r'$t_b$']
elif file_root == 'LMC_HMXB' or file_root == 'mock_2' or file_root == 'mock_3' or file_root == 'J0513' or file_root == 'J0513_flatsfh' or file_root == 'J0513_PT' or file_root == 'J0513_flatsfh_PT':
    var = [r'$M_1$',r'$M_2$',r'$a$',r'$e$',r'$v_k$',r'$\theta_k$',r'$\phi_k$',r'$\alpha$',r'$\delta$',r'$t_b$']

factor = 100.0

n_var = len(var)
#fig, ax = plt.subplots(int(n_var/2), 2, figsize=(8,12))
fig = plt.figure(figsize=(4,3)) 


# Plot the zero correlation line
plt.axhline(0.0, color='k', linewidth=2, linestyle='dashed', alpha=0.5) 

N = 50

if file_root == 'HMXB':
    xmax = 10000/factor
else:
    xmax = 80000/factor

xmin = 0

for k in np.arange(n_var):
    
    
    kx = int(k%(n_var/2))
    ky = int(k/(n_var/2))
    
    # Plot the autocorrelation of the flatchain
    autocorr = np.zeros(N)
    series = Series(data=chains.reshape((n_chains*length, n_var)).T[k])
    for i in np.arange(N):
        autocorr[i] = Series.autocorr(series, lag=int(i*float(xmax-xmin)/N))

    plt.plot(np.linspace(xmin,xmax,N)*factor, autocorr, linewidth=2, label=var[k]) 
#    ax[kx,ky].plot(np.linspace(xmin,xmax,N)*factor, autocorr, color='k', linewidth=2)


        
    # Plot the autocorrelation of 10 sample chains
#    for j in np.arange(10):
#        autocorr = np.zeros(N)
#        series = Series(data=chains[j,:,k])

#        for i in np.arange(N):
#            autocorr[i] = Series.autocorr(series, lag=int(i*float(xmax-xmin)/N))

#        ax[kx,ky].plot(np.linspace(xmin,xmax,N)*factor, autocorr, color='k', alpha=0.1)

    
#    ax[kx,ky].axhline(0.0, color='k', alpha=0.3, linewidth=3)
#    ax[kx,ky].set_xlabel(r'lag (steps)')
#    ax[kx,ky].set_ylabel(r'Autocorrelation')
#    ax[kx,ky].text(8, 0.8, var[k])


plt.legend(ncol=2)
plt.xlabel('lag (steps)')
plt.ylabel('Autocorrelation')


file_out = "../figures/" + file_root + "_acor.pdf"     

plt.tight_layout()
plt.savefig(file_out) 

