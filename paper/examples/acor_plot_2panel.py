import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import dart_board
import pickle



from pandas import Series



file_root_1 = sys.argv[1]
file_in_1 = "../data/" + file_root_1 + "_chain.npy"

file_root_2 = sys.argv[2]
file_in_2 = "../data/" + file_root_2 + "_chain.npy"


# Delay for first data set
if len(sys.argv) < 4:
    delay_1 = 20000
else:
    delay_1 = int(sys.argv[3])

# Delay for second data set
if len(sys.argv) < 5:
    delay_2 = 200000
else:
    delay_2 = int(sys.argv[4])

# Thin for first data set
if len(sys.argv) < 6:
    thin_1 = 100
else:
    thin_1 = int(sys.argv[5])

# Thin for second data set
if len(sys.argv) < 7:
    thin_2 = 1000
else:
    thin_2 = int(sys.argv[5])


# Adjust delay for chain thinning
delay_1 = int(delay_1 / thin_1)
delay_2 = int(delay_2 / thin_2)



chains_1 = np.load(file_in_1)
if chains_1.ndim == 3:
    chains_1 = chains_1[:,delay_1:,:]
elif chains_1.ndim == 4:
    chains_1 = chains[0,:,delay_1:,:]
else:
    sys.exit()

chains_2 = np.load(file_in_2)
if chains_2.ndim == 3:
    chains_2 = chains_2[:,delay_2:,:]
elif chains_2.ndim == 4:
    chains_2 = chains_2[0,:,delay_2:,:]
else:
    sys.exit()



n_chains_1, length_1, n_var_1 = chains_1.shape
#chains = chains.reshape((n_chains*length, n_var))
print(chains_1.shape)

n_chains_2, length_2, n_var_2 = chains_2.shape
print(chains_2.shape)





if file_root_1 == 'HMXB' or file_root_1 == 'mock_1' or 'J0513_nosfh' in file_root_1:
    var_1 = [r'$M_1$',r'$M_2$',r'$a$',r'$e$',r'$v_k$',r'$\theta_k$',r'$\phi_k$',r'$t_b$']
elif file_root_1 == 'LMC_HMXB' or 'mock_2' in file_root_1 or 'mock_3' in file_root_1 or 'J0513' in file_root_1:
    var_1 = [r'$M_1$',r'$M_2$',r'$a$',r'$e$',r'$v_k$',r'$\theta_k$',r'$\phi_k$',r'$\alpha$',r'$\delta$',r'$t_b$']


if file_root_2 == 'HMXB' or file_root_2 == 'mock_1' or 'J0513_nosfh' in file_root_2:
    var_2 = [r'$M_1$',r'$M_2$',r'$a$',r'$e$',r'$v_k$',r'$\theta_k$',r'$\phi_k$',r'$t_b$']
elif file_root_2 == 'LMC_HMXB' or 'mock_2' in file_root_2 or 'mock_3' in file_root_2 or 'J0513' in file_root_2:
    var_2 = [r'$M_1$',r'$M_2$',r'$a$',r'$e$',r'$v_k$',r'$\theta_k$',r'$\phi_k$',r'$\alpha$',r'$\delta$',r'$t_b$']




n_var_1 = len(var_1)
n_var_2 = len(var_2)
fig, ax = plt.subplots(2, 1, figsize=(4,6))
# fig = plt.figure(figsize=(4,3))


# Plot the zero correlation line
for a in ax:
    a.axhline(0.0, color='k', linewidth=2, linestyle='dashed', alpha=0.5)

N = 50

if file_root_1 == 'HMXB':
    xmax_1 = 20000/thin_1
else:
    xmax_1 = 200000/thin_1

if file_root_2 == 'HMXB':
    xmax_2 = 20000/thin_2
else:
    xmax_2 = 200000/thin_2


xmin = 0

for k in np.arange(n_var_1):

    kx = int(k%(n_var_1/2))
    ky = int(k/(n_var_1/2))

    # Plot the autocorrelation of the flatchain
    autocorr = np.zeros(N)
    series = Series(data=chains_1.reshape((n_chains_1*length_1, n_var_1)).T[k])
    for i in np.arange(N):
        autocorr[i] = Series.autocorr(series, lag=int(i*float(xmax_1-xmin)/N))

    ax[0].plot(np.linspace(xmin,xmax_1,N)*thin_1/1000, autocorr, linewidth=2, label=var_1[k])


for k in np.arange(n_var_2):

    kx = int(k%(n_var_2/2))
    ky = int(k/(n_var_2/2))

    # Plot the autocorrelation of the flatchain
    autocorr = np.zeros(N)
    series = Series(data=chains_2.reshape((n_chains_2*length_2, n_var_2)).T[k])
    for i in np.arange(N):
        autocorr[i] = Series.autocorr(series, lag=int(i*float(xmax_2-xmin)/N))

    ax[1].plot(np.linspace(xmin,xmax_2,N)*thin_2/1000, autocorr, linewidth=2, label=var_2[k])



for a in ax:
    a.legend(ncol=2)
    a.set_xlabel(r'lag (steps $\times$1000)')
    a.set_ylabel('Autocorrelation')

ax[0].set_title(file_root_1)
if file_root_2 == 'J0513_low':
    ax[1].set_title(r'J0513 - Low $a$ Solution')
elif file_root_2 == 'J0513_high':
    ax[1].set_title(r'J0513 - High $a$ Solution')
else:
    ax[1].set_title(file_root_2)

file_out = "../figures/" + file_root_1 + "_" + file_root_2 + "_acor.pdf"

plt.tight_layout()
plt.savefig(file_out)
