import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from .utils import A_to_P, P_to_A



N_times = 500
# Colors
C0 = 'C0'
C1 = 'C1'



def func_Roche_radius(M1, M2, A):
    """ Get Roche lobe radius (Eggleton 1983)

    Parameters
    ----------
    M1 : float
        Primary mass (Msun)
    M2 : float
        Secondary mass (Msun)
    A : float
        Orbital separation (any unit)

    Returns
    -------
    Roche radius : float
        in units of input, A
    """
    q = M1 / M2
    return A * 0.49*q**(2.0/3.0) / (0.6*q**(2.0/3.0) + np.log(1.0 + q**(1.0/3.0)))



def evolve_binary(evolve, M1_in, M2_in, P_orb_in, ecc, t_min, t_max,
                  v1_kick=(0.0, 0.0, 0.0), v2_kick=(0.0, 0.0, 0.0),
                  metallicity=0.02, verbose_output=False, model_kwargs={}):

    times = np.linspace(t_min, t_max, N_times)

    R1_out = np.array([])
    R2_out = np.array([])

    M1_out = np.array([])
    M2_out = np.array([])

    Teff1_out = np.array([])
    Teff2_out = np.array([])

    Mdot1_out = np.array([])
    Mdot2_out = np.array([])

    P_orb_out = np.array([])
    ecc_out = np.array([])

    k1_out = np.array([], dtype='i8')
    k2_out = np.array([], dtype='i8')

    L1_out = np.array([])
    L2_out = np.array([])

    for time in times:

        output = evolve(M1_in, M2_in, P_orb_in, ecc,
                        v1_kick[0], v1_kick[1], v1_kick[2],
                        v2_kick[0], v2_kick[1], v2_kick[2],
                        time, metallicity, verbose_output,
                        **model_kwargs)

        R1_out = np.append(R1_out, output[9])
        R2_out = np.append(R2_out, output[10])

        M1_out = np.append(M1_out, output[0])
        M2_out = np.append(M2_out, output[1])

        Teff1_out = np.append(Teff1_out, output[11])
        Teff2_out = np.append(Teff2_out, output[12])

        Mdot1_out = np.append(Mdot1_out, output[5])
        Mdot2_out = np.append(Mdot2_out, output[6])

        P_orb_tmp = A_to_P(output[0], output[1], output[2])
        P_orb_out = np.append(P_orb_out, P_orb_tmp)
        ecc_out = np.append(ecc_out, output[3])

        k1_out = np.append(k1_out, output[15])
        k2_out = np.append(k2_out, output[16])

        L1_out = np.append(L1_out, output[13])
        L2_out = np.append(L2_out, output[14])


    return times, R1_out, R2_out, M1_out, M2_out, Teff1_out, Teff2_out, Mdot1_out, Mdot2_out, \
                P_orb_out, ecc_out, L1_out, L2_out, k1_out, k2_out


def plot_k_type(ax_1, ax_2, ax_k_type_list, times, k1_out, k2_out):


    k_type_colors = ['plum', 'sandybrown', 'lightseagreen', 'moccasin', 'chartreuse',
                     'deepskyblue', 'gold', 'rosybrown', 'm', 'darkgreen', 'grey',
                     'sienna', 'palevioletred', 'navy', 'tan', 'black']
    k_type = ['MS conv', 'MS', 'HG', 'GB', 'CHeB', 'EAGB', 'TPAGB', 'HeMS', 'HeHG',
              'HeGB', 'HeWD', 'COWD', 'ONeWD', 'NS', 'BH', 'no remnant']

    # k-type plots
    for a in [ax_1, ax_2]:

        a.axis('off')

        for j,k in enumerate(np.unique(k1_out)):
            a.fill_between(times[k1_out == k], 0.7, 0.9, color=k_type_colors[k])
        for j,k in enumerate(np.unique(k2_out)):
            a.fill_between(times[k2_out == k], 0.4, 0.6, color=k_type_colors[k])

        a.set_title('Stellar Type')

        a.set_xlim(np.min(times), np.max(times))



    # Add legend
    ax_k_type_list.axis('off')
    k_type_all = np.unique(np.array([k1_out, k2_out]))
    patches = [ mpatches.Patch(color=k_type_colors[k], label=k_type[k]) for k in k_type_all ]
    leg = ax_k_type_list.legend(handles=patches, mode='expand', ncol=len(k_type_all))
    leg.get_frame().set_alpha(0.0)



def plot_radius(ax, times, R1_out, R2_out, M1_out, M2_out, P_orb_out, ecc_out, sys_obs):

    # Radius
    ax.plot(times, R1_out, color=C0)
    ax.plot(times, R2_out, color=C1)

    # Roche radii - at periastron
    A_out = P_to_A(M1_out, M2_out, P_orb_out)
    R1_Roche = func_Roche_radius(M1_out, M2_out, A_out*(1.0-ecc_out))
    R2_Roche = func_Roche_radius(M2_out, M1_out, A_out*(1.0-ecc_out))
    ax.plot(times, R1_Roche, color=C0, linestyle='--')
    ax.plot(times, R2_Roche, color=C1, linestyle='--')


    for key, value in sys_obs.items():
        if key == 'R1': ax.axhline(value, color=C0, linestyle='dashed')
        if key == 'R2': ax.axhline(value, color=C1, linestyle='dashed')


#     ax[0].axhline(16.4, color=C0, linestyle='dashed')
#     ax[0].axhline(21.0, color=C1, linestyle='dashed')

    ax.set_yscale('log')
    ax.set_ylim(0.5, ax.get_ylim()[1])
#     ax.set_ylim(0.5, 20.0)

    ax.set_ylabel(r'Radius ($R_{\odot}$)')

    ax.set_xlim(np.min(times), np.max(times))
    # ax.set_xlabel("Time (Myr)")

#     if title is not None: ax[2].set_title(title)


def plot_mass(ax, times, M1_out, M2_out, sys_obs):

    # Mass
    ax.plot(times, M1_out, color=C0)
    ax.plot(times, M2_out, color=C1)

    for key, value in sys_obs.items():
        if key == 'M1': ax.axhline(value, color=C0, linestyle='dashed')
        if key == 'M2': ax.axhline(value, color=C1, linestyle='dashed')

#     ax[1].axhline(3.26, color=C0, linestyle='dashed')
#     ax[1].axhline(1.91, color=C1, linestyle='dashed')

    ax.set_ylabel(r'Mass ($M_{\odot}$)')
    ax.set_xlim(np.min(times), np.max(times))
    # ax.set_xlabel("Time (Myr)")


def plot_Teff(ax, times, Teff1_out, Teff2_out, sys_obs):



    # Effective temperature
    ax.plot(times, Teff1_out, color=C0)
    ax.plot(times, Teff2_out, color=C1)

    for key, value in sys_obs.items():
        if key == 'T1': ax.axhline(value, color=C0, linestyle='dashed')
        if key == 'T2': ax.axhline(value, color=C1, linestyle='dashed')

#     ax[2].axhline(5210, color=C0, linestyle='dashed')
#     ax[2].axhline(4470, color=C1, linestyle='dashed')

    ax.set_yscale('log')
    ax.set_ylabel(r'T$_{\rm eff}$ (K)')
    ax.set_xlim(np.min(times), np.max(times))
    ax.set_xlabel("Time (Myr)")


def plot_Mdot(ax, times, Mdot1_out, Mdot2_out):

    # Mass accretion rate
    ax.plot(times, np.log10(np.clip(Mdot1_out, 1.0e-16, None)), color=C0)
    ax.plot(times, np.log10(np.clip(Mdot2_out, 1.0e-16, None)), color=C1)
    ax.set_ylabel(r'Mass Accretion Rate ($M_{\odot}$ yr$^{-1}$)')

    ax.set_ylim(-14, ax.get_ylim()[1])
    ax.set_xlim(np.min(times), np.max(times))
    # ax.set_xlabel("Time (Myr)")


def plot_P_orb(ax, times, P_orb_out, t_max, sys_obs):

    ax.plot(times, P_orb_out, color='k')

    for key, value in sys_obs.items():
        if key == 'P_orb': ax.axhline(value, color='k', linestyle='dashed')

    ax.set_xlim(np.min(times), t_max)

    ax.set_ylabel(r'P$_{\rm orb}$ (days)')
    ax.set_yscale('log')
    # ax.set_xlabel("Time (Myr)")


def plot_ecc(ax, times, ecc_out, t_max, sys_obs):

    ax.plot(times, ecc_out, color='k')

    for key, value in sys_obs.items():
        if key == 'ecc': ax.axhline(value, color='k', linestyle='dashed')


    ax.set_xlim(np.min(times), t_max)

    ax.set_ylabel('Eccentricity')
    ax.set_xlabel("Time (Myr)")


def plot_HR_diagram(ax, L1_out, L2_out, Teff1_out, Teff2_out):

    ax.plot(Teff1_out, L1_out, color=C0)
    ax.plot(Teff2_out, L2_out, color=C1)


    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(40000, 2000)

    ax.set_xlabel(r"log T$_{\rm eff}$ (K)")
    ax.set_ylabel("log L (erg/s)")

    ax.set_xticks([2000, 5000, 10000, 20000, 40000])



def plot_binary_evol(times, R1_out, R2_out, M1_out, M2_out, Teff1_out, Teff2_out,
                     Mdot1_out, Mdot2_out, P_orb_out, ecc_out, L1_out, L2_out, k1_out, k2_out,
                     title=None, file_out=None, sys_obs={}):


    fig, ax = plt.subplots(5, 1, figsize=(10,10))


    gs = gridspec.GridSpec(4, 2,
                           width_ratios=[2,2],
                           height_ratios=[1,2,2,2]
                           )

    ax = np.array([plt.subplot(x) for x in gs])


    # k-type panels
    ax_k_type_list = fig.add_axes([0.08, 0.75, 0.9, 0.1])
    plot_k_type(ax[0], ax[1], ax_k_type_list, times, k1_out, k2_out)

    # Radius panel
    plot_radius(ax[2], times, R1_out, R2_out, M1_out, M2_out, P_orb_out, ecc_out, sys_obs)

    # Mass panel
    plot_mass(ax[4], times, M1_out, M2_out, sys_obs)

    # Teff panel
    plot_Teff(ax[6], times, Teff1_out, Teff2_out, sys_obs)

    # Mass accretion rate panel
    plot_Mdot(ax[3], times, Mdot1_out, Mdot2_out)

    # Orbital period panel
    plot_P_orb(ax[5], times[k2_out<15], P_orb_out[k2_out<15], np.max(times), sys_obs)

    # Plot eccentricity panel
    plot_ecc(ax[7], times[k2_out<15], ecc_out[k2_out<15], np.max(times), sys_obs)


    # Plot HR diagram
    # idx_1 = np.where(k1_out < 10)[0]
    # idx_2 = np.where(k2_out < 10)[0]
    # plot_HR_diagram(ax[7], L1_out[k1_out<10], L2_out[k2_out<10], Teff1_out[k1_out<10], Teff2_out[k2_out<10])


    plt.tight_layout()
    if file_out is not None:
        plt.savefig(file_out)
    else:
        plt.show()


def evolve_return_type(evolve, M1, M2, P_orb, ecc, time,
                       v1_kick=(0.0, 0.0, 0.0), v2_kick=(0.0, 0.0, 0.0), metallicity=0.02,
                       verbose_output=False, sys_obs={}, model_kwargs={}):

    output, sys_type = evolve(M1, M2, P_orb, ecc,
                              v1_kick[0], v1_kick[1], v1_kick[2],
                              v2_kick[0], v2_kick[1], v2_kick[2],
                              time, metallicity, verbose_output,
                              print_history=False, sys_type=True,
                              **model_kwargs)

    return output, sys_type

def evolve_and_print(evolve, M1, M2, P_orb, ecc, time,
                     v1_kick=(0.0, 0.0, 0.0), v2_kick=(0.0, 0.0, 0.0), metallicity=0.02,
                     verbose_output=False, sys_obs={}, model_kwargs={}):


    output = evolve(M1, M2, P_orb, ecc,
                    v1_kick[0], v1_kick[1], v1_kick[2],
                    v2_kick[0], v2_kick[1], v2_kick[2],
                    time, metallicity, verbose_output,
                    print_history=True, **model_kwargs)



def evolve_and_plot(evolve, M1, M2, P_orb, ecc, t_max, t_min=0.1,
                    v1_kick=(0.0, 0.0, 0.0), v2_kick=(0.0, 0.0, 0.0), metallicity=0.02,
                    file_out=None, sys_obs={}, model_kwargs={}):

    # Evolve binary
    times, R1_out, R2_out, M1_out, M2_out, Teff1_out, Teff2_out, \
            Mdot1_out, Mdot2_out, P_orb_out, ecc_out, L1_out, L2_out, k1_out, \
            k2_out = evolve_binary(evolve, M1, M2, P_orb, ecc, t_min, t_max,
                                   v1_kick, v2_kick, metallicity, model_kwargs=model_kwargs)


    # Plot binary
    plot_binary_evol(times, R1_out, R2_out, M1_out, M2_out, Teff1_out, Teff2_out, Mdot1_out, Mdot2_out,
                     P_orb_out, ecc_out, L1_out, L2_out, k1_out, k2_out, file_out=file_out, sys_obs=sys_obs)
