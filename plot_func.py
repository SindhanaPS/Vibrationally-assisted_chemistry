import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as tck
import math as m
from cmath import pi
from scipy.constants import c,pi,h,N_A,R,Boltzmann
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter

kcal_to_J = 4184
J_to_muJ = 10**(6)
J_to_mJ = 10**(3)
Lcmm1_tom2 = 10**(-3)/10**(-1)
kB = Boltzmann

def plotdT(fname2,tsol,DeltaTsol,flagp):

    aspect_ratio = 2 / 6

    # Base height in inches
    base_height = 15

    # Calculate width based on the aspect ratio
    width = base_height * aspect_ratio

    fig = plt.figure(figsize=(width, base_height*1.3))
    gs = GridSpec(2, 1, height_ratios=[0.3, 1], hspace=0.05)
    plt.rcParams.update({
    'font.size': 25,
    })

    ax1 = fig.add_subplot(gs[0])

    formatter = ScalarFormatter(useMathText=True)  # Use \(10^x\) format
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))  # Use scientific notation for numbers <1e-2 or >1e2

    formatter2 = ScalarFormatter(useMathText=True)  # Use \(10^x\) format
    formatter2.set_scientific(True)
    formatter2.set_powerlimits((-2, 2))  # Use scientific notation for numbers <1e-2 or >1e2

    ax1.xaxis.set_major_formatter(formatter2)
    ax1.yaxis.set_major_formatter(formatter)

    # Make the border of the plot bold
    for spine in ax1.spines.values():
       spine.set_linewidth(2)  # Adjust the border (spine) linewidth

    ax1.tick_params(axis='both', which='major', labelsize=22, width=3, direction='in', length=7)

    # Plot for Delta T
    ax1.set_ylabel(r'$\Delta$T$_{\text{loc}}$ ($\degree$C)', fontsize='25')
#    ax1.plot(tsol, DeltaTsol, color='#0877c4', linestyle='dashed', linewidth=3)
    ax1.plot(tsol, DeltaTsol, color='black', linewidth=2, linestyle='--')

    # Shared X-axis label
    ax1.set_xlabel('Time (s)', fontsize='25')

    if flagp == 0:
       ax1.set_xscale('log')

    ax1.xaxis.get_offset_text().set_y(-50)
    plt.tight_layout()
    plt.savefig(fname2,bbox_inches='tight',dpi=100)

def plotModeTemp(fname, Ekcal, ModeSe, TempInd, Deltak_k, hwCO, Ea, trep, A, e_coeff, Ei, flagp, flagw0):

    fig = plt.figure()
    plt.rcParams.update({
    'font.size': 20,
    })

    ax1 = plt.gca()

    # Make the border of the plot bold
    for spine in ax1.spines.values():
       spine.set_linewidth(2)  # Adjust the border (spine) linewidth

    ax1.tick_params(axis='both', which='major', labelsize=22, width=3, direction='in', length=7)

    if ModeSe[0] == 0:
       ModeSe = ModeSe[1:]
       EJ1 = Ekcal[1:]*kcal_to_J/N_A
       PW1 = EJ1/trep

    if TempInd[0] == 0:
       TempInd = TempInd[1:]
       EJ2 =  Ekcal[1:]*kcal_to_J/N_A
       PW2 = EJ2/trep

    if Deltak_k[0] == 0:
       Deltak_k = Deltak_k[1:]
       EJ3 =  Ekcal[1:]*kcal_to_J/N_A
       PW3 = EJ3/trep

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))  # Apply 10^x for numbers < 1e-2 or > 1e2
    ax1.xaxis.set_major_formatter(formatter)

    if flagp == 0:
       plt.axvline(x=Ea*kcal_to_J/N_A, linestyle='-', linewidth=2, color='lightgrey') 
       plt.plot(EJ1, Deltak_k, linewidth=3.5, color='#000000', label='Total')  # Black
       plt.plot(EJ2, ModeSe, linewidth=2.5, linestyle='--', color='#FDB515', label='Vibrationally-assisted')
       plt.plot(EJ3, TempInd, linewidth=2.5, linestyle='--', color='#56B4E9', label='Temperature-induced')

       Ei_idx = np.argmin(np.abs(EJ1 - Ei))  # find closest x in EJ1 to Ei
       plt.plot(Ei, Deltak_k[Ei_idx], marker='*', color='black', markersize=20, label=None)

       ax1.set_xlabel(r'$E_{\text{abs}}$ ($\text{J}$)')

       scale = A*N_A*J_to_muJ/(e_coeff*Lcmm1_tom2*m.log(10))                   # conversion from E_abs in J to E_pulse in \mu J
       secax = ax1.secondary_xaxis('top',
               functions=(lambda x: scale * x, lambda x: x / scale))
       secax.set_xlabel(r'$E_{\text{pulse}}$ ($\mu\text{J}$)',labelpad=10)
    elif flagp == 1:
       plt.plot(PW1, Deltak_k, linewidth=3.5, color='#000000', label='Total')  # Black
       plt.plot(PW2, ModeSe, linewidth=2.5, linestyle='--', color='#FDB515', label='Vibrationally-assisted')
       plt.plot(PW3, TempInd, linewidth=2.5, linestyle='--', color='#56B4E9', label='Temperature-induced')

       Ei_idx = np.argmin(np.abs(EJ1 - Ei))  # find closest x in EJ1 to Ei
       plt.plot(Ei/trep, Deltak_k[Ei_idx], marker='*', color='black', markersize=20, label=None)

       ax1.set_xlabel(r'$P_{\text{abs}}$ ($\text{W}$)')

       scale = A*N_A*J_to_mJ/(e_coeff*Lcmm1_tom2*m.log(10))                   # conversion from P_abs in W to P_cw in mW
       secax = ax1.secondary_xaxis('top',
               functions=(lambda x: scale * x, lambda x: x / scale))
       secax.set_xlabel(r'$P_{\text{cw}}$ ($\text{mW}$)',labelpad=10)

    plt.yscale('log')
    ax1.legend(frameon=False, fontsize=15)
    plt.xlim(left=0)
    ax1.set_ylabel('$\Delta k/k$')

    secax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    secax.tick_params(axis='x', which='both', direction='in', length=7, width=3, labelsize=22)

    plt.tight_layout()
    plt.savefig(fname,bbox_inches='tight',dpi=100)

def plotModeTempEa(fname, Ea, ModeSe, TempInd, Deltak_k, Tout):

    fig = plt.figure()
    plt.rcParams.update({
    'font.size': 20,
    })

    ax1 = plt.gca()

    # Make the border of the plot bold
    for spine in ax1.spines.values():
       spine.set_linewidth(2)  # Adjust the border (spine) linewidth

    ax1.tick_params(axis='both', which='major', labelsize=22, width=3, direction='in', length=7)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))  # Apply 10^x for numbers < 1e-2 or > 1e2
    ax1.xaxis.set_major_formatter(formatter)

    plt.plot(Ea, Deltak_k, linewidth=3.5, color='#000000', label='Total')  # Black
    plt.plot(Ea, ModeSe, linewidth=2.5, linestyle='--', color='#FDB515', label='Vibrationally-assisted')
    plt.plot(Ea, TempInd, linewidth=2.5, linestyle='--', color='#56B4E9', label='Temperature-induced')

    ax1.set_xlabel(r'$V_{B}$ ($\text{kcal mol}^{-1}$)')

    scale = kcal_to_J/(N_A*kB*Tout)                   # conversion from Ea to Ea/kBT
    secax = ax1.secondary_xaxis('top',
            functions=(lambda x: scale * x, lambda x: x / scale))
    secax.set_xlabel(r'$V_{B}/k_BT_{\text{out}}$',labelpad=10)

    plt.yscale('log')
    ax1.legend(frameon=False, fontsize=15)
    plt.xlim(left=0)
    ax1.set_ylabel('$\Delta k/k$')

    secax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    secax.tick_params(axis='x', which='both', direction='in', length=7, width=3, labelsize=22)

    plt.tight_layout()
    plt.savefig(fname,bbox_inches='tight',dpi=100)

def plotkt(fname, Ekcal, t, kt, dTti, ktTloc, k0Tout):
    
    fig = plt.figure()
    plt.rcParams.update({
    'font.size': 22,
    })

    ax1 = plt.gca()

    # Make the border of the plot bold
    for spine in ax1.spines.values():
       spine.set_linewidth(2)  # Adjust the border (spine) linewidth

    for i in range(len(Ekcal)):
       Ei = Ekcal[i]
       plt.plot(t, (kt[i]-k0Tout)/np.mean(k0Tout), label=fr'$E={Ei:.2e}$')

#    plt.yscale('log')
    plt.xlim(0, 10**(-12))
#    plt.xscale('log')
    ax1.set_ylabel('$\Delta$k(t)/k')
    ax1.set_xlabel('Energy (kcal mol$^{-1}$)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname,bbox_inches='tight',dpi=100)

def plotkIVR(fname, Ekcal, kavg, kIVR, Ea, A, e_coeff):
 
    fig = plt.figure()
    plt.rcParams.update({
    'font.size': 22,
    })

    ax1 = plt.gca()

    # Make the border of the plot bold
    for spine in ax1.spines.values():
       spine.set_linewidth(2)  # Adjust the border (spine) linewidth

    ax1.tick_params(axis='both', which='major', labelsize=22, width=3, direction='in', length=7)
    ax1.tick_params(axis='both', which='minor', direction='in', width=1, length=4)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))  # Apply 10^x for numbers < 1e-2 or > 1e2
    ax1.xaxis.set_major_formatter(formatter)

    plt.yscale('log')
    plt.axvline(x=Ea*kcal_to_J/N_A, linestyle='-', linewidth=2, color='lightgrey')
    plt.plot(Ekcal*kcal_to_J/N_A, kIVR/kavg[0], linewidth=3.5, color='#000000')  # Black

    scale = A*N_A*J_to_muJ/(e_coeff*Lcmm1_tom2*m.log(10))                   # conversion from E_abs in J to E_pulse in \mu J
    secax = ax1.secondary_xaxis('top',
            functions=(lambda x: scale * x, lambda x: x / scale))
    secax.set_xlabel(r'$E_{\text{pulse}}$ ($\mu\text{J}$)',labelpad=10)
    plt.xlim(left=0)
 
    ax1.set_ylabel(r'$k_{IVR}/k_0(T_{\text{out}})$')
    ax1.set_xlabel(r'$E_{\text{abs}}$ ($\text{J}$)')
    secax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    secax.tick_params(axis='x', which='both', direction='in', length=7, width=3, labelsize=22)

    plt.tight_layout()
    plt.savefig(fname,bbox_inches='tight',dpi=100)

def plotPreact(fname,E,kIVR,tIVR,Ea,A,e_coeff):

    fig = plt.figure()
    plt.rcParams.update({
    'font.size': 22,
    })

    ax1 = plt.gca()

    # Make the border of the plot bold
    for spine in ax1.spines.values():
       spine.set_linewidth(2)  # Adjust the border (spine) linewidth

    Pr = np.zeros_like(E)

    for i in range(len(Pr)):
       if np.exp(-tIVR*kIVR[i]) < 1-10**(-3): 
          Pr[i] = 1 - np.exp(-tIVR*kIVR[i])
       else:
          Pr[i] = tIVR*kIVR[i]

    ax1.tick_params(axis='both', which='major', labelsize=22, width=3, direction='in', length=7)
    ax1.tick_params(axis='both', which='minor', direction='in', width=1, length=4)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))  # Apply 10^x for numbers < 1e-2 or > 1e2
    ax1.xaxis.set_major_formatter(formatter)

    plt.yscale('log')
    plt.axvline(x=Ea*kcal_to_J/N_A, linestyle='-', linewidth=2, color='lightgrey')
    plt.plot(E*kcal_to_J/N_A,Pr,linewidth=3.5,color='#D62728')

    scale = A*N_A*J_to_muJ/(e_coeff*Lcmm1_tom2*m.log(10))                   # conversion from E_abs in J to E_pulse in \mu J
    secax = ax1.secondary_xaxis('top',
            functions=(lambda x: scale * x, lambda x: x / scale))
    secax.set_xlabel(r'$E_{\text{pulse}}$ ($\mu\text{J}$)',labelpad=10)
    plt.xlim(left=0)

    ax1.set_ylabel(r'$P_{\mathrm{react}}$')
    ax1.set_xlabel(r'$E_{\text{abs}}$ ($\text{J}$)')
    secax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    secax.tick_params(axis='x', which='both', direction='in', length=7, width=3, labelsize=22)

    plt.tight_layout()
    plt.savefig(fname,bbox_inches='tight',dpi=100)
