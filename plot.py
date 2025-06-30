import numpy as np
from plot_func import *
import math as m
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc,cm
from matplotlib import rcParams
from scipy.constants import c,pi,h,N_A,R,Boltzmann

flagp = 1
flagw0 = 0
flagVC = 0

tIVR = 10**(-12)           # in s
trep = 10**(-3)            # in s
r = 100*10**(-6)           # laser spot radius
A = pi*r**2                # laser spot area
e_coeff = 200              # in L mol^(-1) cm^(-1)

#####################################################
#                Importing data                     #
#####################################################

hwCO,Ea,Ei = np.loadtxt("constants.txt")

if flagp == 0 and flagVC == 0:
   if flagw0 == 0:
      data1 = np.loadtxt('PulseLowT.txt')
      data3 = np.loadtxt('PulseLow.txt')
      fname1 = "fig2c.pdf"
      fname2 = "fig2b.pdf"
      fname3 = "fig3.pdf"
      fname4 = "fig4.pdf"
   elif flagw0 == 1:
      data1 = np.loadtxt('PulseHighT.txt')
      data3 = np.loadtxt('PulseHigh.txt')
      fname1 = "figS1b.pdf"
      fname2 = "figS1a.pdf"
      fname3 = "figS2.pdf"
      fname4 = "figS3.pdf"

elif flagp == 1 and flagVC == 0:
   if flagw0 == 0:
      data1 = np.loadtxt('CWLowT.txt')
      data3 = np.loadtxt('CWLow.txt')
      fname1 = "fig2f.pdf"
      fname2 = "fig2e.pdf"
   elif flagw0 == 1:
      data1 = np.loadtxt('CWHighT.txt')
      data3 = np.loadtxt('CWHigh.txt')
      fname1 = "figS1d.pdf"
      fname2 = "figS1c.pdf"

tSol = data1[:,0]
DTSol = data1[:,1]

ES = data3[:,0] 
ModeSeS = data3[:,1]
TempIndS = data3[:,2]
DkS = data3[:,3]

if flagp == 0:
   PrS = data3[:,4]
   kavg = data3[:,5]
   kIVR = data3[:,6]

######################################################
#                 Formatting                         #
######################################################

font = {
        'weight' : 'normal',
        'size'   : 16}

plt.rc('font', **font)

# Set additional rc parameters
rcParams['mathtext.fontset'] = 'cm'
rcParams['text.latex.preamble'] = r'\usepackage{physics} \usepackage{amsmath} \usepackage{gensymb}'

######################################################
#                 Plotting                           #
######################################################

plotdT(fname1, tSol, DTSol, flagp)
plotModeTemp(fname2, ES, ModeSeS, TempIndS, DkS, hwCO, Ea, trep, A, e_coeff, Ei, flagp, flagw0)
if flagp == 0:
   plotkIVR(fname3, ES, kavg, kIVR, Ea, A, e_coeff)
   plotPreact(fname4,ES,kIVR,tIVR,Ea,A,e_coeff)
