import numpy as np
from plot_func import *
import math as m
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc,cm
from matplotlib import rcParams
from scipy.constants import c,pi,h,N_A,R,Boltzmann

flagp = 0
flagw0 = 0
flagVC = 1

Tout = 295.0                 # in K
tIVR = 10**(-12)
trep = 10**(-3)
tVC = 10**(-9)

#####################################################
#                Importing data                     #
#####################################################

if flagp == 0 and flagVC == 1:
   if flagw0 == 0:
      data3 = np.loadtxt('EaPulseLow.txt')
      fname = "fig5.pdf"
   elif flagw0 == 1:
      data3 = np.loadtxt('EaPulseHigh.txt')
      fname = "figExtra.pdf"

elif flagp == 1 and flagVC == 1:
   if flagw0 == 0:
      data3 = np.loadtxt('EaCWLow.txt')
      fname = "figExtra.pdf"
   elif flagw0 == 1:
      data3 = np.loadtxt('EaCWHigh.txt')
      fname = "figExtra.pdf"

Ea = data3[:,0] 
ModeSeS = data3[:,1]
TempIndS = data3[:,2]
DkS = data3[:,3]

if flagp == 0:
   kavg = data3[:,4]
   kIVR = data3[:,5]
   Tmax = data3[:,6]
elif flagp == 1:
   Tmax = data3[:,4]

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


plotModeTempEa(fname, Ea, ModeSeS, TempIndS, DkS, Tout)
