import numpy as np
from func import *
import math as m
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc,cm
from matplotlib import rcParams
from scipy.constants import c,pi,h,N_A,R,Boltzmann


###############################
# flagp = 0 pulsed laser
# flagp = 1 cw laser
###############################
flagp = 0

###############################
# flagw0 = 0 low frequency mode
# flagw0 = 1 high frequency mode
###############################
flagw0 = 1

###############################
# flagVC = 0 small tauVC
# flagVC = 1 large tauVC
###############################
flagVC = 0 

########################################### Parameters #################################################

if flagw0 == 0:
   w0 = 200                   # in cm^-1
elif flagw0 == 1:
   w0 = 1700

if flagp == 1:
   wD = w0 

wCO = 1700                 # in cm^-1
Tout = 295.0                 # in K 
Ea = 14                    # in kcal mol^-1
kbare = 4*10**(-7)         # in s^-1

tIVR = 10**(-12)           # in s
trep = 10**(-3)            # in s

fmol = 1
R0 = 7*10**(-10)         # in m
kappa = 0.14            # thermal conductivity of toluene (in W.m^(-1).K^(-1))
rho = 0.87              # density of toluene (in g.cm^-3)
Cs = 1.7                # specific heat of toluene (in J.g^(-1).K^(-1))
Nmol = 1

twindow = 2*tIVR             # in s 
#########################################################################################################

#################### Derived parameters #####################
cminv_to_rads = 2*pi*c*100
cminv_to_Hz = c*100
kcal_to_J = 4184
cm3_to_m3 = 10**(-6)
kB = Boltzmann

alpha = kappa/((rho/cm3_to_m3)*Cs)      # Thermal diffusivity (in m^2s^(-1))      

if flagVC == 0:
   tVC = R0**2/(3*alpha)                   # Vibrational cooling time constant (in s)
elif flagVC == 1:
   tVC = 10**(-9)                          # Vibrational cooling time constant (in s)

print('tVC=',tVC)

w0r = w0*cminv_to_rads

Cloc = HeatCapacity(Tout)
print('Nmol=',Nmol,'Cloc=',Cloc) 

if flagp == 1:
   wDr = wD*cminv_to_rads

tw0 = 2*pi/w0r                         # in s
zeta = 1/(2*tIVR*w0*cminv_to_rads)     # dimensionless

hw0 = h*w0*cminv_to_Hz
EhwkT = hw0/(2*np.tanh(np.divide(hw0,2*kB*Tout)))

A0 = kbare*(2*pi/w0r)*np.exp(Ea*kcal_to_J/(N_A*EhwkT))
xB = np.sqrt(2*Ea*kcal_to_J/(N_A*w0r**2))
VB = 0.5*w0r**2*xB**2
k0 = A0*(w0r/(2*pi))*np.exp(-VB/(EhwkT))

print(alpha,tVC)
print(zeta)
print('A0=',A0,'xB=',xB,'kbare=',kbare,'k0=',k0,'VB=',VB,'maxk=',A0*(w0r/(2*pi)))

#############################################################

if flagp == 0:
   if flagw0 == 0:
      fname = 'PulseLow.txt'
      fnameT = 'PulseLowT.txt'
   elif flagw0 == 1:
      fname = 'PulseHigh.txt'
      fnameT = 'PulseHighT.txt'
   
   ######################### Splitting the total time into two parts
   ######################### t1part and t2part
 
   tmax = trep                        # in s
   nt = [tw0, tIVR, tVC, trep]

   t1 = nt[0]
   t2 = nt[1]
   t3 = nt[2]
   t4 = nt[3]
   print('tIVR=',tIVR)
   print('tVC=',tVC)
   print('tw0=',t1,'tIVR=',t2,'tVC=',t3,'trep=',t4)
   nt.sort()
   print(t1,t2,t3,t4)
   ratt = [t2/t1,t3/t2,t4/t3]
   print(ratt)
   imax = np.argmax(ratt)
   print(imax)

   if imax == 0:
      tsplit = 10*t1
      N1 = 1000
      N2 = int(tmax*10/t2)
   elif imax == 1:
      tsplit = 10*t2
      N1 = int(1000*t2*10/t1)
      N2 = int(t4*1000/t3)
   elif imax ==2:
      tsplit = 10*t3
      if t3 < 10*t2:
         N1 = int(1000*t3*10/t1)
      else:
         N1 = int(10*t3/t1)
      N2 = 1000

   print(tsplit,N1,N2)
   t1part = np.linspace(0,tsplit,N1)
   t2part = np.linspace(tsplit,tmax,N2)

   ########################## Average rate constant for different initial energy values in array E

   E = np.linspace(0,4*h*wCO*cminv_to_Hz,50)

   ModeSe = np.zeros_like(E)          # Mode selective contribution
   TempInd = np.zeros_like(E)         # Temperature induced contribution
   Deltak_k = np.zeros_like(E)        # Relative change in rate constant
   kavg = np.zeros_like(E)            # Rate constant
   kIVR = np.zeros_like(E)            # Rate constant within tIVR
   Pr = np.zeros_like(E)              # Probability of reacting 
   Tmax = np.zeros_like(E)            # maximum temperature

   for i in range(np.size(E)):
      ModeSe[i],TempInd[i],Deltak_k[i],kavg[i],kIVR[i],Tmax[i] = RateChangeEachEi(E[i],t1part,t2part,zeta,w0,fmol,Nmol,Cloc,tIVR,tVC,Tout,A0,xB,tsplit,twindow)

      if np.exp(-tIVR*kIVR[i]) < 1-10**(-3): 
         Pr[i] = 1 - np.exp(-tIVR*kIVR[i])
      else:
         Pr[i] = tIVR*kIVR[i]

   Ekcal = E*N_A/kcal_to_J
   # Save output to a file
   np.savetxt(fname,np.transpose([Ekcal,ModeSe,TempInd,Deltak_k,Pr,kavg,kIVR,Tmax]))

   ################## Temperature plots ###########################
   Ei = 3*h*wCO*cminv_to_Hz                       # in J

   t = np.concatenate((t1part, t2part[1:]))
   w0r = w0*cminv_to_rads

   # Mean position and momentum
   Xt1,Pt1 = PulseXP(t1part, zeta, w0, Ei)
   Xt2,Pt2 = PulseXP(t2part, zeta, w0, Ei)      

   Xt = np.concatenate((Xt1, Xt2[1:]))
   Pt = np.concatenate((Pt1, Pt2[1:]))

   print('Xt0=',Xt[0])
   # Temperature calculation
   Tt1 = Tloc(t1part, fmol, Nmol, Cloc, tIVR, tVC, Tout, Tout, zeta, w0, Ei, 0, 0, 0)
   Tt2 = Tloc(t2part, fmol, Nmol, Cloc, tIVR, tVC, Tout, Tt1.y[0][-1], zeta, w0, Ei, 0, 0, 0)
   Tt = np.concatenate((Tt1.y[0], Tt2.y[0][1:]))

   n1 = len(Tt1.y[0])        # includes the first point
   Ttavg = RunningAverageT(Tt, t, Tout, twindow)

   # Save output to a file
   np.savetxt(fnameT,np.transpose([t,Ttavg-Tout]))
   np.savetxt("constants.txt", [h*wCO*cminv_to_Hz*N_A/kcal_to_J,Ea,Ei])

   ###################################################################

elif flagp == 1:
   if flagw0 == 0:
      fname = 'CWLow.txt'
      fnameT = 'CWLowT.txt'
   elif flagw0 == 1:
      fname = 'CWHigh.txt'
      fnameT = 'CWHighT.txt'

   ####################### Time ########################
   tD = 2*pi/wDr                     # in s

   nt = [tw0, tIVR, tVC, tD]
   tmin = min(nt)

   N1 = max(1000,tD*10/tmin)
   t = np.linspace(0,tD,N1)

   ########################## Average rate constant for different initial energy values in array E

   E = np.linspace(0,4*h*wCO*cminv_to_Hz,50)

   ModeSe = np.zeros_like(E)          # Mode selective contribution
   TempInd = np.zeros_like(E)         # Temperature induced contribution
   Deltak_k = np.zeros_like(E)        # Relative change in rate constant
   Tmax = np.zeros_like(E)            # maximum temperature

   for i in range(np.size(E)):
      ModeSe[i],TempInd[i],Deltak_k[i],Tt,Tmax[i] = RateChangeEachEiCW(E[i],t,zeta,w0,fmol,Nmol,Cloc,tIVR,tVC,Tout,A0,xB,wD,trep,twindow)
      if E[i] >= 3*h*wCO*cminv_to_Hz and E[i-1] < 3*h*wCO*cminv_to_Hz:
         Ttsave = Tt

   Ekcal = E*N_A/kcal_to_J
   # Save output to a file
   np.savetxt(fnameT,np.transpose([t,Ttsave-Tout]))
   np.savetxt(fname,np.transpose([Ekcal,ModeSe,TempInd,Deltak_k,Tmax]))

