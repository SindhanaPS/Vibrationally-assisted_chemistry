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
flagw0 = 0

###############################
# flagVC = 0 small tauVC
# flagVC = 1 large tauVC
###############################
flagVC = 1

########################################### Parameters #################################################

if flagw0 == 0:
   w0 = 200                   # in cm^-1
elif flagw0 == 1:
   w0 = 1700

if flagp == 1:
   wD = w0

wCO = 1700                 # in cm^-1
Tout = 295.0                 # in K 
kbare = 4*10**(-7)         # in s^-1

Epulse = 100 * 10**(-6)    # Energy per pulse in J 
r = 100*10**(-6)           # laser spot radius
A = pi*r**2                # laser spot area
e_coeff = 200              # in L mol^(-1) cm^(-1)
tIVR = 10**(-12)           # in s
trep = 10**(-3)            # in s

fmol = 1
R0 = 7*10**(-10)         # in m
kappa = 0.14            # thermal conductivity of toluene (in W.m^(-1).K^(-1))
rho = 0.87              # density of toluene (in g.cm^-3)
Cs = 1.7                # specific heat of toluene (in J.g^(-1).K^(-1))
Nmol = 1

#########################################################################################################

#################### Derived parameters #####################
cminv_to_rads = 2*pi*c*100
cminv_to_Hz = c*100
kcal_to_J = 4184
cm3_to_m3 = 10**(-6)
Lcmm1_tom2 = 10**(-3)/10**(-1)
kB = Boltzmann

E = Epulse*(e_coeff*Lcmm1_tom2*m.log(10))/(A*N_A)   # Energy absorbed per molecule in J
alpha = kappa/((rho/cm3_to_m3)*Cs)                  # Thermal diffusivity (in m^2s^(-1))      

if flagVC == 0:
   tVC = R0**2/(3*alpha)                   # Vibrational cooling time constant (in s)
elif flagVC == 1:
   tVC = 10**(-9)                          # Vibrational cooling time constant (in s)

print('Epulse=',Epulse,'Eabs=',E,'tVC=',tVC)

w0r = w0*cminv_to_rads

Cloc = HeatCapacity(Tout)
print('Nmol=',Nmol,'Cloc=',Cloc)

if flagp == 1:
   wDr = wD*cminv_to_rads

tw0 = 2*pi/w0r                         # in s
zeta = 1/(2*tIVR*w0*cminv_to_rads)     # dimensionless

hw0 = h*w0*cminv_to_Hz
EhwkT = hw0/(2*np.tanh(np.divide(hw0,2*kB*Tout)))

#############################################################


if flagp == 0:
   if flagw0 == 0:
      fname = 'EaPulseLow.txt'
   elif flagw0 == 1:
      fname = 'EaPulseHigh.txt'

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

   ########################## Average rate constant for different barrier heights in Eaarray

   Eaarray = np.linspace(1,14,50)           # in kcal mol^-1

   ModeSe = np.zeros_like(Eaarray)          # Mode selective contribution
   TempInd = np.zeros_like(Eaarray)         # Temperature induced contribution
   Deltak_k = np.zeros_like(Eaarray)        # Relative change in rate constant
   kavg = np.zeros_like(Eaarray)            # Rate constant
   kIVR = np.zeros_like(Eaarray)            # Rate constant within tIVR
   Tmax = np.zeros_like(Eaarray)            # Maximum temperature

   for i in range(np.size(Eaarray)):
      Ea = Eaarray[i]
      A0 = kbare*(2*pi/w0r)*np.exp(Ea*kcal_to_J/(N_A*EhwkT))
      xB = np.sqrt(2*Ea*kcal_to_J/(N_A*w0r**2))
      VB = 0.5*w0r**2*xB**2
      k0 = A0*(w0r/(2*pi))*np.exp(-VB/(EhwkT))
      print('A0=',A0,'xB=',xB,'kbare=',kbare,'k0=',k0,'VB=',VB,'maxk=',A0*(w0r/(2*pi)))

      ModeSe[i],TempInd[i],Deltak_k[i],kavg[i],kIVR[i],Tmax[i] = RateChangeEachEi(E,t1part,t2part,zeta,w0,fmol,Nmol,Cloc,tIVR,tVC,Tout,A0,xB,tsplit)

   # Save output to a file
   np.savetxt(fname,np.transpose([Eaarray,ModeSe,TempInd,Deltak_k,kavg,kIVR,Tmax]))

elif flagp == 1:
   if flagw0 == 0:
      fname = 'EaCWLow.txt'
   elif flagw0 == 1:
      fname = 'EaCWHigh.txt'

   ####################### Time ########################
   tD = 2*pi/wDr                     # in s

   nt = [tw0, tIVR, tVC, tD]
   tmin = min(nt)

   N1 = max(1000,tD*10/tmin)
   t = np.linspace(0,tD,N1)

   ########################## Average rate constant for different initial energy values in array E

   Eaarray = np.linspace(1,20,5)           # in kcal mol^-1

   ModeSe = np.zeros_like(Eaarray)          # Mode selective contribution
   TempInd = np.zeros_like(Eaarray)         # Temperature induced contribution
   Deltak_k = np.zeros_like(Eaarray)        # Relative change in rate constant

   for i in range(np.size(Eaarray)):
      Ea = Eaarray[i]
      A0 = kbare*(2*pi/w0r)*np.exp(Ea*kcal_to_J/(N_A*EhwkT))
      xB = np.sqrt(2*Ea*kcal_to_J/(N_A*w0r**2))
      VB = 0.5*w0r**2*xB**2
      k0 = A0*(w0r/(2*pi))*np.exp(-VB/(EhwkT))
      print('A0=',A0,'xB=',xB,'kbare=',kbare,'k0=',k0,'VB=',VB,'maxk=',A0*(w0r/(2*pi)))

      ModeSe[i],TempInd[i],Deltak_k[i],Tt,Tmax[i] = RateChangeEachEiCW(E,t,zeta,w0,fmol,Nmol,Cloc,tIVR,tVC,Tout,A0,xB,wD,trep)

   # Save output to a file
   np.savetxt(fname,np.transpose([Eaarray,ModeSe,TempInd,Deltak_k,Tmax]))
