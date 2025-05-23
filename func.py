import numpy as np
import math as m
from scipy.special import erf
from scipy.constants import c,pi,h,N_A,Boltzmann
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

cminv_to_rads = 2*pi*c*100
cminv_to_Hz = c*100
kB = Boltzmann
#EPS = np.finfo(float).eps
EPS = 10**(-15)

def HeatCapacity(Tout):

   # Import vibrational spectrum of Amd5b where frequency is in cm^-1 units
   data1 = np.loadtxt('spectrum.txt')
   Cv = 0

   # Adding the translational and rotational contributions to the heat capacity
   Cv += 3*kB

   # Frequency list for Amd5b
   fList = data1[:,0]*cminv_to_Hz

   for freq in fList:
      hw = h*freq
      Theta = hw/(kB*Tout)
      Cv += kB*np.exp(Theta)*(Theta/(np.exp(Theta)-1))**2

   return(Cv)

def PulseXP(t, zeta, w0, E):
   # Computes average position of a damped thermal harmonic oscillator at time t where at t=0 v=0 and energy E is added to the system

   w0r = w0*cminv_to_rads
   phi = -np.arctan(np.sqrt(1-zeta**2)/zeta)
   x0 = np.sqrt(2*E/w0r**2)/np.sin(phi)
   p0 = -x0*w0r

   x = x0*np.multiply(np.exp(-zeta*w0r*t),np.sin(np.sqrt(1-zeta**2)*w0r*t+phi))
   p = p0*np.multiply(np.exp(-zeta*w0r*t),np.sin(np.sqrt(1-zeta**2)*w0r*t))

   # If input was scalar, return scalars
   if x.size == 1:
      return float(x), float(p)
   else:
      return x, p

def CWXP(t, zeta, w0, F0, wD):
  
   w0r = w0*cminv_to_rads
   wDr = wD*cminv_to_rads
   if w0 != wD:
      delta = np.arctan(2*wD*zeta*w0/(w0**2-wD**2))
   else:
      delta = pi/2
 
   x = F0*np.cos(wDr*t-delta)/np.sqrt((wDr**2-w0r**2)**2+(2*zeta*w0r*wDr)**2)
   p = -F0*wDr*np.sin(wDr*t-delta)/np.sqrt((wDr**2-w0r**2)**2+(2*zeta*w0r*wDr)**2)

   # If input was scalar, return scalars
   if x.size == 1:
      return float(x), float(p)
   else:
      return x, p

# Define the functions p(t) and f(t)
def p(t,tVC):
    return -1.0 / np.float64(tVC)

def f_rhs(t, p_val, fmol, Nmol, Cloc, tIVR, tVC, Tout, zeta, w0, E, F0, wD, flag1):

    if flag1 == 0:
       Xt,Pt = PulseXP(t, zeta, w0, E)     
    elif flag1 == 1:
       Xt,Pt = CWXP(t, zeta, w0, F0, wD)

    Pt2_term = (fmol * Nmol / Cloc) * (1.0 / tIVR) * Pt**2
    val = Pt2_term - p_val * Tout

    return val

# Define the first-order ODE
# dy/dt = f(t) + p(t)*y
def ode_system(t, y, fmol, Nmol, Cloc, tIVR, tVC, Tout, zeta, w0, E, F0, wD, flag1):

    p_val = p(t, tVC)
    f_val = f_rhs(t, p_val, fmol, Nmol, Cloc, tIVR, tVC, Tout, zeta, w0, E, F0, wD, flag1)

    dydt = f_val + p_val * np.float64(y[0])

    return [dydt]

def Tloc(t, fmol, Nmol, Cloc, tIVR, tVC, Tout, T0, zeta, w0, E, F0, wD, flag1):

   t_span = [t[0],t[-1]]

   # Solve the ODE
   sol = solve_ivp(ode_system, t_span, [T0], args=(fmol, Nmol, Cloc, tIVR, tVC, Tout, zeta, w0, E, F0, wD, flag1), t_eval=t, method='BDF', rtol=1e-10, atol=1e-10, max_step=t[1]-t[0])

   return(sol)

def Delta2XP(t, T, w0):

   w0r = w0*cminv_to_rads
   hw0 = h*w0*cminv_to_Hz
   kBT = kB * np.array(T)

   Delta2Xt = hw0/(2*w0r**2*np.tanh(np.divide(hw0,2*kBT)))
   Delta2Pt = hw0/(2*np.tanh(np.divide(hw0,2*kBT)))

   return(Delta2Xt,Delta2Pt)

def RateConstantPump(t, Xt, Pt, D2Xt, D2Pt, w0, A0, xB, Tout, Ei, Tt, zeta, tsplit):
   # Computes the quantum rate constant of a reaction using transition state theory (Wigner function of driven harmonic oscillator) at time t 
   # using positions and momenta of a damped thermal harmonic oscillator

   Xtk = Xt
   Ptk = Pt
   w0r = w0*cminv_to_rads
   hw0 = h*w0*cminv_to_Hz
   VB = 0.5*w0r**2*xB**2
   print('VB=',VB,'hw0=',hw0,'kBT=',kB*Tout)
   kt = np.zeros_like(t)  

   for ti in range(t.size):

      if ti != t.size-1:
         dt = t[ti+1]-t[ti] 
      else:
         dt = t[ti]-t[ti-1]

      if t[ti] < tsplit:
         Eti = 0.5*w0r**2*Xt[ti]**2+0.5*Pt[ti]**2

         if Eti > VB:
            Xtk[ti] = xB
            Ptk[ti] = 0

         Px = np.sqrt(np.divide(1,2*pi*D2Xt[ti]))*np.exp(-np.divide(np.power(xB-Xtk[ti],2),2*D2Xt[ti]))
         Pvv = np.sqrt(np.divide(D2Pt[ti],2*pi))*np.exp(-np.divide(np.power(Ptk[ti],2),2*D2Pt[ti])) + 0.5*np.multiply(Ptk[ti],(1+erf(np.multiply(np.sqrt(np.divide(1,2*D2Pt[ti])),Ptk[ti]))))

         kt[ti] = A0*np.multiply(Px,Pvv)

      else:
         ktprev = 0
         counter1 = 0
         Navg = 20
         while True:
            deltat = np.linspace(0,2*pi/w0r,Navg)
            t2 = t[ti] + deltat
            Xt2,Pt2 = PulseXP(t2, zeta, w0, Ei)
            D2Xt2,D2Pt2= Delta2XP(t2, Tt[ti]*np.ones_like(t2), w0)

            Px = np.sqrt(np.divide(1,2*pi*D2Xt2))*np.exp(-np.divide(np.power(xB-Xt2,2),2*D2Xt2))
            Pvv = np.sqrt(np.divide(D2Pt2,2*pi))*np.exp(-np.divide(np.power(Pt2,2),2*D2Pt2)) + 0.5*np.multiply(Pt2,(1+erf(np.multiply(np.sqrt(np.divide(1,2*D2Pt2)),Pt2))))
         
            ktemp = A0*np.multiply(Px,Pvv)
            kt[ti] = np.mean(ktemp)
         
            if kt[ti]-ktprev < 10**(-9)*kt[ti]:
#               if counter1 > 0:
#                  print("Final kt[ti]=",kt[ti],ktprev,kt[ti]-ktprev,10**(-9)*kt[ti],ti)
               break
            else:
               counter1 += 1
#               print("Current kt[ti]=",kt[ti],ktprev,ti)
               ktprev = kt[ti]
               Navg = Navg*5

   return(kt)

def RateConstant(t, Xt, Pt, D2Xt, D2Pt, w0, A0, xB, Tout, Tt, zeta):
   # Computes the quantum rate constant of a reaction using transition state theory (Wigner function of driven harmonic oscillator) at time t 
   # using positions and momenta of a driven-damped thermal harmonic oscillator

   Xtk = Xt
   Ptk = Pt
   w0r = w0*cminv_to_rads
   hw0 = h*w0*cminv_to_Hz
   VB = 0.5*w0r**2*xB**2
   print('VB=',VB,'hw0=',hw0,'kBT=',kB*Tout)
   kt = np.zeros_like(t)  

   for ti in range(t.size):

      if ti != t.size-1:
         dt = t[ti+1]-t[ti] 
      else:
         dt = t[ti]-t[ti-1]

      Eti = 0.5*w0r**2*Xt[ti]**2+0.5*Pt[ti]**2

      Px = np.sqrt(np.divide(1,2*pi*D2Xt[ti]))*np.exp(-np.divide(np.power(xB-Xtk[ti],2),2*D2Xt[ti]))
      Pvv = np.sqrt(np.divide(D2Pt[ti],2*pi))*np.exp(-np.divide(np.power(Ptk[ti],2),2*D2Pt[ti])) + 0.5*np.multiply(Ptk[ti],(1+erf(np.multiply(np.sqrt(np.divide(1,2*D2Pt[ti])),Ptk[ti]))))

      kt[ti] = A0*np.multiply(Px,Pvv)

   return(kt)

def RateAvg(t, kt):

   print("len(kt):", len(kt), "len(t):", len(t))
   kinteg = np.trapz(kt,t)

   kinteg = kinteg/(t[-1]-t[0])

   return kinteg

def Perc(kavg, k0Tloc, k0Tout):
   
   ModeSe = (kavg - k0Tloc)/k0Tout 
   TempInd = (k0Tloc-k0Tout)/k0Tout
   Deltak_k = ModeSe + TempInd

   # Set near-zero contributions to zero
   if abs(ModeSe) < EPS:
      ModeSe = 0.0
   if abs(TempInd) < EPS:
      TempInd = 0.0
   if abs(Deltak_k) < EPS:
      Deltak_k = 0.0

   return(ModeSe,TempInd,Deltak_k)

#def Preact(Ei,zeta,w0,fmol,Nmol,Cloc,tIVR,tVC,Tout,A0,xB):
#   t = np.linspace(0,tIVR,10000)
#   w0r = w0*cminv_to_rads

   # Mean position and momentum
#   Xt,Pt = PulseXP(t, zeta, w0, Ei)

   # Temperature calculation
#   Tt = Tloc(t, fmol, Nmol, Cloc, tIVR, tVC, Tout, Tout, zeta, w0, Ei, 0, 0, 0)

   # Delta X, P calcuation
#   D2Xt,D2Pt = Delta2XP(t, Tt.y[0], w0)

   # Rate constant
#   kt = RateConstantPump(t, Xt, Pt, D2Xt, D2Pt, w0, A0, xB, Tout, Ei, Tt.y[0], zeta, 20)

   # Average rate
#   kIVR = RateAvg(t, kt)

#   Pr = 1 - np.exp(-tIVR*kIVR)
#   print('kIVR*tIVR',kIVR*tIVR,'Pr',Pr)

#   return(Pr)

def RateChangeEachEi(Ei,t1part,t2part,zeta,w0,fmol,Nmol,Cloc,tIVR,tVC,Tout,A0,xB,tsplit):

   print('Ei=',Ei)
   t = np.concatenate((t1part, t2part[1:]))
   w0r = w0*cminv_to_rads

   # Mean position and momentum
   Xt1,Pt1 = PulseXP(t1part, zeta, w0, Ei)
   Xt2,Pt2 = PulseXP(t2part, zeta, w0, Ei)        # The energy input here needs to be the energy at t = 0 which is why it is Ei

   Xt = np.concatenate((Xt1, Xt2[1:]))
   Pt = np.concatenate((Pt1, Pt2[1:]))

   # Temperature calculation
   Tt1 = Tloc(t1part, fmol, Nmol, Cloc, tIVR, tVC, Tout, Tout, zeta, w0, Ei, 0, 0, 0)
   Tt2 = Tloc(t2part, fmol, Nmol, Cloc, tIVR, tVC, Tout, Tt1.y[0][-1], zeta, w0, Ei, 0, 0, 0)
   Tt = np.concatenate((Tt1.y[0], Tt2.y[0][1:]))

   print("Tt1 min:", np.min(Tt1.y[0]), "max:", np.max(Tt1.y[0]), "mean:", np.mean(Tt1.y[0]))
   print("Tt1 unique values:", np.unique(Tt1.y[0]))
   print("Tt2 min:", np.min(Tt2.y[0]), "max:", np.max(Tt2.y[0]), "mean:", np.mean(Tt2.y[0]))
   print("Tt2 unique values:", np.unique(Tt2.y[0]))

   # Delta X, P calcuation
   D2Xt1,D2Pt1= Delta2XP(t1part, Tt1.y[0], w0)
   D2Xt2,D2Pt2= Delta2XP(t2part, Tt2.y[0], w0)

   print("D2Xt1 min:", np.min(D2Xt1), "max:", np.max(D2Xt1), "mean:", np.mean(D2Xt1))
   print("D2Xt1 unique values:", np.unique(D2Xt1))
   print("D2Xt2 min:", np.min(D2Xt2), "max:", np.max(D2Xt2), "mean:", np.mean(D2Xt2))
   print("D2Xt2 unique values:", np.unique(D2Xt2))

   D2Xt = np.concatenate((D2Xt1, D2Xt2[1:]))
   D2Pt = np.concatenate((D2Pt1, D2Pt2[1:]))

   # Rate constant
   # k0(Tloc)
   k0Tloc = RateConstant(t, np.zeros_like(Xt), np.zeros_like(Pt), D2Xt, D2Pt, w0, A0, xB, Tout, Tt, zeta)
   k0TlocAvg = RateAvg(t, k0Tloc)

   # k0(Tout)
   D2Xt0,D2Pt0 = Delta2XP(t, Tout*np.ones_like(Tt), w0)
   k0Tout = RateConstant(t, np.zeros_like(Xt), np.zeros_like(Pt), D2Xt0, D2Pt0, w0, A0, xB, Tout, Tout*np.ones_like(t), zeta)
   k0ToutAvg = RateAvg(t, k0Tout)
   print('k0ToutAvg=',k0ToutAvg)
   print('k0TlocAvg=',k0TlocAvg)

   print("kt1")
   kt1 = RateConstant(t1part, Xt1, Pt1, D2Xt1, D2Pt1, w0, A0, xB, Tout, Tt1.y[0], zeta)
   print("kt2")
   kt2 = RateConstantPump(t2part, Xt2, Pt2, D2Xt2, D2Pt2, w0, A0, xB, Tout, Ei, Tt2.y[0], zeta, tsplit)
   kt = RateConstantPump(t, Xt, Pt, D2Xt, D2Pt, w0, A0, xB, Tout, Ei, Tt, zeta, tsplit)

   print("kt1 min:", np.min(kt1), "max:", np.max(kt1), "mean:", np.mean(kt1))
   print("kt1 unique values:", np.unique(kt1))
   print("kt2 min:", np.min(kt2[1:]), "max:", np.max(kt2[1:]), "mean:", np.mean(kt2[1:]))
   print("kt2 unique values:", np.unique(kt2[1:]))

   # Average rate
   kavg1 = RateAvg(t1part, kt1)
   print('kavg1=',kavg1)
   kavg2 = RateAvg(t2part[1:], kt2[1:])
   print('kavg2=',kavg2)
   kavg = RateAvg(t, kt)
   print('kavg=',kavg)
   
   # Mode-selective and Temperature contributions
   ModeSe,TempInd,Deltak_k = Perc(kavg, k0TlocAvg, k0ToutAvg)

   print('ModeSe=',ModeSe,'TempInd=',TempInd,'Deltakk=',Deltak_k)
      
   return (ModeSe,TempInd,Deltak_k,kavg,kavg1)


def RateChangeEachEiCW(Ei,t,zeta,w0,fmol,Nmol,Cloc,tIVR,tVC,Tout,A0,xB,wD,trep):

   print('Ei=',Ei)
   w0r = w0*cminv_to_rads   
   wDr = wD*cminv_to_rads   

   #Calculate F0
   F0 = np.sqrt(Ei*((wDr**2-w0r**2)**2+(2*zeta*w0r*wDr)**2)/(trep*zeta*wDr**2*w0r))

   # Mean position and momentum
   Xt,Pt = CWXP(t, zeta, w0, F0, wD)

   # Temperature calculation
   T0 = Tout + tVC*np.mean(np.square(Pt))/(Cloc*tIVR) 
   Tt = Tloc(t, fmol, Nmol, Cloc, tIVR, tVC, Tout, T0, zeta, w0, Ei, F0, wD, 1)

   # Delta X, P calcuation
   D2Xt,D2Pt= Delta2XP(t, Tt.y[0], w0)

   # Rate constant
   kt = RateConstant(t, Xt, Pt, D2Xt, D2Pt, w0, A0, xB, Tout, Tt.y[0], zeta)

   kavg = RateAvg(t, kt)
   print('kavg=',kavg)

   # k0(Tloc)
   k0Tloc = RateConstant(t, np.zeros_like(Xt), np.zeros_like(Pt), D2Xt, D2Pt, w0, A0, xB, Tout, Tt.y[0], zeta)
   k0TlocAvg = RateAvg(t, k0Tloc)

   # k0(Tout)
   D2Xt0,D2Pt0 = Delta2XP(t, Tout*np.ones_like(t), w0)
   k0Tout = RateConstant(t, np.zeros_like(Xt), np.zeros_like(Pt), D2Xt0, D2Pt0, w0, A0, xB, Tout, Tout*np.ones_like(t), zeta)
   k0ToutAvg = RateAvg(t, k0Tout)
   print('k0ToutAvg=',k0ToutAvg)
   print('k0TlocAvg=',k0TlocAvg)

   # Mode-selective and Temperature contributions
   ModeSe,TempInd,Deltak_k = Perc(kavg, k0TlocAvg, k0ToutAvg)

   print('ModeSe=',ModeSe,'TempInd=',TempInd,'Deltakk=',Deltak_k)

   return (ModeSe,TempInd,Deltak_k,Tt.y[0])
