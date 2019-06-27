#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:13:16 2019

@author: sotzee
"""
from scipy.integrate import quad
from scipy.special import erf,erfi
from scipy.misc import derivative
from scipy.constants import c,G,e
import numpy as np
dlnx_cs2=1e-6


def Gamma(x,gammas):
    return np.exp((np.array(gammas)*np.array([x**0,x,x**2])).sum(axis=0))
def baryon_ratio(x,gammas): #\int_{0}^x\frac{dx'}{\Gamma(x')}
    x_0=np.array([x,0])
    integration_x_0=np.real(np.pi**0.5/(4*gammas[2])**0.5*np.exp(-gammas[0]+gammas[1]**2/(4*gammas[2]))*erf((gammas[2]*x_0+0.5*gammas[1])/(gammas[2])**0.5))
    return np.exp(integration_x_0[0]-integration_x_0[1])
def to_integerate(x,gammas):
    return np.exp(x)/(Gamma(x,gammas)*baryon_ratio(x,gammas))
def density(x,gammas,density_0,pressure_0):
    return baryon_ratio(x,gammas)*(density_0+pressure_0*quad(to_integerate,0,x,args=gammas)[0])
#time for i in range(100000):a=baryon_ratio(1.1,[2,2,2])
#time for i in range(100000):a=density(0.,[1,.2,.3],60,1)

class Spectral3(object):
    def __init__(self,args):
        self.nep0,self.gammas=args
        self.gammas=[float(self.gammas[0]),float(self.gammas[1]),float(self.gammas[2])]
        self.baryon_density_s=0.16
        self.baryon_density_0,self.density_0,self.pressure_0=self.nep0
        self.unit_mass=c**4/(G**3*self.density_0*1e51*e)**0.5
        self.unit_radius=c**2/(G*self.density_0*1e51*e)**0.5
        self.unit_N=self.unit_radius**3*self.baryon_density_0*1e45
    def eosDensity(self,pressure):
        return density(np.log(pressure/self.pressure_0),self.gammas,self.density_0,self.pressure_0)
    def eosBaryonDensity(self,pressure):
        return self.baryon_density_0*baryon_ratio(np.log(pressure/self.pressure_0),self.gammas)
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
    def eosCs2(self,pressure):
        return 1.0/derivative(self.eosDensity,pressure,dx=pressure*dlnx_cs2)

def equations(gamma12,args,args_extra):
    gamma1,gamma2=gamma12
    #gamma1=float(gamma1)#Don't quite understand why? Maybe the above integration 'quad' only support float type, but not float64
    #gamma2=float(gamma2)
    n_s,n1,p1,e1,dpdn1,n2=args_extra
    p2,e2=args
    gamma0=np.log(dpdn1*n1/p1)
    spectra3=Spectral3([[n1,e1,p1],[gamma0,gamma1,gamma2]])
    return spectra3.eosBaryonDensity(p2)-n2, spectra3.eosDensity(p2)-e2

def logic_causal_p2(sol_x,init_i,args,args_extra):
    n_s,n1,p1,e1,dpdn1,n2=args_extra
    p2,e2=args
    gamma0=np.log(dpdn1*n1/p1)
    Gamma2=Gamma(np.log(p2/p1),[gamma0,sol_x[0],sol_x[1]])
    cs2=Gamma2*p2/(e2+p2)
    #print('cs2=%f'%cs2)
    return 0<cs2<=1 and np.max(np.abs(sol_x))<10

from eos_class import EOS_BPS
eos_bps=EOS_BPS()
ns=0.16
n0=0.06
p0=eos_bps.eosPressure_frombaryon(n0)
e0=eos_bps.eosDensity(p0)
dpdn0=eos_bps.eosCs2(p0)*eos_bps.eosChempo(p0)
gamma0=np.log(dpdn0*n0/p0)

import scipy.optimize as opt
n2=0.16
E_array=np.linspace(31,34,61)
L_array=np.linspace(20,80,121)
e2_array=(939-16+E_array)*n2
p2_array=L_array/3.*n2
result_x=np.zeros((len(E_array),len(L_array),2))
result_logic=np.zeros((len(E_array),len(L_array))).astype('bool')

import pickle
from scipy import interpolate
from solver_equations import solve_equations
#f=open('spectral3_match_gamma1gamma2_trial_E_28_38_L_5_200.dat','rb')
f=open('spectral3_match_gamma1gamma2_trial_E_30_34_L_20_120.dat','rb')
gamma12_trial=pickle.load(f)
f.close()
#E_trial = np.linspace(28,38,11)
#L_trial = np.linspace(5,200,40)
E_trial = np.linspace(30,34,41)
L_trial = np.linspace(20,120,101)
EL_trial = np.meshgrid(E_trial, L_trial)
gamma1_trial = interpolate.interp2d(EL_trial[0], EL_trial[1], gamma12_trial[0], kind='cubic')
gamma2_trial = interpolate.interp2d(EL_trial[0], EL_trial[1], gamma12_trial[1], kind='cubic')
def get_gamma12_trial(E,L):
    return [gamma1_trial(E,L)[0],gamma2_trial(E,L)[0]]

eos=[]
for i in range(len(e2_array)):
    for j in reversed(range(len(p2_array))):
        e2=e2_array[i]
        p2=p2_array[j]
# =============================================================================
#         result=opt.root(equations,np.array([gamma1_trial(E_array[i],L_array[j]),gamma2_trial(E_array[i],L_array[j])]),tol=1e-8,args=([p2,e2],[ns,n0,p0,e0,dpdn0,n2]))
#         result_x[i,j]=result.x
#         result_logic[i,j]=result.success
# =============================================================================
        try:
            result_logic[i,j],result_x[i,j]=solve_equations(equations,(2*result_x[i,j+1]-result_x[i,j+2],),[p2,e2],vary_list=np.array([1,0.5,0.3,0.2,0.1,0.08,-0.08,-0.1,-0.2,-0.3,-0.5,-1]),tol=1e-10,equations_extra_args=[ns,n0,p0,e0,dpdn0,n2])
        except:
            result_logic[i,j],result_x[i,j]=solve_equations(equations,(get_gamma12_trial(E_array[i],L_array[j]),),[p2,e2],vary_list=np.linspace(1.,1.,1),tol=1e-10,equations_extra_args=[ns,n0,p0,e0,dpdn0,n2])
        if(np.max(np.abs(equations(result_x[i,j],[p2,e2],[ns,n0,p0,e0,dpdn0,n2])))<1e-5):
            result_logic[i,j]=True
        else:
            result_logic[i,j],result_x[i,j]=solve_equations(equations,(result_x[i,j],),[p2,e2],vary_list=np.array([1,0.5,-0.5,-1]),tol=1e-10,equations_extra_args=[ns,n0,p0,e0,dpdn0,n2])
            print('E=%f,L=%f'%(E_array[i],L_array[j]))
            print('e=%f,p=%f'%(e2_array[i],p2_array[j]))
            print(result_x[i,j])
            print(equations(result_x[i,j],[p2,e2],[ns,n0,p0,e0,dpdn0,n2]))
            result_logic[i,j]=False
        eos.append(Spectral3([[n0,e0,p0],[gamma0,result_x[i,j,0],result_x[i,j,1]]]))
eos=np.array(eos).reshape(result_logic.shape)
eos=eos[:,::-1]
eos_shape=eos.shape
