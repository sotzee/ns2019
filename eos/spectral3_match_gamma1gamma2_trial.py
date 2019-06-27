#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:05:17 2019

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
    e2,p2=args
    gamma0=np.log(dpdn1*n1/p1)
    spectra3=Spectral3([[n1,e1,p1],[gamma0,gamma1,gamma2]])
    return spectra3.eosBaryonDensity(p2)-n2, spectra3.eosDensity(p2)-e2

def logic_causal_p2(sol_x,init_i,args,args_extra):
    n_s,n1,p1,e1,dpdn1,n2=args_extra
    e2,p2=args
    gamma0=np.log(dpdn1*n1/p1)
    Gamma2=Gamma(np.log(p2/p1),[gamma0,sol_x[0],sol_x[1]])
    cs2=Gamma2*p2/(e2+p2)
    #print('cs2=%f'%cs2)
    return 0<cs2<=1 and np.max(np.abs(sol_x))<10

from eos_class import EOS_SLY4
eos_sly=EOS_SLY4()
ns=0.16
n0=0.06
p0=eos_sly.eosPressure_frombaryon(n0)
e0=eos_sly.eosDensity(p0)
dpdn0=eos_sly.eosCs2(p0)*eos_sly.eosChempo(p0)
gamma0=np.log(dpdn0*n0/p0)
n2=0.16
# =============================================================================
# p2=2.666667
# e2=152.8
# from solver_equations import solve_equations
# solve_equations(equations,[[1.,-0.5]],[p2,e2],vary_list=np.linspace(1.,1.,1),logic_success_f=logic_causal_p2,tol=1e-10,equations_extra_args=[ns,n0,p0,e0,dpdn0,n2])
# =============================================================================

from solver_equations import explore_solutions
args=np.mgrid[((939-16+28)*n2):((939-16+38)*n2):41j,(5/3.*n2):(200/3.*n2):196j]
init_index_tadpole=((-1,100),)
init_result_tadpole=(([1.0199972366419645, -0.24390483603142918]),)
result_logic,result_x=explore_solutions(equations,args,init_index_tadpole,init_result_tadpole,vary_list=np.array([1.,1.5,2]),equations_extra_args=[ns,n0,p0,e0,dpdn0,n2])#,logic_success_f=logic_causal_p2)

import pickle
f=open('spectral3_match_gamma1gamma2_trial_E_28_38_L_5_200.dat','wb')
pickle.dump(result_x,f)
f.close()

# =============================================================================
# import scipy.optimize as opt
# n2=0.16
# E_array=np.linspace(31,34,61)
# L_array=np.linspace(20,80,121)
# e2_array=(939-16+E_array)*n2
# p2_array=L_array/3.*n2
# result_x=np.zeros((len(E_array),len(L_array),2))
# result_logic=np.zeros((len(E_array),len(L_array))).astype('bool')
# 
# import pickle
# from scipy import interpolate
# from solver_equations import solve_equations
# #f=open('spectral3_match_gamma1gamma2_trial_E_28_38_L_5_200.dat','rb')
# f=open('spectral3_match_gamma1gamma2_trial_E_30_34_L_20_120.dat','rb')
# gamma12_trial=pickle.load(f)
# f.close()
# #E_trial = np.linspace(28,38,11)
# #L_trial = np.linspace(5,200,40)
# E_trial = np.linspace(30,34,41)
# L_trial = np.linspace(20,120,101)
# EL_trial = np.meshgrid(E_trial, L_trial)
# gamma1_trial = interpolate.interp2d(EL_trial[0], EL_trial[1], gamma12_trial[0], kind='cubic')
# gamma2_trial = interpolate.interp2d(EL_trial[0], EL_trial[1], gamma12_trial[1], kind='cubic')
# def get_gamma12_trial(E,L):
#     return [gamma1_trial(E,L)[0],gamma2_trial(E,L)[0]]
# 
# eos=[]
# for i in range(len(e2_array)):
#     for j in reversed(range(len(p2_array))):
#         e2=e2_array[i]
#         p2=p2_array[j]
# # =============================================================================
# #         result=opt.root(equations,np.array([gamma1_trial(E_array[i],L_array[j]),gamma2_trial(E_array[i],L_array[j])]),tol=1e-8,args=([p2,e2],[ns,n0,p0,e0,dpdn0,n2]))
# #         result_x[i,j]=result.x
# #         result_logic[i,j]=result.success
# # =============================================================================
#         try:
#             result_logic[i,j],result_x[i,j]=solve_equations(equations,(2*result_x[i,j+1]-result_x[i,j+2],),[p2,e2],vary_list=np.array([1,0.5,0.3,0.2,0.1,0.08,-0.08,-0.1,-0.2,-0.3,-0.5,-1]),tol=1e-10,equations_extra_args=[ns,n0,p0,e0,dpdn0,n2])
#         except:
#             result_logic[i,j],result_x[i,j]=solve_equations(equations,(get_gamma12_trial(E_array[i],L_array[j]),),[p2,e2],vary_list=np.linspace(1.,1.,1),tol=1e-10,equations_extra_args=[ns,n0,p0,e0,dpdn0,n2])
#         if(np.max(np.abs(equations(result_x[i,j],[p2,e2],[ns,n0,p0,e0,dpdn0,n2])))<1e-5):
#             result_logic[i,j]=True
#         else:
#             result_logic[i,j],result_x[i,j]=solve_equations(equations,(result_x[i,j],),[p2,e2],vary_list=np.array([1,0.5,-0.5,-1]),tol=1e-10,equations_extra_args=[ns,n0,p0,e0,dpdn0,n2])
#             print('E=%f,L=%f'%(E_array[i],L_array[j]))
#             print('e=%f,p=%f'%(e2_array[i],p2_array[j]))
#             print(result_x[i,j])
#             print(equations(result_x[i,j],[p2,e2],[ns,n0,p0,e0,dpdn0,n2]))
#             result_logic[i,j]=False
#         eos.append(Spectral3([[n0,e0,p0],[gamma0,result_x[i,j,0],result_x[i,j,1]]]))
# eos=np.array(eos).reshape(result_logic.shape)
# eos=eos[:,::-1]
# eos_shape=eos.shape
# =============================================================================


# =============================================================================
# args=np.mgrid[0.2:0.7:101j,-0.13:-0.01:121j]
# gamma1_array,gamma2_array=args
# eos_shape=gamma1_array.shape
# eos=[]
# for i in range(len(gamma1_array)):
#     for j in reversed(range(len(gamma2_array[0]))):
#         eos.append(Spectral3([[n0,e0,p0],[gamma0,gamma1_array[i,j],gamma2_array[i,j]]]))
# eos=np.array(eos).reshape(eos_shape)
# 
# logic_causal_at_p_300=[]
# logic_cant_reach_p_300=[]
# logic_neutron_matter_upper_p1_30=[]
# logic_neutron_matter_lower_p1_84=[]
# for eos_i in eos.flatten():
#     logic_causal_at_p_300.append(eos_i.eosCs2(300.)<=1)
#     logic_cant_reach_p_300.append(eos_i.eosBaryonDensity(300.)<0.16*50)
#     logic_neutron_matter_upper_p1_30.append(eos_i.eosBaryonDensity(30.)>0.16*1.85)
#     logic_neutron_matter_lower_p1_84.append(eos_i.eosBaryonDensity(8.4)<0.16*1.85)
# logic_causal_at_p_300=np.array(logic_causal_at_p_300).reshape(eos_shape)
# logic_cant_reach_p_300=np.array(logic_cant_reach_p_300).reshape(eos_shape)
# logic_at_p_300=np.logical_and(logic_causal_at_p_300,logic_cant_reach_p_300)
# logic_neutron_matter_upper_p1_30=np.array(logic_neutron_matter_upper_p1_30).reshape(eos_shape)
# logic_neutron_matter_lower_p1_84=np.array(logic_neutron_matter_lower_p1_84).reshape(eos_shape)
# logic_neutron_matter=np.logical_and(logic_neutron_matter_upper_p1_30,logic_neutron_matter_lower_p1_84)
# logic_neutron_matter_causal_at_p_300=np.logical_and(logic_at_p_300,logic_neutron_matter)
# =============================================================================


#check with Lee Lindblom 2018 on ap4 and mpa1, these two has small gamma3 as we neglect here
# =============================================================================
# path='../EOS_Tables_Ozel/'
# eos=[]
# EOS_LIST=['ap4','mpa1']
# EOS_ns_LIST=[0.16,0.16]
# for EOS_i,EOS_ns_i in zip(EOS_LIST[:2],EOS_ns_LIST[:2]):
#     eos_array_i=np.loadtxt(path+EOS_i+'.dat',skiprows=0)
#     nb=toMevfm(eos_array_i[:,0]/1.66*1e24,'baryondensity')
#     pr=toMevfm(eos_array_i[:,1],'density')
#     ep=toMevfm(eos_array_i[:,2],'density')
#     chempo_min=(ep[0]+pr[0])
#     eos.append(EOS_intepolation(chempo_min,EOS_ns_i,nb,ep,pr))
# 
# from physicalconst import mass_per_baryon
# gammas_list=[[0.8651, 0.1548, -0.0151],[1.0215, 0.1653, -0.0235]]
# for i in range(2):
#     n0=toMevfm(2e14/mass_per_baryon,'baryondensity')
#     p0=eos[i].eosPressure_frombaryon(n0)
#     e0=eos[i].eosDensity(p0)
#     eos.append(Spectral3([[n0,e0,p0],gammas_list[i]]))
#     EOS_LIST.append(EOS_LIST[i]+'_Spectral3')
# import show_properity as sp
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(1, 1,figsize=(8,6),sharex=True,sharey=True)
# sp.show_eos_unparallel(axes,eos,0,1,500,pressure_range=[1,300,'log'],legend=EOS_LIST)
# =============================================================================
