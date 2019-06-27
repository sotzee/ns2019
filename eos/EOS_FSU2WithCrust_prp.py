#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:25:25 2019

@author: sotzee
"""

import numpy as np
from unitconvert import toMevfm,toMev4
from solver_equations import solve_equations,explore_solutions,Calculation_unparallel
from Parallel_process import main_parallel,main_parallel_unsave
from scipy.misc import derivative
import scipy.constants as const
from scipy.interpolate import interp1d
import pickle

def equations(x,args,args_extra):
    m_eff,J,L,self_W=args
    baryon_density_sat,bd_energy,incompressibility,mass_args=args_extra
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g2_Phi,g2_W,g2_rho,b,c,Lambda=x
    k_F=(3*np.pi**2*baryon_density_sat/2)**(1./3.)
    E_F=(k_F**2+m_eff**2)**0.5
    Phi_0=m-m_eff
    W_0=bd_energy-E_F
    n_scalar_sym=(m_eff/np.pi**2)*(E_F*k_F-m_eff**2*np.log((k_F+E_F)/m_eff))
    eq1=m*((m_Phi)**2*Phi_0/g2_Phi + (m*b)*Phi_0**2 + c*Phi_0**3 - n_scalar_sym)
    eq2=m*((m_W)**2*W_0/g2_W + (self_W/6)*W_0**3 - baryon_density_sat)
    tmp_2=g2_W/(m_W**2+self_W/2*g2_W*W_0**2)
    tmp_1=(incompressibility/(9*baryon_density_sat)-np.pi**2/(2*k_F*E_F)-tmp_2)
    eq3=m**2*(E_F**2*tmp_1*((m_Phi)**2/g2_Phi+2*b*m*Phi_0+3*c*Phi_0**2+(1/np.pi**2)*(k_F/E_F*(E_F**2+2*m_eff**2)-3*m_eff**2*np.log((k_F+E_F)/m_eff)))+m_eff**2)
    eq4=(1./(4*np.pi**2))*(k_F*E_F*(E_F**2+k_F**2)-m_eff**4*np.log((k_F+E_F)/m_eff))+ (Phi_0*m_Phi)**2/(2*g2_Phi) + (m*b)*Phi_0**3/3 + c*Phi_0**4/4 -(W_0*m_W)**2/(2*g2_W) - (self_W/6)*W_0**4/4 - baryon_density_sat*E_F
    tmp_J_0=k_F**2/(6*E_F)
    tmp_J_1=baryon_density_sat*g2_rho/(8*(m_rho)**2+16*Lambda*(W_0)**2*g2_rho) #origin fomula was baryon_density_sat/(8*(m_rho/g_rho)**2+16*Lambda*(W_0)**2)
    eq5=m**3*(tmp_J_0+tmp_J_1-J)
    eq6=m**3*(tmp_J_0*(1+(m_eff**2-3*baryon_density_sat*E_F*tmp_1)/E_F**2)+3*tmp_J_1*(1-32*tmp_2*W_0*Lambda*tmp_J_1)-L)
    return eq1,eq2,eq3,eq4,eq5,eq6

def logic_sol(sol_x,init_i,args):
    return sol_x[:3].min()>0 and sol_x[2].max()<250 #this 250 is tuned in order to make PNM energy density stay in range. using 1000 with make Max(PNM energy density) go to 

def eos_J_L_around_sym(baryon_density_sat,bd_energy,incompressibility,_args):
    mass_args,eos_args,args=_args
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args
    baryon_density_sat=baryon_density_s_MeV4
    bd_energy=m-BindE
    incompressibility=K
    m_eff=args[0]
    k_F=(3*np.pi**2*baryon_density_sat/2)**(1./3.)
    E_F=(k_F**2+m_eff**2)**0.5
    W_0=bd_energy-E_F
    tmp_2=g2_W/(m_W**2+self_W/2*g2_W*W_0**2)
    tmp_1=(incompressibility/(9*baryon_density_sat)-np.pi**2/(2*k_F*E_F)-tmp_2)
    tmp_J_0=k_F**2/(6*E_F)
    tmp_J_1=baryon_density_sat*g2_rho/(8*(m_rho)**2+16*Lambda*(W_0)**2*g2_rho) #origin fomula was baryon_density_sat/(8*(m_rho/g_rho)**2+16*Lambda*(W_0)**2)
    J=tmp_J_0+tmp_J_1
    L=tmp_J_0*(1+(m_eff**2-3*baryon_density_sat*E_F*tmp_1)/E_F**2)+3*tmp_J_1*(1-32*tmp_2*W_0*Lambda*tmp_J_1)
    return J,L

def Calculation_J_L_around_sym(eos_args_args):
    return eos_J_L_around_sym(baryon_density_s_MeV4,m-BindE,K,(mass_args,eos_args_args[:7],eos_args_args[7:]))

def equations_PNM(x,eos_args,args_extra):
    m_eff,W_0=x
    n,mass_args=args_extra
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args
    n_n=n
    n3=-n_n
    k_F_n=(3*np.pi**2*n_n)**(1./3)
    E_F_n=(k_F_n**2+m_eff**2)**0.5
    n_scalar=(m_eff/(2*np.pi**2))*((E_F_n*k_F_n-m_eff**2*np.log((k_F_n+E_F_n)/m_eff)))
    Phi_0=m-m_eff
    eq2=m*((m_Phi)**2*Phi_0/g2_Phi + (m*b)*Phi_0**2 + c*Phi_0**3 - n_scalar)
    rho_0=0.5*n3/((m_rho)**2/g2_rho + 2*Lambda*W_0**2)
    eq3=m*((m_W)**2*W_0/g2_W + (self_W/6)*W_0**3 + 2*Lambda*W_0*rho_0**2 - n)
    return eq2,eq3

def logic_sol_PNM(sol_x,init_i,args):
    return sol_x[1]>0 and sol_x[0]<m  and np.abs(sol_x[1]-init_i[1])/init_i[1]<0.10

def pressure_density_PNM(PNM_args,eos_args,extra_args):
    g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args
    n,mass_args=extra_args
    m_e,m,m_Phi,m_W,m_rho=mass_args
    m_eff,W_0=PNM_args
    n_n=n
    n3=-n_n
    k_F_n=(3*np.pi**2*n_n)**(1./3)
    E_F_n=(k_F_n**2+m_eff**2)**0.5
    Phi_0=m-m_eff
    rho_0=0.5*n3/((m_rho)**2/g2_rho + 2*Lambda*W_0**2)
    chempo_n=E_F_n+W_0-rho_0/2
    energy_density=((E_F_n*k_F_n**3+E_F_n**3*k_F_n-m_eff**4*np.log((k_F_n+E_F_n)/m_eff)))/(8*np.pi**2)+(m_Phi*Phi_0)**2/(2*g2_Phi)+(m_W*W_0)**2/(2*g2_W)+(m_rho*rho_0)**2/(2*g2_rho)+b*m*Phi_0**3/3+c*Phi_0**4/4+self_W*W_0**4/8+3*Lambda*(W_0*rho_0)**2
    pressure=chempo_n*n_n-energy_density
    return toMevfm(np.array([energy_density,pressure]),'mev4')

def eos_equations(y,eos_args,equations_extra_args):
    mass_args=equations_extra_args
    m_e,m,m_Phi,m_W,m_rho=mass_args
    n,g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args
    m_eff,W_0,k_F_n=y
    
    n_n=k_F_n**3/(3*np.pi**2)
    n_p=n-n_n
    n_p=(np.sign(n_p)+1)/2*np.abs(n_p)
    n3=n_p-n_n
    k_F_p=(3*np.pi**2*n_p)**(1./3)
    k_F_e=k_F_p
    E_F_e=(k_F_e**2+m_e**2)**0.5
    E_F_p=(k_F_p**2+m_eff**2)**0.5
    E_F_n=(k_F_n**2+m_eff**2)**0.5

# =============================================================================
#     if(m_eff<=0):
#         n_scalar=(m_eff/(2*np.pi**2))*((E_F_p*k_F_p)+(E_F_n*k_F_n))
#     else:
#         n_scalar=(m_eff/(2*np.pi**2))*((E_F_p*k_F_p-m_eff**2*np.log((k_F_p+E_F_p)/m_eff))+(E_F_n*k_F_n-m_eff**2*np.log((k_F_n+E_F_n)/m_eff)))
# =============================================================================
    n_scalar=(m_eff/(2*np.pi**2))*((E_F_p*k_F_p-m_eff**2*np.log((k_F_p+E_F_p)/m_eff))+(E_F_n*k_F_n-m_eff**2*np.log((k_F_n+E_F_n)/m_eff)))
    Phi_0=m-m_eff
    eq2=m*((m_Phi)**2*Phi_0/g2_Phi + (m*b)*Phi_0**2 + c*Phi_0**3 - n_scalar)
    rho_0=0.5*n3/((m_rho)**2/g2_rho + 2*Lambda*W_0**2)
    eq3=m*((m_W)**2*W_0/g2_W + (self_W/6)*W_0**3 + 2*Lambda*W_0*rho_0**2 - n)
    #eq5=((E_F_e*k_F_e**3+E_F_e**3*k_F_e-m_e**4*np.log((k_F_e+E_F_e)/m_e))+(E_F_p*k_F_p**3+E_F_p**3*k_F_p-m_eff**4*np.log((k_F_p+E_F_p)/m_eff))+(E_F_n*k_F_n**3+E_F_n**3*k_F_n-m_eff**4*np.log((k_F_n+E_F_n)/m_eff)))/(8*np.pi**2)+(m_Phi*Phi_0/g_Phi)**2/2+(m_W*W_0/g_W)**2/2+(m_rho*rho_0/g_rho)**2/2+b*m*Phi_0**3/3+c*Phi_0**4/4+self_W*W_0**4/8+3*Lambda*(W_0*rho_0)**2-energy_density
    chempo_e=E_F_e
    chempo_p=E_F_p+W_0+rho_0/2
    chempo_n=E_F_n+W_0-rho_0/2
    eq6=chempo_e+chempo_p-chempo_n
    #eq7=chempo_e*n_e+chempo_p*n_p+chempo_n*n_n-energy_density-pressure
    eq6=chempo_e+chempo_p-chempo_n
    return eq2,eq3,eq6

def eos_logic_sol(sol_x,init_i,args):
    return sol_x[1]>0 and sol_x[0]<m and sol_x[2]**3/(3*np.pi**2)<args[0] #and (np.abs(sol_x-init_i)/init_i).max()<0.1 # and np.abs((sol_x-init_i)/init_i).max()<0.1

def Calculation_parallel_sub(args_i,other_args):
    logic_calculatable_array_i,init_array_i,args_array_i=args_i
    #equations,vary_list,tol,logic_success_f=other_args
    vary_list,tol,equations_extra_args=other_args
    success_i,result_i=solve_equations(eos_equations,init_array_i[logic_calculatable_array_i],vary_list=vary_list,tol=tol,args=args_array_i,logic_success_f=eos_logic_sol,equations_extra_args=equations_extra_args)
    return [success_i]+result_i
    
def Calculation_parallel(equations,logic_calculatable_array,init_array,args_array,vary_list=np.linspace(1.,1.,1),tol=1e-12,logic_success_f=eos_logic_sol,equations_extra_args=[]):
    main_parallel_result=main_parallel_unsave(Calculation_parallel_sub,zip(logic_calculatable_array,init_array,args_array),other_args=(vary_list,tol,equations_extra_args))
    return main_parallel_result[:,0].astype('bool'),main_parallel_result[:,1:]

def eos_pressure_density(eos_array_args,eos_args_with_baryon,mass_args):
    m_e,m,m_Phi,m_W,m_rho=mass_args
    n,g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args_with_baryon
    m_eff,W_0,k_F_n=eos_array_args
    n_n=k_F_n**3/(3*np.pi**2)
    n_p=n-n_n
    n3=n_p-n_n
    n_e=n_p
    k_F_p=(3*np.pi**2*n_p)**(1./3)
    k_F_e=k_F_p
    E_F_e=(k_F_e**2+m_e**2)**0.5
    E_F_p=(k_F_p**2+m_eff**2)**0.5
    E_F_n=(k_F_n**2+m_eff**2)**0.5
    Phi_0=m-m_eff
    rho_0=0.5*n3/((m_rho)**2/g2_rho + 2*Lambda*W_0**2)
    chempo_e=E_F_e
    chempo_p=E_F_p+W_0+rho_0/2
    chempo_n=E_F_n+W_0-rho_0/2
    energy_density=((E_F_e*k_F_e**3+E_F_e**3*k_F_e-m_e**4*np.log((k_F_e+E_F_e)/m_e))+(E_F_p*k_F_p**3+E_F_p**3*k_F_p-m_eff**4*np.log((k_F_p+E_F_p)/m_eff))+(E_F_n*k_F_n**3+E_F_n**3*k_F_n-m_eff**4*np.log((k_F_n+E_F_n)/m_eff)))/(8*np.pi**2)+(m_Phi*Phi_0)**2/(2*g2_Phi)+(m_W*W_0)**2/(2*g2_W)+(m_rho*rho_0)**2/(2*g2_rho)+b*m*Phi_0**3/3+c*Phi_0**4/4+self_W*W_0**4/8+3*Lambda*(W_0*rho_0)**2
    pressure=chempo_e*n_e+chempo_p*n_p+chempo_n*n_n-energy_density
    return toMevfm(np.array([n,energy_density,pressure]),'mev4')

dir_name='RMF'
path='../'

baryon_density_s=0.16
m=939
BindE=16
K=240
mass_args=(0.5109989461,939,550,783,763)
baryon_density_s_MeV4=toMev4(baryon_density_s,'mevfm')
equations_extra_args=(baryon_density_s_MeV4,m-BindE,K,mass_args)

args=np.mgrid[0.5*939:0.8*939:16j,30:36:4j,0:120:13j,0:0.03:4j]
args_flat=args.reshape((-1,np.prod(np.shape(args)[1:]))).transpose()

J,m_eff,self_W,L=args
args_shape=np.shape(m_eff)

init_args= (90.,90.,90.,0.001,0.001,0.)
init_index=tuple(np.array(args_shape)/2)
eos_args_logic,eos_args=explore_solutions(equations,args,(init_index,),(init_args,),vary_list=np.array([1.]),tol=1e-12,logic_success_f=logic_sol,Calculation_routing=Calculation_unparallel,equations_extra_args=equations_extra_args)
eos_args=np.concatenate((eos_args,args[[3]]))
eos_args_flat=eos_args.reshape((-1,np.prod(args_shape))).transpose()
eos_args_logic_flat=eos_args_logic.astype(bool).flatten()

#for checking symmetric and pure neutron matter properities.
f_J_L_around_sym='./'+dir_name+'/RMF_J_L_around_sym.dat'
error_log=path+dir_name+'/error.log'
J_L_around_sym=main_parallel(Calculation_J_L_around_sym,np.concatenate((eos_args_flat,args_flat),axis=1)[eos_args_logic_flat],f_J_L_around_sym,error_log)

init_index=(0,-1,-1,0)
init_args=(args[0][init_index],384.5)
equations_PNM_extra_args=(baryon_density_s_MeV4,mass_args)
PNM_saturation_logic,PNM_args=explore_solutions(equations_PNM,eos_args,(init_index,),(init_args,),vary_list=np.array([1.]),tol=1e-12,logic_success_f=logic_sol_PNM,Calculation_routing=Calculation_unparallel,equations_extra_args=equations_PNM_extra_args)
PNM_saturation_logic=np.logical_and(PNM_saturation_logic,eos_args_logic)
PNM_saturation=pressure_density_PNM(PNM_args,eos_args,equations_PNM_extra_args)

N=100
baryon_density=baryon_density_s_MeV4*np.exp(np.linspace(0,np.log(12),N))
baryon_density=np.tile(baryon_density,np.concatenate(([1],args_shape,[1]))).transpose(np.concatenate(([0,len(args_shape)+1],np.array(range(1,len(args_shape)+1)))))
eos_args_with_baryon=np.tile(eos_args,np.concatenate(([N],np.full(len(args_shape)+1,1)))).transpose(np.concatenate(([1,0],np.array(range(2,len(args_shape)+2)))))
eos_args_with_baryon=np.concatenate((baryon_density,eos_args_with_baryon))

#init_index=tuple(np.array(args_shape)/2)
init_index=tuple(np.array(args_shape)/2)
init_args=(PNM_args[0][init_index],PNM_args[1][init_index],((3*np.pi**2)*baryon_density_s_MeV4)**0.33)
init_index=(0,)+init_index
eos_equations_extra_args=mass_args
eos_array_logic_,eos_array_args=explore_solutions(eos_equations,eos_args_with_baryon,(init_index,),(init_args,),vary_list=np.array([1.]),tol=1e-12,logic_success_f=eos_logic_sol,Calculation_routing=Calculation_parallel,equations_extra_args=eos_equations_extra_args)
eos_check_discontinuity=np.logical_or((np.abs((eos_array_args[:,1:]-eos_array_args[:,:-1])/eos_array_args[:,:-1]))<0.1,eos_array_args[:,1:]==0).min(axis=(0,1))
eos_array_logic=np.logical_and(PNM_saturation_logic,eos_check_discontinuity)
eos_array=eos_pressure_density(eos_array_args,eos_args_with_baryon,mass_args)#.reshape((3,N,-1)).transpose((2,1,0))

# =============================================================================
# f_eos_RMF='./'+dir_name+'/RMF_eos.dat'
# error_log=path+dir_name+'/error.log'
# eos_flat=main_parallel(Calculation_creat_eos_RMF,zip(args[:,eos_array_logic].transpose(),eos_args[:,eos_array_logic].transpose(),eos_array[:,:,eos_array_logic].transpose((2,0,1))),f_eos_RMF,error_log)
# 
# eos_success=[]
# matching_success=[]
# positive_pressure=[]
# for eos_i in eos_flat:
#     matching_success.append(eos_i.matching_success)
#     eos_success.append(eos_i.eos_success)
#     positive_pressure.append(eos_i.positive_pressure)
# eos_success=np.array(eos_success)
# matching_success=np.array(matching_success)
# positive_pressure=np.array(positive_pressure)
# print('len(eos)=%d'%len(eos_success))
# print('len(eos[positive_pressure])=%d'%len(positive_pressure[positive_pressure]))
# print('len(eos[matching_success])=%d'%len(matching_success[matching_success]))
# print('len(eos[eos_success])=%d'%len(eos_success[eos_success]))
# =============================================================================
