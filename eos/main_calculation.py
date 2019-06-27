#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:44:16 2019

@author: sotzee
"""

import numpy as np
from toolbox import pickle_load,pickle_dump
# =============================================================================
# dir_name='Spectral3'
# path='../'
# need_intepolation=True
# =============================================================================

# =============================================================================
# dir_name='PiecewiseCSS3'
# from eos_class import EOS_PiecewiseCSS3WithCrust
# path='../'
# need_intepolation=False
# =============================================================================

dir_name='PiecewisePoly3'
from eos_class import EOS_PiecewisePoly3WithCrust
path='../'
need_intepolation=False

if(need_intepolation):
    eos_array_list_success,args,logic_success_eos,p_185=pickle_load(path,dir_name,['eos_array_list_success','args','logic_success_eos','p185'])
    unitary_gas_constrain=p_185>3.74
    neutron_matter_constrain=np.logical_and(p_185>8.4,p_185<30.)
    
    from eos_class import EOS_interpolation
    eos_success=[]
    for eos_array_i in eos_array_list_success:
        eos_success.append(EOS_interpolation(0.16,eos_array_i))
    eos_success=np.array(eos_success)
else:
    eos_success,args,logic_success_eos,p_185=pickle_load(path,dir_name,['eos_success','args','logic_success_eos','p185'])
    unitary_gas_constrain=p_185>3.74
    neutron_matter_constrain=np.logical_and(p_185>8.4,p_185<30.)
eos_shape=args[0].shape

from Parallel_process import main_parallel
from Hadronic_calculation import Calculation_maxmass,Calculation_onepointfour,Calculation_one,Calculation_MRBIT,Calculation_chirpmass_Lambdabeta6

print('Calculating Maxmass of %d EoS calculated'%(len(eos_success)))
f_maxmass_result=path+dir_name+'/hadronic_calculation_maxmass.dat'
error_log=path+dir_name+'/hadronic_calculation_maxmass.log'
maxmass_result=np.full(eos_shape+(3,),np.array([0,0,1]),dtype='float')
maxmass_result[logic_success_eos]=main_parallel(Calculation_maxmass,eos_success,f_maxmass_result,error_log)
maxmass_result=maxmass_result.transpose((len(eos_shape),)+tuple(range(len(eos_shape))))
logic_maxmass=maxmass_result[1]>=2
print('Maximum mass constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_success),len(eos_success[logic_maxmass[logic_success_eos]])))
logic_causality=maxmass_result[2]<1
print('Causality constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_success),len(eos_success[logic_causality[logic_success_eos]])))
logic=np.logical_and(logic_maxmass,logic_causality)
print('Maximum mass and causality constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_success),len(eos_success[logic[logic_success_eos]])))

print('Setting Maxmass of %d EoS calculated'%(len(eos_success)))
setMaxmass_result=[]
for eos_success_i,maxmass_result_i in zip(eos_success,maxmass_result[:,logic_success_eos].transpose()):
    setMaxmass_result.append(eos_success_i.setMaxmass(maxmass_result_i))
logic=np.copy(logic_success_eos)
logic[logic_success_eos]=np.array(setMaxmass_result)
print('%d EoS success by.' %(len(eos_success[logic[logic_success_eos]])))
print('Dumping to %s'%(path+dir_name))
pickle_dump(path,dir_name,([maxmass_result,'maxmass_result'],[logic,'logic']))

print('Calculating properities of 1.4 M_sun star for %d EoS.'%len(eos_success[logic[logic_success_eos]]))
f_onepointfour_result=path+dir_name+'/hadronic_calculation_onepointfour.dat'
error_log=path+dir_name+'/hadronic_calculation_onepointfour.log'
Properity_onepointfour=np.full(eos_shape+(8,),np.array([0,0,0,0,0,0,0,0]),dtype='float').transpose((-1,)+tuple(range(len(eos_shape))))
Properity_onepointfour=main_parallel(Calculation_onepointfour,eos_success[logic[logic_success_eos]],f_onepointfour_result,error_log)
print('properities of 1.4 M_sun star of %d EoS calculated.' %(len(eos_success[logic[logic_success_eos]])))

print('Calculating properities of 1.0 M_sun star for %d EoS.'%len(eos_success[logic[logic_success_eos]]))
f_one=path+dir_name+'/hadronic_calculation_one.dat'
error_log=path+dir_name+'/hadronic_calculation_one.log'
Properity_one=np.full(eos_shape+(8,),np.array([0,0,0,0,0,0,0,0]),dtype='float').transpose((-1,)+tuple(range(len(eos_shape))))
Properity_one=main_parallel(Calculation_one,eos_success[logic[logic_success_eos]],f_one,error_log)
print('properities of 1.0 M_sun star of %d EoS calculated.' %(len(eos_success[logic[logic_success_eos]])))

print('Setting properity of 1.0 and 1.4 M_sun star of %d EoS calculated'%len(eos_success[logic[logic_success_eos]]))
for eos_success_i,Properity_one_i,Properity_onepointfour_i in zip(eos_success[logic[logic_success_eos]],Properity_one,Properity_onepointfour):
    eos_success_i.setProperity(Properity_one_i,Properity_onepointfour_i)
#pickle_dump(path,dir_name,([eos_success,'eos_success'],))

print('Calculating MRBIT for %d EoS.'%len(eos_success[logic[logic_success_eos]]))
f_MRBIT=path+dir_name+'/hadronic_calculation_MRBIT.dat'
error_log=path+dir_name+'/hadronic_calculation_MRBIT.log'
MRBIT_result=main_parallel(Calculation_MRBIT,eos_success[logic[logic_success_eos]],f_MRBIT,error_log)
print('mass, compactness and tidal Lambda of %d EoS calculated.' %len(eos_success[logic[logic_success_eos]]))


print('Calculating binary neutron star...')
mass_beta_Lambda_result=MRBIT_result[:,:3]
f_chirpmass_Lambdabeta6_result=path+dir_name+'/hadronic_calculation_chirpmass_Lambdabeta6.dat'
error_log=path+dir_name+'hadronic_calculation_chirpmass_Lambdabeta6.log'
chirp_q_Lambdabeta6_Lambda1Lambda2=main_parallel(Calculation_chirpmass_Lambdabeta6,np.concatenate((mass_beta_Lambda_result,np.tile(Properity_onepointfour[:,3],(mass_beta_Lambda_result.shape[-1],1,1)).transpose()),axis=1),f_chirpmass_Lambdabeta6_result,error_log)

# =============================================================================
# from Parallel_process import main_parallel_unsave
# def Calculation_eos(eos_args_args_array,eos_low):
#     return EOS_Spectral3(eos_args_args_array,eos_low)
# from Parallel_process import main_parallel_unsave
# eos_flat=main_parallel_unsave(Calculation_creat_EOS_Spectral3,args.reshape((3,-1)).transpose(),other_args=eos_sly4)
# eos=eos_flat.reshape(eos_shape)
# =============================================================================
