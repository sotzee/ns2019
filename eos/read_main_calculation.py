#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 19:39:02 2019

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
        eos_success.append(EOS_interpolation(0.06,eos_array_i))
    eos_success=np.array(eos_success)
else:
    eos_success,args,logic_success_eos,p_185=pickle_load(path,dir_name,['eos_success','args','logic_success_eos','p185'])
    unitary_gas_constrain=p_185>3.74
    neutron_matter_constrain=np.logical_and(p_185>8.4,p_185<30.)
eos_shape=args[0].shape


logic_success_eos,logic,maxmass_result,Properity_one,Properity_onepointfour,MRBIT_result=pickle_load(path,dir_name,['logic_success_eos','logic','maxmass_result','hadronic_calculation_one','hadronic_calculation_onepointfour','hadronic_calculation_MRBIT'])

setMaxmass_result=[]
for eos_success_i,maxmass_result_i in zip(eos_success,maxmass_result[:,logic_success_eos].transpose()):
    setMaxmass_result.append(eos_success_i.setMaxmass(maxmass_result_i))

for eos_success_i,Properity_one_i,Properity_onepointfour_i in zip(eos_success[logic[logic_success_eos]],Properity_one,Properity_onepointfour):
    eos_success_i.setProperity(Properity_one_i,Properity_onepointfour_i)

# =============================================================================
# eos_ap4=EOS_AP4()
# eos_sly4=EOS_SLY4()
# eos_ap4_spl=EOS_AP4(s_k=[0,3])
# eos_sly4_spl=EOS_SLY4(s_k=[0,3])
# import matplotlib.pyplot as plt
# import show_properity as sp
# fig, axes = plt.subplots(1, 1,figsize=(8,6),sharex=True,sharey=True)
# sp.show_eos(axes,[eos_ap4,eos_sly4,eos_ap4_spl,eos_sly4_spl],0,5,500,pressure_range=[0.01,0.1,'log'],legend=[1,2,3,4])
# =============================================================================
