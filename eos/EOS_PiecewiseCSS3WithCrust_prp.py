#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:26:14 2019

@author: sotzee
"""

from eos_class import EOS_PiecewiseCSS3WithCrust
import numpy as np
from toolbox import log_array
#args=np.mgrid[0.06:0.08:2j,0.01:0.5:10j,0.1:1.:10j,0.1:1.0:19j]

args=np.array(np.meshgrid(np.array([0.06,0.08]),log_array([0.01,0.5],20,60)[2],log_array([0.1,1],5,30)[2],np.linspace(0.1,1,19)))
eos_shape=args[0].shape

def Calculation_creat_EOS(eos_args_args_array,other):
    return EOS_PiecewiseCSS3WithCrust(eos_args_args_array)
from Parallel_process import main_parallel_unsave
eos_flat=main_parallel_unsave(Calculation_creat_EOS,args.reshape((4,-1)).transpose())
eos=eos_flat.reshape(eos_shape)

eos_array_list_success=[]
p_185=[]
logic_success_eos=[]
for eos_i in eos_flat:
    if(eos_i.success_eos):
        p_185.append(eos_i.eosPressure_frombaryon(0.16*1.85))
    else:
        p_185.append(0)
    logic_success_eos.append(eos_i.success_eos)
p_185=np.array(p_185).reshape(eos_shape)
unitary_gas_constrain=p_185>3.74
neutron_matter_constrain=np.logical_and(p_185>8.4,p_185<30.)
logic_success_eos=np.array(logic_success_eos).reshape(eos_shape)

dir_name='PiecewiseCSS3'
path='../'
from toolbox import ensure_dir,pickle_dump
ensure_dir(path,dir_name)
pickle_dump(path,dir_name,([eos[logic_success_eos],'eos_success'],[args,'args'],[logic_success_eos,'logic_success_eos'],[p_185,'p185']))

# =============================================================================
# p185=[]
# for cs2_1 in log_array([0.01,0.5],20,40)[2]:
#     a=EOS_PiecewiseCSS3WithCrust([0.06,cs2_1,1,0.3])
#     p185.append(a.eosPressure_frombaryon(0.16*1.85))
#     plt.plot([0],[p185],'.')
#     plt.ylim([0,30])
# p185=np.array(p185)
# =============================================================================

# =============================================================================
# p185=[]
# p37=[]
# p74=[]
# for eos_flat_i in eos_flat:
#     p185.append(eos_flat_i.eosPressure_frombaryon(0.16*1.85))
#     p37.append(eos_flat_i.eosPressure_frombaryon(0.16*3.7))
#     p74.append(eos_flat_i.eosPressure_frombaryon(0.16*7.4))
# p185=np.array(p185)
# p37=np.array(p37)
# p74=np.array(p74)
# =============================================================================

# =============================================================================
# import matplotlib.pyplot as plt
# import show_properity as sp
# fig, axes = plt.subplots(1, 1,figsize=(8,6),sharex=True,sharey=True)
# sp.show_eos(axes,eos.flatten(),0,1,100,pressure_range=[0.01,500,'log'])
# =============================================================================
