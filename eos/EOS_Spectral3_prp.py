#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:35:21 2019

@author: sotzee
"""

from eos_class import EOS_SLY4,EOS_Spectral3
import numpy as np

eos_sly4=EOS_SLY4()
args=np.mgrid[0.06:0.08:2j,0:0.8:401j,-0.13:0.04:341j]
args=args[:,:,:,args[2,0,0]!=0]
eos_shape=args[0].shape

def Calculation_creat_EOS_Spectral3(eos_args_args_array,eos_low):
    return EOS_Spectral3(eos_args_args_array,eos_low)
from Parallel_process import main_parallel_unsave
eos_flat=main_parallel_unsave(Calculation_creat_EOS_Spectral3,args.reshape((3,-1)).transpose(),other_args=eos_sly4)
eos=eos_flat.reshape(eos_shape)

eos_array_list_success=[]
p_185=[]
logic_success_eos=[]
for eos_i in eos_flat:
    if(eos_i.success_eos):
        p_185.append(eos_i.eosPressure_frombaryon(0.16*1.85))
        eos_array_list_success.append(eos_i.eos_array)
    else:
        p_185.append(0)
    logic_success_eos.append(eos_i.success_eos)
p_185=np.array(p_185).reshape(eos_shape)
unitary_gas_constrain=p_185>3.74
neutron_matter_constrain=np.logical_and(p_185>8.4,p_185<30.)
logic_success_eos=np.array(logic_success_eos).reshape(eos_shape)

dir_name='Spectral3'
path='../'
from toolbox import ensure_dir,pickle_dump
ensure_dir(path,dir_name)
pickle_dump(path,dir_name,([eos_array_list_success,'eos_array_list_success'],[args,'args'],[logic_success_eos,'logic_success_eos'],[p_185,'p185']))


#import matplotlib.pyplot as plt
#plt.plot(args[1,logic_success_eos],args[2,logic_success_eos],'.')

#import show_properity as sp
#fig, axes = plt.subplots(1, 1,figsize=(8,6),sharex=True,sharey=True)
#sp.show_eos(axes,eos[logic_success_eos],0,5,500,pressure_range=[0.01,200,'log'])
