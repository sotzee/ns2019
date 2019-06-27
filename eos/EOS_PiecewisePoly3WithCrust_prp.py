#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:27:26 2019

@author: sotzee
"""

from eos_class import EOS_PiecewisePoly3WithCrust
import numpy as np

args=np.mgrid[0.06:0.06:1j,3.75:30:43j,100:300:41j,0:1:40j]
args[3]=1-250+450*5**args[3]
eos_shape=args[0].shape

def Calculation_creat_EOS(eos_args_args_array,other):
    return EOS_PiecewisePoly3WithCrust(eos_args_args_array)
from Parallel_process import main_parallel_unsave
eos_flat=main_parallel_unsave(Calculation_creat_EOS,args.reshape((4,-1)).transpose())
eos=eos_flat.reshape(eos_shape)

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

dir_name='PiecewisePoly3'
path='../'
from toolbox import ensure_dir,pickle_dump
ensure_dir(path,dir_name)
pickle_dump(path,dir_name,([eos[logic_success_eos],'eos_success'],[args,'args'],[logic_success_eos,'logic_success_eos'],[p_185,'p185']))
