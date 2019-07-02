#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:25:25 2019

@author: sotzee
"""

from eos_class import EOS_Spectral3_match,EOS_SLY4
import numpy as np
from toolbox import pickle_load,pickle_dump
dir_name='FSU2'
dir_prp='prp'
path='../'
args,eos_array_logic,eos_array_list_success=pickle_load(path+dir_name+'/',dir_prp,['args','eos_array_logic','eos_array_list_success'])
eos_shape=args[0].shape

from Parallel_process import main_parallel_unsave
def Calculation_creat_EOS(eos_args_args_array,other):
    return EOS_Spectral3_match(eos_args_args_array,other)
eos_sly4=EOS_SLY4()
eos_success=main_parallel_unsave(Calculation_creat_EOS,eos_array_list_success,other_args=EOS_SLY4())

pickle_dump(path,dir_name,([eos_success,'eos_success'],))
# =============================================================================
# eos_sly4=EOS_SLY4()
# eos_success=[]
# for eos_array_list_success_i in eos_array_list_success:
#     eos_success.append(EOS_Spectral3_match(eos_array_list_success_i,eos_sly4))
# eos_success=np.array(eos_success)
# =============================================================================

success_match=[]
causal_match=[]
p_185_success=[]
success_eos=[]
for i in range(len(eos_success)):
    success_match.append(eos_success[i].success_match)
    causal_match.append(eos_success[i].causal_match)
    success_eos.append(eos_success[i].success_eos)
    p_185_success.append(eos_success[i].eosPressure_frombaryon(0.16*1.85))
logic_success_match=np.copy(eos_array_logic)
logic_success_match[eos_array_logic]=success_match
logic_causal_match=np.copy(eos_array_logic)
logic_causal_match[eos_array_logic]=causal_match
logic_success_causal_match=np.logical_and(logic_success_match,logic_causal_match)
logic_success_eos=np.copy(logic_success_causal_match)
logic_success_eos[eos_array_logic]=np.logical_and(logic_success_causal_match[eos_array_logic],success_eos)
p_185=np.zeros(eos_shape)
p_185[eos_array_logic]=p_185_success

print(eos_array_logic[eos_array_logic].shape)
print(logic_success_match[logic_success_match].shape)
print(logic_causal_match[logic_causal_match].shape)
print(logic_success_causal_match[logic_success_causal_match].shape)
print(logic_success_eos[logic_success_eos].shape)

eos_success=eos_success[logic_success_eos[eos_array_logic]]

pickle_dump(path,dir_name,([eos_success,'eos_success'],[args,'args'],[logic_success_eos,'logic_success_eos'],[p_185,'p185']))

#eos=eos_flat.reshape(eos_shape)


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
