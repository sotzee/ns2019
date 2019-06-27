#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:09:01 2019

@author: sotzee
"""


import pickle
import numpy as np
import os
def ensure_dir(path,dir_name):
    try:
        os.stat(path+dir_name)
    except:
        os.mkdir(path+dir_name)

def pickle_load(path,dir_name,data_name_list):
    data=[]
    for data_name_i in data_name_list:
        f=open(path+dir_name+'/'+data_name_i+'.dat','rb')
        data.append(pickle.load(f))
        f.close()
    return data

def pickle_dump(path,dir_name,tuple_data_dataname):
    for data_i,data_name_i in tuple_data_dataname:
        f=open(path+dir_name+'/'+data_name_i+'.dat','wb')
        pickle.dump(data_i,f)
        f.close()
        

index_roundoff_compensate=2e-14
def log_array(array_lim,delta_factor,N): #delta factor is (y_N-y_{N-1}/(y1-y0)
    k=np.log(delta_factor)/(N-1)
    a=(array_lim[1]-array_lim[0])/(np.exp(k*N)-1)
    return a,k,array_lim[0]+a*(np.exp(np.linspace(0,N*k,N+1))-1)
def log_array_extend(array_lim_min,a,k,N): #delta factor is (y_N-y_{N-1}/(y1-y0)
    return array_lim_min+a*(np.exp(np.linspace(0,N*k,N+1))-1)
def log_index(array_i,array_lim_min,a,k,N):
    array_i=array_i-index_roundoff_compensate
    #print(array_i,array_lim,a,k,-(-np.log(np.where(array_i<array_lim.min(),0,(array_i-array_lim.min())/a)+1)/k).astype('int'))
    return N-1-(N-np.log(np.where(array_i<array_lim_min,0,(array_i-array_lim_min)/a)+1)/k).astype('int')

import scipy.optimize as opt
def log_array_centered(array_lim,delta_factor,N1_N2):
    N1,N2=N1_N2
    a,k,right_side=log_array(array_lim[1:3],delta_factor,N2)
    def eq(k_,a,k):
        return a*k*(np.exp(k_*N1)-1)-(array_lim[1]-array_lim[0])*k_
    if(right_side[1]-right_side[0]>(array_lim[1]-array_lim[0])/N1):
        left_side=np.linspace(array_lim[0],array_lim[1],N1+1)[:-1]
    else:
        k_=opt.newton(eq,k,args=(a,k))
        a_=a*k/k_
        left_side = log_array_extend(array_lim[1],-a_,k_,N1)[::-1][:-1]
    return np.concatenate((left_side,right_side))
    