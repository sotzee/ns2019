#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:35:33 2019

@author: sotzee
"""
import numpy as np
from MassRadius_hadronic import MassRadius
from FindMaxmass import Maxmass
from Find_OfMass import Properity_ofmass
from toolbox import log_array_centered
Preset_Pressure_final=1e-8
Preset_rtol=1e-4

def Calculation_maxmass(eos_i):
    try:
        maxmass_result=Maxmass(Preset_Pressure_final,Preset_rtol,eos_i)[1:3]
        maxmass_result+=[eos_i.eosCs2(maxmass_result[0])]
    except RuntimeWarning:
        print('Runtimewarning happens at calculating max mass:')
        print(eos_i.args)
    return maxmass_result

def Calculation_onepointfour(eos_i):
    try:
        Properity_onepointfour=Properity_ofmass(1.4,10.,eos_i.pc_max,MassRadius,Preset_Pressure_final,Preset_rtol,1,eos_i)
    except:
        Properity_onepointfour=Properity_ofmass(1.4,1.,eos_i.pc_max,MassRadius,Preset_Pressure_final,Preset_rtol,1,eos_i)
    return Properity_onepointfour

def Calculation_one(eos_i):
    try:
        Properity_one=Properity_ofmass(1.0,10.,eos_i.pc_max,MassRadius,Preset_Pressure_final,Preset_rtol,1,eos_i)
    except:
        Properity_one=Properity_ofmass(1.0,1.,eos_i.pc_max,MassRadius,Preset_Pressure_final,Preset_rtol,1,eos_i)
    return Properity_one

def Calculation_mass_beta_Lambda(eos_i,pc_list=10**np.linspace(0,-1.5,40)):
    mass,beta,Lambda=[[],[],[]]
    for pc_i in pc_list:
        try:
            MR_result=MassRadius(eos_i.pc_max*pc_i,Preset_Pressure_final,Preset_rtol,'MRT',eos_i)
        except RuntimeWarning:
            print('Runtimewarning happens at calculating max mass:')
        mass.append(MR_result[0])
        beta.append(MR_result[2])
        Lambda.append(MR_result[4])
    return [mass,beta,Lambda]
def Calculation_MRBIT(eos_i,delta_factor=20,N1_N2=[10,20]):
    mass,radius,beta,M_binding,momentofinertia,k2,tidal=[[],[],[],[],[],[],[]]
    pc_list=log_array_centered([eos_i.properity_one[0],eos_i.properity_onepointfour[0],eos_i.pc_max],delta_factor,N1_N2)
    for pc_i in pc_list:
        try:
            MR_result=MassRadius(pc_i,Preset_Pressure_final,Preset_rtol,'MRBIT',eos_i)
        except RuntimeWarning:
            print('Runtimewarning happens at calculating max mass:')
        mass.append(MR_result[0])
        radius.append(MR_result[1])
        beta.append(MR_result[2])
        M_binding.append(MR_result[3])
        momentofinertia.append(MR_result[4])
        k2.append(MR_result[5])
        tidal.append(MR_result[6])
    return [mass,radius,beta,M_binding,momentofinertia,k2,tidal]

def mass_chirp(mass1,mass2):
    return (mass1*mass2)**0.6/(mass1+mass2)**0.2
def tidal_binary(q,tidal1,tidal2):
    return 16.*((12*q+1)*tidal1+(12+q)*q**4*tidal2)/(13*(1+q)**5)
def Calculation_chirpmass_Lambdabeta6(args_i,M_min=1.1,M_max=3.0):
    mass_i=args_i[0,:]
    Lambda_i=args_i[2,:]
    beta_onepointfour_i=args_i[3,0]
    logic_mass=np.logical_and(mass_i>M_min,mass_i<M_max)
    mass1,mass2 = np.meshgrid(mass_i[logic_mass],mass_i[logic_mass])
    Lambda1,Lambda2 = np.meshgrid(Lambda_i[logic_mass],Lambda_i[logic_mass])
    q=mass2/mass1
    chirp_mass=mass_chirp(mass1,mass2).flatten()
    Lambda_binary_beta6=(beta_onepointfour_i/1.4*chirp_mass)**6*tidal_binary(q,Lambda1,Lambda2).flatten()
    q=q.flatten()
    Lambda2Lambda1=(Lambda2/Lambda1).flatten()
    return [chirp_mass,q,Lambda_binary_beta6,Lambda2Lambda1]