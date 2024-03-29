from scipy import interpolate
import scipy.integrate
import scipy.special
import numpy as np
from unitconvert import toMevfm
from scipy.misc import derivative
from scipy.constants import c,G,e
dlnx_cs2=1e-6


def quality_control_basic(eos_array):
    logic_faithful_range=np.logical_and(eos_array[0]<3,eos_array[2]<3000)
    logic_positive=(eos_array>0).min(axis=0)
    eos_array=eos_array[:,np.logical_and(logic_faithful_range,logic_positive)]
    logic_mono=np.concatenate((np.array([True]),(eos_array[:,1:]-eos_array[:,:-1]>0).min(axis=0)))
    eos_array=eos_array[:,logic_mono]
    eos_array_low=eos_array[:,eos_array[2]<200]
    success_causality=(((eos_array_low[2,1:]-eos_array_low[2,:-1])/(eos_array_low[1,1:]-eos_array_low[1,:-1]))<1).min()
    success_stiff_enough=eos_array_low[0,-1]<1.6
    success_baryondensity_range=eos_array[0,-1]>0.3
    success_pressure_range=eos_array[2,0]<1e-8 and eos_array[2,-1]>200
    success_eos=success_causality and success_stiff_enough and success_baryondensity_range and success_pressure_range
    #print(success_causality , success_stiff_enough , success_baryondensity_range , success_pressure_range)
    return success_eos,eos_array
# =============================================================================
# from scipy.interpolate import interp1d
# class EOS_interpolation(object):
#     def __init__(self,baryon_density_s,eos_array,quality_control_f=quality_control_basic,s_k=[0,3]):
#         self.success,self.eos_array=quality_control_f(eos_array)
#         n_array,energy_array,pressure_array=self.eos_array
#         if(self.success):
#             self.eosPressure_frombaryon = interp1d(n_array,pressure_array, kind='quadratic')
#             self.eosDensity  = interp1d(pressure_array,energy_array, kind='quadratic')
#             self.eosBaryonDensity = interp1d(pressure_array,n_array, kind='quadratic')
#             self.chempo_surface=(pressure_array[0]+energy_array[0])/n_array[0]
#             self.baryon_density_s=baryon_density_s
#             self.pressure_s=self.eosPressure_frombaryon(self.baryon_density_s)
#             self.density_s=self.eosDensity(self.pressure_s)
#             self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
#             self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5
#             self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
#         else:
#             self.eosPressure_frombaryon,self.eosDensity,self.eosBaryonDensity = ['undefined due to bad eos_array']*3
#     def __getstate__(self):1.60000000e-01
#         state = self.__dict__.copy()
#         #print(state)
#         for dict_intepolation in ['eosPressure_frombaryon','eosDensity','eosBaryonDensity']:
#             del state[dict_intepolation]
#         return state
#     def __setstate__(self, state):
#         self.__dict__.update(state)
#         n_array,energy_array,pressure_array=self.eos_array
#         self.eosPressure_frombaryon = interp1d(n_array,pressure_array, kind='quadratic')
#         self.eosDensity  = interp1d(pressure_array,energy_array, kind='quadratic')
#         self.eosBaryonDensity = interp1d(pressure_array,n_array, kind='quadratic')
#     def eosChempo(self,pressure):
#         return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
#     def eosCs2(self,pressure):
#         return 1.0/derivative(self.eosDensity,pressure,dx=pressure*dlnx_cs2)
# =============================================================================

class EOS_interpolation(object):
    def __init__(self,baryon_density_s,eos_array,quality_control_f=quality_control_basic,s_k=[0,3]): #defalt s=0,k=3 equal quadratic 1d intepolation
        self.success_eos,self.eos_array=quality_control_f(eos_array)
        self.p_max=self.eos_array[2,-1]
        n_array,energy_array,pressure_array=self.eos_array
        self.s,self.k=s_k
        if(self.success_eos):
            self.eosPressure_frombaryon = interpolate.UnivariateSpline(n_array,pressure_array, k=self.k,s=self.s)
            self.eosDensity  = interpolate.UnivariateSpline(pressure_array,energy_array, k=self.k,s=self.s)
            self.eosBaryonDensity = interpolate.UnivariateSpline(pressure_array,n_array, k=self.k,s=self.s)
            self.chempo_surface=(pressure_array[0]+energy_array[0])/n_array[0]
            self.baryon_density_s=baryon_density_s
            self.pressure_s=self.eosPressure_frombaryon(self.baryon_density_s)
            self.density_s=self.eosDensity(self.pressure_s)
            self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
            self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5
            self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
        else:
            self.eosPressure_frombaryon,self.eosDensity,self.eosBaryonDensity = ['undefined due to bad eos_array']*3
    def __getstate__(self):
        state = self.__dict__.copy()
        for dict_intepolation in ['eosPressure_frombaryon','eosDensity','eosBaryonDensity']:
            del state[dict_intepolation]
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        n_array,energy_array,pressure_array=self.eos_array
        self.eosPressure_frombaryon = interpolate.UnivariateSpline(n_array,pressure_array, k=self.k,s=self.s)
        self.eosDensity  = interpolate.UnivariateSpline(pressure_array,energy_array, k=self.k,s=self.s)
        self.eosBaryonDensity = interpolate.UnivariateSpline(pressure_array,n_array, k=self.k,s=self.s)
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
    def eosCs2(self,pressure):
        return 1.0/derivative(self.eosDensity,pressure,dx=pressure*dlnx_cs2)
    def setMaxmass(self,result_maxmaxmass):
        self.pc_max,self.mass_max,self.cs2_max=result_maxmaxmass
        self.maxmass_success=self.cs2_max<1 and 2<self.mass_max<5 and self.pc_max<self.p_max
        self.eos_success_all=self.maxmass_success and self.success_eos
        return self.eos_success_all
    def setProperity(self,Properity_one,Properity_onepointfour):
        self.properity_one,self.properity_onepointfour=Properity_one,Properity_onepointfour

class EOS_SLY4(EOS_interpolation):
    def __init__(self,s_k=[0,2]):
        path='../EOS_Tables_Ozel/'
        eos_array=np.loadtxt(path+'sly.dat',skiprows=0)
        eos_array[:,0]=toMevfm(eos_array[:,0]/1.66*1e24,'baryondensity')
        eos_array[:,1]=toMevfm(eos_array[:,1],'density')
        eos_array[:,2]=toMevfm(eos_array[:,2],'density')
        eos_array[:,1:]=eos_array[:,[2,1]]
        eos_array=eos_array.transpose()
        EOS_interpolation.__init__(self,0.159,eos_array,s_k=s_k)
#nb_min=1e-7 which is not low enough for '../EOS_CompOSE/'
# =============================================================================
#         path='../EOS_CompOSE/'
#         nb=np.loadtxt(path+'SLY4'+'/eos.nb',skiprows=2)
#         mn,mp=np.loadtxt(path+'SLY4'+'/eos.thermo',max_rows=1)[:2]
#         thermo=np.loadtxt(path+'SLY4'+'/eos.thermo',skiprows=1)
#         eos_array=np.array([nb,(thermo[:,9]+1)*(nb*mn),thermo[:,3]*nb])
#         #print((thermo[:,9]+1)*(nb*mn))
#         #print(thermo[:,3]*nb)
#         EOS_interpolation.__init__(self,0.159,eos_array,s_k=s_k)
# =============================================================================
        
        
class EOS_AP4(EOS_interpolation):
    def __init__(self,s_k=[0,2]):
#energy density at saturation seems wrong(way too low) for '../EOS_Tables_Ozel/'
# =============================================================================
#         path='../EOS_Tables_Ozel/'
#         eos_array=np.loadtxt(path+'ap4.dat',skiprows=0)
#         eos_array[:,0]=toMevfm(eos_array[:,0]/1.66*1e24,'baryondensity')
#         eos_array[:,1]=toMevfm(eos_array[:,1],'density')
#         eos_array[:,2]=toMevfm(eos_array[:,2],'density')
#         eos_array[:,1:]=eos_array[:,[2,1]]
#         eos_array=eos_array.transpose()
#         EOS_interpolation.__init__(self,0.16,eos_array,s_k=s_k)
# =============================================================================
        path='../EOS_CompOSE/'
        nb=np.loadtxt(path+'APR'+'/eos.nb',skiprows=2)
        mn,mp=np.loadtxt(path+'APR'+'/eos.thermo',max_rows=1)[:2]
        thermo=np.loadtxt(path+'APR'+'/eos.thermo',skiprows=1)
        eos_array=np.array([nb,(thermo[:,9]+1)*(nb*mn),thermo[:,3]*nb])
        EOS_interpolation.__init__(self,0.16,eos_array,s_k=s_k)

class EOS_Spectral3(EOS_interpolation):
    def Gamma(self,x):
        return np.exp((np.array(self.gammas)*np.array([x**0,x,x**2])).sum(axis=0))
    def baryon_ratio(self,x): #\int_{0}^x\frac{dx'}{\Gamma(x')}
        x_0=np.array([x,0])
        integration_x_0=np.real(np.pi**0.5/(4*self.gammas[2])**0.5*np.exp(-self.gammas[0]+self.gammas[1]**2/(4*self.gammas[2]))*scipy.special.erf((self.gammas[2]*x_0+0.5*self.gammas[1])/(self.gammas[2])**0.5))
        return np.exp(integration_x_0[0]-integration_x_0[1])
    def to_integerate(self,x,gammas):
        return np.exp(x)/(self.Gamma(x)*self.baryon_ratio(x))
    def density(self,x):
        return self.baryon_ratio(x)*(self.density_0+self.pressure_0*scipy.integrate.quad(self.to_integerate,0,x,args=self.gammas)[0])
    def __init__(self,args,eos_low):
        self.args=args
        self.baryon_density_0=args[0]
        self.pressure_0=eos_low.eosPressure_frombaryon(self.baryon_density_0)
        self.density_0=eos_low.eosDensity(self.pressure_0)
        dpdn0=eos_low.eosCs2(self.pressure_0)*eos_low.eosChempo(self.pressure_0)
        gamma0=np.log(dpdn0*self.baryon_density_0/self.pressure_0)
        self.gammas=[float(gamma0),float(args[1]),float(args[2])]
        self.eos_array=np.copy(eos_low.eos_array)
        to_alter_logic=self.eos_array[0]>self.baryon_density_0
        for i in np.linspace(0,len(to_alter_logic)-1,len(to_alter_logic)).astype(int)[to_alter_logic]:
            self.eos_array[0,i]=self.baryon_density_0*self.baryon_ratio(np.log(self.eos_array[2,i]/self.pressure_0))
            self.eos_array[1,i]=self.density(np.log(self.eos_array[2,i]/self.pressure_0))
        EOS_interpolation.__init__(self,self.baryon_density_0,self.eos_array)
    def __getstate__(self):
        return EOS_interpolation.__getstate__(self)
    def __setstate__(self, state):
        EOS_interpolation.__setstate__(self, state)

from toolbox import log_array
import scipy.optimize as opt
import pickle
class EOS_Spectral3_match(EOS_interpolation):
    try:
        f=open('spectral3_match_gamma1gamma2_trial_E_28_38_L_5_200.dat','rb')
        gamma12_trial=pickle.load(f)
        f.close()
        trial_shape=gamma12_trial.shape[1:]
        e_trial=np.linspace(0.16*(939-16+28),0.16*(939-16+38),trial_shape[0])
        p_trial=np.linspace((0.16*5./3),(0.16*200./3),trial_shape[1])
        gamma1_trial = interpolate.interp2d(e_trial, p_trial, gamma12_trial[0,:,:].transpose(), kind='cubic')
        gamma2_trial = interpolate.interp2d(e_trial, p_trial, gamma12_trial[1,:,:].transpose(), kind='cubic')
    except:
        print('EOS_Spectral3_match EOS CLASS in eos_class.py not initialized.')
        print('Need to run spectral3_match_gamma1gamma2_trial.py first.')
    def equations(self,gamma12,args):
        gamma1,gamma2=gamma12
        gamma0,n0,e0,p0,n1,e1,p1=args
        x1=np.log(p1/p0)
        gammas=[float(gamma0),float(gamma1),float(gamma2)]
        baryon_ratio1=self.baryon_ratio(x1,gammas)
        density1=baryon_ratio1*(e0+p0*scipy.integrate.quad(self.to_integerate,0,x1,args=gammas)[0])
        return n0*baryon_ratio1/n1-1, density1/e1-1
    def Spectral3_match(self,args):
        gamma0,n0,e0,p0,n1,e1,p1=args
        return opt.root(self.equations,np.array([self.gamma1_trial(e1,p1),self.gamma2_trial(e1,p1)]),tol=1e-8,args=(args,))
    def Spectral3_match_after_initial_fail(self,args):  #this is to deal with the solutions when gamma2 close to zero, trial value is usual too small, so that the erf() is not working properly.
        gamma0,n0,e0,p0,n1,e1,p1=args                   #I go around the problem by  increase the trial value to 1e-4, and try multiply initial trial value to get the best sulution.
        init_list=([self.gamma1_trial(e1,p1)[0],self.gamma2_trial(e1,p1)[0]],)
        if(np.abs(self.gamma2_trial(e1,p1)[0])<1e-4):
            init_list+=([self.gamma1_trial(e1,p1)[0],1.e-4],)
        if(np.abs(self.gamma2_trial(e1,p1)[0]).max()>10):
            init_list+=([60.,-50.],)
        vary_list=np.array([1,0.5,-0.5,-1])
        shape_init=np.array(init_list).shape
        init_vary_list=np.multiply(init_list,np.tile(np.array(np.meshgrid(*([vary_list]*shape_init[1]))), (shape_init[0],)+(1,)*(shape_init[1]+1)).transpose(list(range(2,shape_init[1]+2))+list(range(2)))).reshape((len(vary_list)**shape_init[1]*shape_init[0],shape_init[1]))
        sol_x=[]
        sol_f=[]
        for init_i in init_vary_list:
            sol = opt.root(self.equations,init_i,tol=1e-8,args=(args,),method='hybr')
            sol_x.append(sol.x)
            sol_f.append(sol.fun)
            #print(init_i,sol.x,sol.fun)
            if(np.max(np.abs(sol.fun))<1e-4):
                return True,list(sol.x)
        #sol_x=np.array(sol_x)
        #sol_f_max=np.array(sol_f).max(axis=1)
        return False,[1.,-0.1]

    def Gamma(self,x,gammas):
        return np.exp((np.array(gammas)*np.array([x**0,x,x**2])).sum(axis=0))
    def baryon_ratio(self,x,gammas): #\int_{0}^x\frac{dx'}{\Gamma(x')}
        x_0=np.array([x,0])
        integration_x_0=np.real(np.pi**0.5/(4*gammas[2])**0.5*np.exp(-gammas[0]+gammas[1]**2/(4*gammas[2]))*scipy.special.erf((gammas[2]*x_0+0.5*gammas[1])/(gammas[2])**0.5))
        return np.exp(integration_x_0[0]-integration_x_0[1])
    def to_integerate(self,x,gammas):
        return np.exp(x)/(self.Gamma(x,gammas)*self.baryon_ratio(x,gammas))
    def density(self,x):
        return self.baryon_ratio(x,self.gammas)*(self.density_0+self.pressure_0*scipy.integrate.quad(self.to_integerate,0,x,args=self.gammas)[0])
    def cs2(self,x):
        p=np.exp(x)*self.pressure_0
        return p*self.Gamma(x,self.gammas)/(self.density(x)+p)
    def __init__(self,eos_array_high,eos_low,s_k=[0,1]):
        self.baryon_density_0=0.06
        self.pressure_0=eos_low.eosPressure_frombaryon(self.baryon_density_0)
        self.density_0=eos_low.eosDensity(self.pressure_0)
        dpdn0=eos_low.eosCs2(self.pressure_0)*eos_low.eosChempo(self.pressure_0)
        gamma0=np.log(dpdn0*self.baryon_density_0/self.pressure_0)
        self.baryon_density_1,self.density_1,self.pressure_1=eos_array_high[:,0]
        self.match_args=[gamma0,self.baryon_density_0,self.density_0,self.pressure_0,
                        self.baryon_density_1,self.density_1,self.pressure_1]
        #print('begin matching...')
        match_result=self.Spectral3_match(self.match_args)
        #print('finish matching...')
        self.gammas=[float(gamma0),float(match_result.x[0]),float(match_result.x[1])]
        self.success_match=np.abs(np.array(self.equations(self.gammas[1:],self.match_args))).max()<1e-4
        #print(self.equations(self.gammas[1:],self.match_args))
        if(self.success_match):
            pass
        else:
            self.success_match,gamma12=self.Spectral3_match_after_initial_fail(self.match_args)
            self.gammas=[float(gamma0),float(gamma12[0]),float(gamma12[1])]
            #print(self.equations(self.gammas[1:],self.match_args))
        eos_array_low=eos_low.eos_array[:,eos_low.eos_array[0]<self.baryon_density_0]
        if(self.success_match):
            dp_below_p0 = eos_array_low[2,-1] - eos_array_low[2,-2]
            dp_above_p1 = eos_array_high[2,1] - eos_array_high[2,0]
            p_array_match=log_array([self.pressure_0,self.pressure_1],dp_above_p1/dp_below_p0,10)[2]
            eos_array_match=[]
            cs2_array_match=[]
            for p_i in p_array_match:
                eos_array_match.append([self.baryon_density_0*self.baryon_ratio(np.log(p_i/self.pressure_0),self.gammas),
                                      self.density(np.log(p_i/self.pressure_0)),
                                      p_i])
                cs2_array_match.append(self.cs2(np.log(p_i/self.pressure_0)))
            eos_array_match=np.array(eos_array_match).transpose()
            self.causal_match=np.array(cs2_array_match).max()<=1
            self.eos_array=np.concatenate((eos_array_low,eos_array_match[:,:-1],eos_array_high),axis=1)
            #EOS_interpolation.__init__(self,self.baryon_density_0,self.eos_array)
        else:
            self.causal_match=False
            self.eos_array=np.concatenate((eos_array_low,eos_array_high),axis=1)
            #EOS_interpolation.__init__(self,self.baryon_density_0,eos_array_low)
        EOS_interpolation.__init__(self,self.baryon_density_0,self.eos_array,s_k=s_k)
    def __getstate__(self):
        return EOS_interpolation.__getstate__(self)
    def __setstate__(self, state):
        EOS_interpolation.__setstate__(self, state)


class EOS_PiecewisePoly3(object):
    def __init__(self,args):
        self.density0,self.pressure0,self.baryon_density0,self.pressure1\
        ,self.baryon_density1,self.pressure2,self.baryon_density2\
        ,self.pressure3,self.baryon_density3 = args
        self.args=args
        self.gamma1=np.log(self.pressure1/self.pressure0)\
        /np.log(self.baryon_density1/self.baryon_density0)
        self.gamma2=np.log(self.pressure2/self.pressure1)\
        /np.log(self.baryon_density2/self.baryon_density1)
        self.gamma3=np.log(self.pressure3/self.pressure2)\
        /np.log(self.baryon_density3/self.baryon_density2)
        self.density1=(self.density0-self.pressure0/(self.gamma1-1))\
        *(self.pressure1/self.pressure0)**(1/self.gamma1)\
        +self.pressure1/(self.gamma1-1)
        self.density2=(self.density1-self.pressure1/(self.gamma2-1))\
        *(self.pressure2/self.pressure1)**(1/self.gamma2)\
        +self.pressure2/(self.gamma2-1)
        self.baryon_density_s=0.16
        self.pressure_s=self.pressure0*(self.baryon_density_s/self.baryon_density0)**self.gamma1
        self.density_s=self.eosDensity(self.pressure_s)
        self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
        self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5
        self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
    def eosPressure_frombaryon(self,baryondensity):
        return np.where(baryondensity<self.baryon_density1,
                    self.baryon_density0*(baryondensity/self.baryon_density0)**self.gamma1,
                    np.where(baryondensity<self.baryon_density2,
                    self.baryon_density1*(baryondensity/self.baryon_density1)**self.gamma2,
                    self.baryon_density2*(baryondensity/self.baryon_density2)**self.gamma3))
    def eosDensity(self,pressure):
        return np.where(pressure<self.pressure1,
                    ((self.density0-self.pressure0/(self.gamma1-1))\
                   *(pressure/self.pressure0)**(1/self.gamma1)\
                   +pressure/(self.gamma1-1)),
                    np.where(pressure<self.pressure2,
                        ((self.density1-self.pressure1/(self.gamma2-1))\
                       *(pressure/self.pressure1)**(1/self.gamma2)\
                       +pressure/(self.gamma2-1)),
                         ((self.density2-self.pressure2/(self.gamma3-1))\
                       *(pressure/self.pressure2)**(1/self.gamma3)\
                       +pressure/(self.gamma3-1))))
    def eosBaryonDensity(self,pressure):
        return np.where(pressure<self.pressure1,
                    self.baryon_density0*(pressure/self.pressure0)**(1.0/self.gamma1),
                    np.where(pressure<self.pressure2,
                        self.baryon_density1*(pressure/self.pressure1)**(1.0/self.gamma2),
                        self.baryon_density2*(pressure/self.pressure2)**(1.0/self.gamma3)))
    def eosCs2(self,pressure):
        return np.where(pressure<self.pressure1,
                self.gamma1*pressure/(pressure+self.eosDensity(pressure)),
                np.where(pressure<self.pressure2,
                self.gamma2*pressure/(pressure+self.eosDensity(pressure)),
                self.gamma3*pressure/(pressure+self.eosDensity(pressure))))
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)

class EOS_PiecewisePoly4(object):
    def __init__(self,args):
        self.density0,self.pressure0,self.baryon_density0,self.pressure1\
        ,self.baryon_density1,self.pressure2,self.baryon_density2\
        ,self.pressure3,self.baryon_density3,self.pressure4,self.baryon_density4 = args
        self.args=args
        self.gamma1=np.log(self.pressure1/self.pressure0)\
        /np.log(self.baryon_density1/self.baryon_density0)
        self.gamma2=np.log(self.pressure2/self.pressure1)\
        /np.log(self.baryon_density2/self.baryon_density1)
        self.gamma3=np.log(self.pressure3/self.pressure2)\
        /np.log(self.baryon_density3/self.baryon_density2)
        self.gamma4=np.log(self.pressure4/self.pressure3)\
        /np.log(self.baryon_density4/self.baryon_density3)
        self.density1=(self.density0-self.pressure0/(self.gamma1-1))\
        *(self.pressure1/self.pressure0)**(1/self.gamma1)\
        +self.pressure1/(self.gamma1-1)
        self.density2=(self.density1-self.pressure1/(self.gamma2-1))\
        *(self.pressure2/self.pressure1)**(1/self.gamma2)\
        +self.pressure2/(self.gamma2-1)
        self.density3=(self.density2-self.pressure2/(self.gamma3-1))\
        *(self.pressure3/self.pressure2)**(1/self.gamma3)\
        +self.pressure3/(self.gamma3-1)
        self.baryon_density_s=0.16
        self.pressure_s=self.pressure0*(self.baryon_density_s/self.baryon_density0)**self.gamma1
        self.density_s=self.eosDensity(self.pressure_s)
        self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
        self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5
        self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
    def eosPressure_frombaryon(self,baryondensity):
        return np.where(baryondensity<self.baryon_density1,
                self.baryon_density0*(baryondensity/self.baryon_density0)**self.gamma1,
                np.where(baryondensity<self.baryon_density2,
                self.baryon_density1*(baryondensity/self.baryon_density1)**self.gamma2,
                np.where(baryondensity<self.baryon_density3,
                self.baryon_density2*(baryondensity/self.baryon_density2)**self.gamma3,
                self.baryon_density3*(baryondensity/self.baryon_density3)**self.gamma4)))
    def eosDensity(self,pressure):
        return np.where(pressure<self.pressure1,
                ((self.density0-self.pressure0/(self.gamma1-1))\
               *(pressure/self.pressure0)**(1/self.gamma1)\
               +pressure/(self.gamma1-1)),
                np.where(pressure<self.pressure2,
                ((self.density1-self.pressure1/(self.gamma2-1))\
               *(pressure/self.pressure1)**(1/self.gamma2)\
               +pressure/(self.gamma2-1)),
                np.where(pressure<self.pressure3,
                ((self.density2-self.pressure2/(self.gamma3-1))\
               *(pressure/self.pressure2)**(1/self.gamma3)\
               +pressure/(self.gamma3-1)),
                ((self.density3-self.pressure3/(self.gamma4-1))\
               *(pressure/self.pressure3)**(1/self.gamma4)\
               +pressure/(self.gamma4-1)))))
    def eosBaryonDensity(self,pressure):
        return np.where(pressure<self.pressure1,
                self.baryon_density0*(pressure/self.pressure0)**(1.0/self.gamma1),
                np.where(pressure<self.pressure2,
                self.baryon_density1*(pressure/self.pressure1)**(1.0/self.gamma2),
                np.where(pressure<self.pressure3,
                self.baryon_density2*(pressure/self.pressure2)**(1.0/self.gamma3),
                self.baryon_density3*(pressure/self.pressure3)**(1.0/self.gamma4))))
    def eosCs2(self,pressure):
        return np.where(pressure<self.pressure1,
                self.gamma1*pressure/(pressure+self.eosDensity(pressure)),
                np.where(pressure<self.pressure2,
                self.gamma2*pressure/(pressure+self.eosDensity(pressure)),
                np.where(pressure<self.pressure3,
                self.gamma3*pressure/(pressure+self.eosDensity(pressure)),
                self.gamma4*pressure/(pressure+self.eosDensity(pressure)))))
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)

class EOS_PiecewisePoly3WithCrust(EOS_PiecewisePoly3):
    eos_low=EOS_SLY4()
    chempo_surface=eos_low.chempo_surface
    def __init__(self,args):
        n0,p1,p2,p3=args
        p0=self.eos_low.eosPressure_frombaryon(n0)
        e0=self.eos_low.eosDensity(p0)
        n1,n2,n3=0.16*np.array([1.85,3.7,7.4])
        self.success_eos=p1>5*p0 and p2>95. and p3>1.2*p2
        if(self.success_eos):
            EOS_PiecewisePoly3.__init__(self,[e0,p0,n0,p1,n1,p2,n2,p3,n3])
            self.success_eos=self.success_eos and self.eosCs2((1-dlnx_cs2)*p1)<1.0001 and self.eosCs2((1-dlnx_cs2)*p2)<1.0001
    def eosDensity(self,pressure):
        return np.where(pressure>self.pressure0,EOS_PiecewisePoly3.eosDensity(self,pressure),\
                        self.eos_low.eosDensity(pressure))
    def eosBaryonDensity(self,pressure):
        return np.where(pressure>self.pressure0,EOS_PiecewisePoly3.eosBaryonDensity(self,pressure),\
                        self.eos_low.eosBaryonDensity(pressure))
    def eosCs2(self,pressure): #it is a step function at BPS region, since I use Linear intepolation
        return np.where(pressure>self.pressure0,EOS_PiecewisePoly3.eosCs2(self,pressure),\
                        self.eos_low.eosCs2(pressure))
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
    def setMaxmass(self,result_maxmaxmass):
        self.pc_max,self.mass_max,self.cs2_max=result_maxmaxmass
        self.maxmass_success=self.cs2_max<1 and self.mass_max>2
        self.eos_success_all=self.maxmass_success and self.success_eos
        return self.eos_success_all
    def setProperity(self,Properity_one,Properity_onepointfour):
        self.properity_one,self.properity_onepointfour=Properity_one,Properity_onepointfour
        
class EOS_PiecewisePoly4WithCrust(EOS_PiecewisePoly4):
    eos_low=EOS_SLY4()
    chempo_surface=eos_low.chempo_surface
    def __init__(self,args):
        n0,p1,p2,p3,p4=args
        p0=self.eos_low.eosPressure_frombaryon(n0)
        e0=self.eos_low.eosDensity(p0)
        n1,n2,n3,n4=0.16*np.array([1.,1.85,3.7,7.4])
        EOS_PiecewisePoly4.__init__(self,[e0,p0,n0,p1,n1,p2,n2,p3,n3,p4,n4])
        #self.success_eos=
    def eosDensity(self,pressure):
        return np.where(pressure>self.pressure0,EOS_PiecewisePoly4.eosDensity(self.pressure),\
                        self.eos_low.eosDensity(pressure))
    def eosBaryonDensity(self,pressure):
        return np.where(pressure>self.pressure0,EOS_PiecewisePoly4.eosBaryonDensity(self.pressure),\
                        self.eos_low.eosBaryonDensity(pressure))
    def eosCs2(self,pressure): #it is a step function at BPS region, since I use Linear intepolation
        return 1.0/derivative(self.eosDensity,pressure,dx=pressure*dlnx_cs2)
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
    #def setMaxmass(self,result_maxmaxmass):
    #def setProperity(self,Properity_one,Properity_onepointfour):

class EOS_PiecewiseCSS2(object):
    def __init__(self,args):
        self.density0,self.pressure0,self.baryondensity0,self.cs2_1,self.baryondensity1,self.cs2_2 = args
        self.B0=(self.density0-self.pressure0/self.cs2_1)/(1.0+1.0/self.cs2_1)
        self.pressure1=(self.baryondensity1/self.baryondensity0)**(1.0+self.cs2_1)*(self.pressure0+self.B0)-self.B0
        self.density1=(self.pressure1-self.pressure0)/self.cs2_1+self.density0
        self.B1=(self.density1-self.pressure1/self.cs2_2)/(1.0+1.0/self.cs2_2)
        self.baryon_density_s=0.16
        self.pressure_s=(self.baryon_density_s/self.baryondensity0)**(1.0+self.cs2_1)*(self.pressure0+self.B0)-self.B0
        self.density_s=self.eosDensity(self.pressure_s)
        self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
        self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5
        self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
    def eosPressure_frombaryon(self,baryondensity):
        pressure_1=(baryondensity/self.baryondensity0)**(1.0+self.cs2_1)*(self.pressure0+self.B0)-self.B0
        pressure_2=(baryondensity/self.baryondensity1)**(1.0+self.cs2_2)*(self.pressure1+self.B1)-self.B1
        pressure=np.where(baryondensity>self.baryondensity1,pressure_2,pressure_1)
        return pressure
    def eosDensity(self,pressure):
        density_1 = (pressure-self.pressure0)/self.cs2_1+self.density0
        density_2 = (pressure-self.pressure1)/self.cs2_2+self.density1
        density=np.where(pressure>self.pressure1,density_2,density_1)
        return np.where(density>0,density,0)
    def eosBaryonDensity(self,pressure):
        baryondensity_1 = self.baryondensity0*np.abs((pressure+self.B0)/(self.pressure0+self.B0))**(1.0/(1.0+self.cs2_1))
        baryondensity_2 = self.baryondensity1*np.abs((pressure+self.B1)/(self.pressure1+self.B1))**(1.0/(1.0+self.cs2_2))
        baryondensity=np.where(pressure>self.pressure1,baryondensity_2,baryondensity_1)
        return np.where(baryondensity>0,baryondensity,0)
    def eosCs2(self,pressure):
        return np.where(pressure>self.pressure1,self.cs2_2,self.cs2_1)
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)

class EOS_PiecewiseCSS2WithCrust(EOS_PiecewiseCSS2):
    eos_low=EOS_SLY4()
    chempo_surface=eos_low.chempo_surface
    def __init__(self,args):
        n0,cs2_1,cs2_2=args
        p0=self.eos_low.eosPressure_frombaryon(n0)
        e0=self.eos_low.eosDensity(p0)
        n1=0.16*1.85
        EOS_PiecewiseCSS2.__init__(self,[e0,p0,n0,cs2_1,n1,cs2_2])
        self.success_eos=0.01<cs2_1<1 and 0.1<cs2_2<1 and self.pressure0+self.B0>0 and self.pressure1+self.B1>0
    def eosPressure_frombaryon(self,baryondensity):
        return np.where(baryondensity>self.baryondensity0,EOS_PiecewiseCSS2.eosPressure_frombaryon(self,baryondensity),\
                        self.eos_low.eosPressure_frombaryon(baryondensity))
    def eosDensity(self,pressure):
        return np.where(pressure>self.pressure0,EOS_PiecewiseCSS2.eosDensity(self,pressure),\
                        self.eos_low.eosDensity(pressure))
    def eosBaryonDensity(self,pressure):
        return np.where(pressure>self.pressure0,EOS_PiecewiseCSS2.eosBaryonDensity(self,pressure),\
                        self.eos_low.eosBaryonDensity(pressure))
    def eosCs2(self,pressure): #it is a step function at BPS region, since I use Linear intepolation
        return np.where(pressure>self.pressure0,EOS_PiecewiseCSS2.eosCs2(self,pressure),\
                        self.eos_low.eosCs2(pressure))
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
    def setMaxmass(self,result_maxmaxmass):
        self.pc_max,self.mass_max,self.cs2_max=result_maxmaxmass
        self.maxmass_success=self.mass_max>2
        self.eos_success_all=self.maxmass_success and self.success_eos
        return self.eos_success_all
    def setProperity(self,Properity_one,Properity_onepointfour):
        self.properity_one,self.properity_onepointfour=Properity_one,Properity_onepointfour
    
class EOS_PiecewiseCSS3(object):
    def __init__(self,args):
        self.density0,self.pressure0,self.baryondensity0,self.cs2_1,self.baryondensity1,self.cs2_2,self.baryondensity2,self.cs2_3 = args
        self.B0=(self.density0-self.pressure0/self.cs2_1)/(1.0+1.0/self.cs2_1)
        self.pressure1=(self.baryondensity1/self.baryondensity0)**(1.0+self.cs2_1)*(self.pressure0+self.B0)-self.B0
        self.density1=(self.pressure1-self.pressure0)/self.cs2_1+self.density0
        self.B1=(self.density1-self.pressure1/self.cs2_2)/(1.0+1.0/self.cs2_2)
        self.pressure2=(self.baryondensity2/self.baryondensity1)**(1.0+self.cs2_2)*(self.pressure1+self.B1)-self.B1
        self.density2=(self.pressure2-self.pressure1)/self.cs2_2+self.density1
        self.B2=(self.density2-self.pressure2/self.cs2_3)/(1.0+1.0/self.cs2_3)
        self.baryon_density_s=0.16
        self.pressure_s=(self.baryon_density_s/self.baryondensity0)**(1.0+self.cs2_1)*(self.pressure0+self.B0)-self.B0
        self.density_s=self.eosDensity(self.pressure_s)
        self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
        self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5
        self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
    def eosPressure_frombaryon(self,baryondensity):
        pressure_1=(baryondensity/self.baryondensity0)**(1.0+self.cs2_1)*(self.pressure0+self.B0)-self.B0
        pressure_2=(baryondensity/self.baryondensity1)**(1.0+self.cs2_2)*(self.pressure1+self.B1)-self.B1
        pressure_3=(baryondensity/self.baryondensity2)**(1.0+self.cs2_3)*(self.pressure2+self.B2)-self.B2
        pressure=np.where(baryondensity>self.baryondensity2,pressure_3,np.where(baryondensity>self.baryondensity1,pressure_2,pressure_1))
        return pressure
    def eosDensity(self,pressure):
        density_1 = (pressure-self.pressure0)/self.cs2_1+self.density0
        density_2 = (pressure-self.pressure1)/self.cs2_2+self.density1
        density_3 = (pressure-self.pressure2)/self.cs2_3+self.density2
        density=np.where(pressure>self.pressure2,density_3,np.where(pressure>self.pressure1,density_2,density_1))
        return density
    def eosBaryonDensity(self,pressure):
        baryondensity_1 = self.baryondensity0*np.abs((pressure+self.B0)/(self.pressure0+self.B0))**(1.0/(1.0+self.cs2_1))
        baryondensity_2 = self.baryondensity1*np.abs((pressure+self.B1)/(self.pressure1+self.B1))**(1.0/(1.0+self.cs2_2))
        baryondensity_3 = self.baryondensity2*np.abs((pressure+self.B2)/(self.pressure2+self.B2))**(1.0/(1.0+self.cs2_3))
        baryondensity=np.where(pressure>self.pressure2,baryondensity_3,np.where(pressure>self.pressure1,baryondensity_2,baryondensity_1))
        return baryondensity
    def eosCs2(self,pressure):
        return np.where(pressure>self.pressure2,self.cs2_3,np.where(pressure>self.pressure1,self.cs2_2,self.cs2_1))
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)

class EOS_PiecewiseCSS3WithCrust(EOS_PiecewiseCSS3):
    eos_low=EOS_SLY4()
    chempo_surface=eos_low.chempo_surface
    def __init__(self,args):
        n0,cs2_1,cs2_2,cs2_3=args
        p0=self.eos_low.eosPressure_frombaryon(n0)
        e0=self.eos_low.eosDensity(p0)
        n1,n2=0.16*np.array([1.85,3.7])
        EOS_PiecewiseCSS3.__init__(self,[e0,p0,n0,cs2_1,n1,cs2_2,n2,cs2_3])
        self.success_eos=0.005<cs2_1<1.001 and 0.05<cs2_2<1.001 and 0.05<cs2_3<1.001 and self.pressure0+self.B0>0 and self.pressure1+self.B1>0 and self.pressure2+self.B2>0
    def eosPressure_frombaryon(self,baryondensity):
        return np.where(baryondensity>self.baryondensity0,EOS_PiecewiseCSS3.eosPressure_frombaryon(self,baryondensity),\
                        self.eos_low.eosPressure_frombaryon(baryondensity))
    def eosDensity(self,pressure):
        return np.where(pressure>self.pressure0,EOS_PiecewiseCSS3.eosDensity(self,pressure),\
                        self.eos_low.eosDensity(pressure))
    def eosBaryonDensity(self,pressure):
        return np.where(pressure>self.pressure0,EOS_PiecewiseCSS3.eosBaryonDensity(self,pressure),\
                        self.eos_low.eosBaryonDensity(pressure))
    def eosCs2(self,pressure): #it is a step function at BPS region, since I use Linear intepolation
        return np.where(pressure>self.pressure0,EOS_PiecewiseCSS3.eosCs2(self,pressure),\
                        self.eos_low.eosCs2(pressure))
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
    def setMaxmass(self,result_maxmaxmass):
        self.pc_max,self.mass_max,self.cs2_max=result_maxmaxmass
        self.maxmass_success=self.mass_max>2
        self.eos_success_all=self.maxmass_success and self.success_eos
        return self.eos_success_all
    def setProperity(self,Properity_one,Properity_onepointfour):
        self.properity_one,self.properity_onepointfour=Properity_one,Properity_onepointfour

# =============================================================================
#         if(self.success_eos):
#             self.causality_p300=self.cs2(300.)<1
#         else:
#             self.causality_p300=False
# =============================================================================
        
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





# =============================================================================
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
    
# =============================================================================
#     
# class EOS_BPS(object):
# # =============================================================================
# #     #density in units: g/cm3
# #     densityBPS = np.array([-7.861E0,0.0,7.861E0,  7.900E0,  8.150E0,  1.160E1,  1.640E1,  4.510E1,  2.120E2,  1.150E3,  1.044E4,  2.622E4,  6.587E4,  1.654E5,  4.156E5,  1.044E6,  2.622E6,  6.588E6,  8.294E6,  1.655E7,  3.302E7,  6.590E7,  1.315E8,  2.624E8,  3.304E8,  5.237E8,  8.301E8,  1.045E9,  1.316E9,  1.657E9,  2.626E9,  4.164E9,  6.602E9,  8.313E9,  1.046E10,  1.318E10,  1.659E10,  2.090E10,  2.631E10,  3.313E10,  4.172E10,  5.254E10,  6.617E10,  8.333E10,  1.049E11,  1.322E11,  1.664E11,  1.844E11,  2.096E11,  2.640E11,  3.325E11,  4.188E11,  4.299E11,  4.460E11,  5.228E11,  6.610E11,  7.964E11,  9.728E11,  1.196E12,  1.471E12,  1.805E12,  2.202E12,  2.930E12,  3.833E12,  4.933E12,  6.248E12,  7.801E12,  9.612E12,  1.246E13,  1.496E13, 1.778E13, 2.210E13, 2.988E13, 3.767E13, 5.081E13, 6.193E13, 7.732E13, 9.826E13, 1.262E14])
# #     #pressure in units: g*cm/s
# #     pressureBPS = np.array([-1.010E9,0.0, 1.010E9,  1.010E10,  1.010E11,  1.210E12,  1.400E13,  1.700E14,  5.820E15,  1.900E17,  9.744E18,  4.968E19,  2.431E20,  1.151E21,  5.266E21,  2.318E22,  9.755E22,  3.911E23,  5.259E23,  1.435E24,  3.833E24,  1.006E25,  2.604E25,  6.676E25,  8.738E25,  1.629E26,  3.029E26,  4.129E26,  5.036E26,  6.860E26,  1.272E27,  2.356E27,  4.362E27,  5.662E27,  7.702E27,  1.048E28,  1.425E28,  1.938E28,  2.503E28,  3.404E28,  4.628E28,  5.949E28,  8.089E28,  1.100E29,  1.495E29,  2.033E29,  2.597E29,  2.892E29,  3.290E29,  4.473E29,  5.816E29,  7.538E29,  7.805E29,  7.890E29,  8.352E29,  9.098E29,  9.831E29,  1.083E30,  1.218E30,  1.399E30,  1.638E30,  1.950E30,  2.592E30,  3.506E30,  4.771E30,  6.481E30,  8.748E30,  1.170E31, 1.695E31, 2.209E31, 2.848E31, 3.931E31, 6.178E31, 8.774E31, 1.386E32, 1.882E32, 2.662E32, 3.897E32, 5.861E32])
# #     #baryon density in units: 1/cm3
# #     baryondensityBPS = np.array([-4.73E24,0.0, 4.73E24, 4.76E24, 4.91E24, 6.990E24,  9.900E24,  2.720E25,  1.270E26,  6.930E26,  6.295E27,  1.581E28,  3.972E28,  9.976E28,  2.506E29,  6.294E29,  1.581E30,  3.972E30,  5.000E30,  9.976E30,  1.990E31,  3.972E31,  7.924E31,  1.581E32,  1.990E32,  3.155E32,  5.000E32,  6.294E32,  7.924E32,  9.976E32,  1.581E33,  2.506E33,  3.972E33,  5.000E33,  6.294E33,  7.924E33,  9.976E33,  1.256E34,  1.581E34,  1.990E34,  2.506E34,  3.155E34,  3.972E34,  5.000E34,  6.294E34,  7.924E34,  9.976E34,  1.105E35,  1.256E35,  1.581E35,  1.990E35,  2.506E35,  2.572E35,  2.670E35,  3.126E35,  3.951E35,  4.759E35,  5.812E35,  7.143E35,  8.786E35,  1.077E36,  1.314E36,  1.748E36,  2.287E36,  2.942E36,  3.726E36,  4.650E36,  5.728E36, 7.424E36, 8.907E36, 1.059E37, 1.315E37, 1.777E37, 2.239E37, 3.017E37, 3.675E37, 4.585E37, 5.821E37, 7.468E37])
# #     #density in units: Mevfm3
# #     densityBPS = toMevfm(densityBPS,'density')
# #     #pressure in units: Mevfm3
# #     pressureBPS = toMevfm(pressureBPS,'pressure')
# #     #baryon density in units: 1/fm3
# #     baryondensityBPS = toMevfm(baryondensityBPS,'baryondensity')
# # =============================================================================
#     
#     eos_sly4=np.loadtxt('./eos_sly4.dat')
#     baryondensityBPS=eos_sly4[:,0]
#     densityBPS=eos_sly4[:,1]
#     pressureBPS=eos_sly4[:,2]
# 
#     #density in units: Mevfm3
#     densityBPS = toMevfm(densityBPS,'density')
#     #pressure in units: Mevfm3
#     pressureBPS = toMevfm(pressureBPS,'pressure')
# 
#     eosPressure_frombaryon = interp1d(list(baryondensityBPS[:66])+[0.16/2.7]+list(baryondensityBPS[66:])+[baryondensityBPS[-1]*100],list(pressureBPS[:66])+[0.22201]+list(pressureBPS[66:])+[pressureBPS[-1]*100], kind='quadratic')
#     eosPressure = interp1d(list(densityBPS[:66])+[56.2006]+list(densityBPS[66:])+[densityBPS[-1]*100],list(pressureBPS[:66])+[0.22201]+list(pressureBPS[66:])+[pressureBPS[-1]*100], kind='quadratic')
#     eosDensity  = interp1d(list(pressureBPS[:66])+[0.22201]+list(pressureBPS[66:])+[pressureBPS[-1]*100],list(densityBPS[:66])+[56.2006]+list(densityBPS[66:])+[densityBPS[-1]*100], kind='quadratic')
#     eosBaryonDensity = interp1d(list(pressureBPS[:66])+[0.22201]+list(pressureBPS[66:])+[pressureBPS[-1]*100],list(baryondensityBPS[:66])+[0.16/2.7]+list(baryondensityBPS[66:])+[baryondensityBPS[-1]*100], kind='quadratic')
#     baryon_density_s=0.16
#     chempo_surface=(pressureBPS[0]+densityBPS[0])/baryondensityBPS[0]
#     pressure_s=eosPressure_frombaryon(baryon_density_s)
#     density_s=eosDensity(pressure_s)
#     unit_mass=c**4/(G**3*density_s*1e51*e)**0.5
#     unit_radius=c**2/(G*density_s*1e51*e)**0.5
#     unit_N=unit_radius**3*baryon_density_s*1e45
#     def eosChempo(self,pressure):
#         return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
#     def eosCs2(self,pressure):
#         return 1.0/derivative(self.eosDensity,pressure,dx=pressure*dlnx_cs2)
#         
# 
# class EOS_PiecewisePoly(object):
#     def __init__(self,args):
#         self.density0,self.pressure0,self.baryon_density0,self.pressure1\
#         ,self.baryon_density1,self.pressure2,self.baryon_density2\
#         ,self.pressure3,self.baryon_density3 = args
#         self.args=args
#         self.gamma1=np.log(self.pressure1/self.pressure0)\
#         /np.log(self.baryon_density1/self.baryon_density0)
#         self.gamma2=np.log(self.pressure2/self.pressure1)\
#         /np.log(self.baryon_density2/self.baryon_density1)
#         self.gamma3=np.log(self.pressure3/self.pressure2)\
#         /np.log(self.baryon_density3/self.baryon_density2)
#         self.density1=(self.density0-self.pressure0/(self.gamma1-1))\
#         *(self.pressure1/self.pressure0)**(1/self.gamma1)\
#         +self.pressure1/(self.gamma1-1)
#         self.density2=(self.density1-self.pressure1/(self.gamma2-1))\
#         *(self.pressure2/self.pressure1)**(1/self.gamma2)\
#         +self.pressure2/(self.gamma2-1)
#         self.baryon_density_s=0.16
#         self.pressure_s=self.pressure0*(self.baryon_density_s/self.baryon_density0)**self.gamma1
#         self.density_s=self.eosDensity(self.pressure_s)
#         self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
#         self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5
#         self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
#     def eosDensity(self,pressure):
#         return np.where(pressure<self.pressure1,
#                     ((self.density0-self.pressure0/(self.gamma1-1))\
#                    *(pressure/self.pressure0)**(1/self.gamma1)\
#                    +pressure/(self.gamma1-1)),
#                     np.where(pressure<self.pressure2,
#                         ((self.density1-self.pressure1/(self.gamma2-1))\
#                        *(pressure/self.pressure1)**(1/self.gamma2)\
#                        +pressure/(self.gamma2-1)),
#                          ((self.density2-self.pressure2/(self.gamma3-1))\
#                        *(pressure/self.pressure2)**(1/self.gamma3)\
#                        +pressure/(self.gamma3-1))))
#                         
#     def eosBaryonDensity(self,pressure):
#         return np.where(pressure<self.pressure1,
#                     self.baryon_density0*(pressure/self.pressure0)**(1.0/self.gamma1),
#                     np.where(pressure<self.pressure2,
#                         self.baryon_density1*(pressure/self.pressure1)**(1.0/self.gamma2),
#                         self.baryon_density2*(pressure/self.pressure2)**(1.0/self.gamma3)))
#     def eosCs2(self,pressure):
#         return 1.0/derivative(self.eosDensity,pressure,dx=pressure*dlnx_cs2)
#     def eosChempo(self,pressure):
#         return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
# 
# class EOS_PiecewisePoly_4(object):
#     def __init__(self,args):
#         self.density0,self.pressure0,self.baryon_density0,self.pressure1\
#         ,self.baryon_density1,self.pressure2,self.baryon_density2\
#         ,self.pressure3,self.baryon_density3,self.pressure4,self.baryon_density4 = args
#         self.args=args
#         self.gamma1=np.log(self.pressure1/self.pressure0)\
#         /np.log(self.baryon_density1/self.baryon_density0)
#         self.gamma2=np.log(self.pressure2/self.pressure1)\
#         /np.log(self.baryon_density2/self.baryon_density1)
#         self.gamma3=np.log(self.pressure3/self.pressure2)\
#         /np.log(self.baryon_density3/self.baryon_density2)
#         self.gamma4=np.log(self.pressure4/self.pressure3)\
#         /np.log(self.baryon_density4/self.baryon_density3)
#         self.density1=(self.density0-self.pressure0/(self.gamma1-1))\
#         *(self.pressure1/self.pressure0)**(1/self.gamma1)\
#         +self.pressure1/(self.gamma1-1)
#         self.density2=(self.density1-self.pressure1/(self.gamma2-1))\
#         *(self.pressure2/self.pressure1)**(1/self.gamma2)\
#         +self.pressure2/(self.gamma2-1)
#         self.density3=(self.density2-self.pressure2/(self.gamma3-1))\
#         *(self.pressure3/self.pressure2)**(1/self.gamma3)\
#         +self.pressure3/(self.gamma3-1)
#         self.baryon_density_s=0.16
#         self.pressure_s=self.pressure0*(self.baryon_density_s/self.baryon_density0)**self.gamma1
#         self.density_s=self.eosDensity(self.pressure_s)
#         self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
#         self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5
#         self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
#     def eosDensity(self,pressure):
#         return np.where(pressure<self.pressure1,
#                     ((self.density0-self.pressure0/(self.gamma1-1))\
#                    *(pressure/self.pressure0)**(1/self.gamma1)\
#                    +pressure/(self.gamma1-1)),
#                     np.where(pressure<self.pressure2,
#                         ((self.density1-self.pressure1/(self.gamma2-1))\
#                        *(pressure/self.pressure1)**(1/self.gamma2)\
#                        +pressure/(self.gamma2-1)),
#                         np.where(pressure<self.pressure3,
#                             ((self.density2-self.pressure2/(self.gamma3-1))\
#                            *(pressure/self.pressure2)**(1/self.gamma3)\
#                            +pressure/(self.gamma3-1)),
#                              
#                              ((self.density3-self.pressure3/(self.gamma4-1))\
#                            *(pressure/self.pressure3)**(1/self.gamma4)\
#                            +pressure/(self.gamma4-1)))))
#                         
#     def eosBaryonDensity(self,pressure):
#         return np.where(pressure<self.pressure1,
#                     self.baryon_density0*(pressure/self.pressure0)**(1.0/self.gamma1),
#                     np.where(pressure<self.pressure2,
#                         self.baryon_density1*(pressure/self.pressure1)**(1.0/self.gamma2),
#                         np.where(pressure<self.pressure3,
#                             self.baryon_density2*(pressure/self.pressure2)**(1.0/self.gamma3),
#                             self.baryon_density3*(pressure/self.pressure3)**(1.0/self.gamma4))))
#     def eosCs2(self,pressure):
#         return 1.0/derivative(self.eosDensity,pressure,dx=pressure*dlnx_cs2)
#     def eosChempo(self,pressure):
#         return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
# 
# class EOS_BPSwithPoly(EOS_BPS):
#     def __init__(self,args):
#         self.baryon_density0,self.pressure1,self.baryon_density1\
#         ,self.pressure2,self.baryon_density2,self.pressure3\
#         ,self.baryon_density3=args
#         self.args=args
#         self.pressure0=EOS_BPS.eosPressure_frombaryon(self.baryon_density0)
#         self.density0=EOS_BPS.eosDensity(self.pressure0)
#         args_eosPiecewisePoly=[self.density0,self.pressure0\
#                                            ,self.baryon_density0,self.pressure1\
#                                            ,self.baryon_density1,self.pressure2\
#                                            ,self.baryon_density2,self.pressure3\
#                                            ,self.baryon_density3]
#         self.eosPiecewisePoly=EOS_PiecewisePoly(args_eosPiecewisePoly)
#         self.baryon_density_s=self.eosPiecewisePoly.baryon_density_s
#         self.pressure_s=self.eosPiecewisePoly.pressure_s
#         self.density_s=self.eosPiecewisePoly.density_s
#         self.unit_mass=self.eosPiecewisePoly.unit_mass
#         self.unit_radius=self.eosPiecewisePoly.unit_radius
#         self.unit_N=self.eosPiecewisePoly.unit_N
#     def eosDensity(self,pressure):
#         return np.where(pressure>self.pressure0,self.eosPiecewisePoly.eosDensity(pressure),\
#                         EOS_BPS.eosDensity(pressure))
#     def eosBaryonDensity(self,pressure):
#         return np.where(pressure>self.pressure0,self.eosPiecewisePoly.eosBaryonDensity(pressure),\
#                         EOS_BPS.eosBaryonDensity(pressure))
#     def eosCs2(self,pressure): #it is a step function at BPS region, since I use Linear intepolation
#         return 1.0/derivative(self.eosDensity,pressure,dx=pressure*dlnx_cs2)
#     def eosChempo(self,pressure):
#         return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
# 
# class EOS_BPSwithPoly_4(EOS_BPS):
#     def __init__(self,args):
#         self.baryon_density0,self.pressure1,self.baryon_density1\
#         ,self.pressure2,self.baryon_density2,self.pressure3\
#         ,self.baryon_density3,self.pressure4,self.baryon_density4=args
#         self.args=args
#         self.pressure0=EOS_BPS.eosPressure_frombaryon(self.baryon_density0)
#         self.density0=EOS_BPS.eosDensity(self.pressure0)
#         args_eosPiecewisePoly=[self.density0,self.pressure0\
#                                            ,self.baryon_density0,self.pressure1\
#                                            ,self.baryon_density1,self.pressure2\
#                                            ,self.baryon_density2,self.pressure3\
#                                            ,self.baryon_density3,self.pressure4\
#                                            ,self.baryon_density4]
#         self.eosPiecewisePoly=EOS_PiecewisePoly_4(args_eosPiecewisePoly)
#         self.baryon_density_s=self.eosPiecewisePoly.baryon_density_s
#         self.pressure_s=self.eosPiecewisePoly.pressure_s
#         self.density_s=self.eosPiecewisePoly.density_s
#         self.unit_mass=self.eosPiecewisePoly.unit_mass
#         self.unit_radius=self.eosPiecewisePoly.unit_radius
#         self.unit_N=self.eosPiecewisePoly.unit_N
#     def eosDensity(self,pressure):
#         return np.where(pressure>self.pressure0,self.eosPiecewisePoly.eosDensity(pressure),\
#                         EOS_BPS.eosDensity(pressure))
#     def eosBaryonDensity(self,pressure):
#         return np.where(pressure>self.pressure0,self.eosPiecewisePoly.eosBaryonDensity(pressure),\
#                         EOS_BPS.eosBaryonDensity(pressure))
#     def eosCs2(self,pressure): #it is a step function at BPS region, since I use Linear intepolation
#         return 1.0/derivative(self.eosDensity,pressure,dx=pressure*dlnx_cs2)
#     def eosChempo(self,pressure):
#         return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
# 
# class EOS_CSS(object):
#     def __init__(self,args):
#         self.density0,self.pressure0,self.baryondensity0,self.cs2 = args
#         self.B=(self.density0-self.pressure0/self.cs2)/(1.0+1.0/self.cs2)
#         if(self.B>0):
#             self.baryon_density_s=self.baryondensity0/(self.pressure0/self.B+1)**(1/(self.cs2+1))
#             self.pressure_s=self.B
#             self.density_s=self.B
#         else:
#             self.baryon_density_s=0.16
#             self.pressure_s=-self.B
#             self.density_s=-self.B
#             print('Warning!!! ESS equation get negative Bag constant!!!')
#             print('args=%s'%(args))
#             print('B=%f MeVfm-3'%(self.B))
#         self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
#         self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5
#         self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
#     def eosDensity(self,pressure):
#         density = (pressure-self.pressure0)/self.cs2+self.density0
#         return np.where(density>0,density,0)
#     def eosBaryonDensity(self,pressure):
#         baryondensity = self.baryondensity0*((pressure+self.B)/(self.pressure0+self.B))**(1.0/(1.0+self.cs2))
#         return np.where(baryondensity>0,baryondensity,0)
#     def eosCs2(self,pressure):
#         return self.cs2
#     def eosChempo(self,pressure):
#         return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
# 
# class EOS_CSS_2piece(object):
#     def __init__(self,args):
#         self.density0,self.pressure0,self.baryondensity0,self.cs2_1, self.pressure1,self.cs2_2 = args
#         self.density1=(self.pressure1-self.pressure0)/self.cs2_1+self.density0
#         self.B0=(self.density0-self.pressure0/self.cs2_1)/(1.0+1.0/self.cs2_1)
#         self.B1=(self.density1-self.pressure1/self.cs2_2)/(1.0+1.0/self.cs2_2)
#         self.baryondensity1=self.baryondensity0*((self.pressure1+self.B0)/(self.pressure0+self.B0))**(1.0/(1.0+self.cs2_1))
#         if(self.B0>0):
#             self.baryon_density_s=self.baryondensity0/(self.pressure0/self.B0+1)**(1/(self.cs2_1+1))
#             self.pressure_s=self.B0
#             self.density_s=self.B0
#         else:
#             self.baryon_density_s=0.16
#             self.pressure_s=-self.B0
#             self.density_s=-self.B0
#             print('Warning!!! ESS equation get negative Bag constant!!!')
#             print('args=%s'%(args))
#             print('B=%f MeVfm-3'%(self.B0))
#         self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
#         self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5
#         self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
#     def eosDensity(self,pressure):
#         density_1 = (pressure-self.pressure0)/self.cs2_1+self.density0
#         density_2 = (pressure-self.pressure1)/self.cs2_2+self.density1
#         density=np.where(pressure>self.pressure1,density_2,density_1)
#         return np.where(density>0,density,0)
#     def eosBaryonDensity(self,pressure):
#         baryondensity_1 = self.baryondensity0*((pressure+self.B0)/(self.pressure0+self.B0))**(1.0/(1.0+self.cs2_1))
#         baryondensity_2 = self.baryondensity1*((pressure+self.B1)/(self.pressure1+self.B1))**(1.0/(1.0+self.cs2_2))
#         baryondensity=np.where(pressure>self.pressure1,baryondensity_2,baryondensity_1)
#         return np.where(baryondensity>0,baryondensity,0)
#     def eosCs2(self,pressure):
#         return np.where(pressure>self.pressure1,self.cs2_2,self.cs2_1)
#     def eosChempo(self,pressure):
#         return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
# 
# class EOS_BPSwithPolyCSS(EOS_BPSwithPoly,EOS_CSS):
#     def __init__(self,args):
#         self.baryon_density0,self.pressure1,self.baryon_density1\
#         ,self.pressure2,self.baryon_density2,self.pressure3\
#         ,self.baryon_density3,self.pressure_trans,self.det_density\
#         ,self.cs2=args
#         self.args=args
#         self.eosBPSwithPoly=EOS_BPSwithPoly(args[0:7])
#         self.baryon_density_s=self.eosBPSwithPoly.baryon_density_s
#         self.pressure_s=self.eosBPSwithPoly.pressure_s
#         self.density_s=self.eosBPSwithPoly.density_s
#         self.unit_mass=self.eosBPSwithPoly.unit_mass
#         self.unit_radius=self.eosBPSwithPoly.unit_radius
#         self.unit_N=self.eosBPSwithPoly.unit_N
#         self.density_trans=self.eosBPSwithPoly.eosDensity(self.pressure_trans)
#         self.baryondensity0=self.eosBPSwithPoly.eosBaryonDensity\
#         (self.pressure_trans)/(self.density_trans+self.pressure_trans)\
#         *(self.density_trans+self.pressure_trans+self.det_density)
#         args_eosCSS=[self.density_trans+self.det_density,self.pressure_trans\
#                      ,self.baryondensity0,self.cs2]
#         self.eosCSS=EOS_CSS(args_eosCSS)
#     def eosDensity(self,pressure):
#         return np.where(pressure<self.pressure_trans,self.eosBPSwithPoly.eosDensity(pressure),self.eosCSS.eosDensity(pressure))
#     def eosBaryonDensity(self,pressure):
#         return np.where(pressure<self.pressure_trans,self.eosBPSwithPoly.eosBaryonDensity(pressure),self.eosCSS.eosBaryonDensity(pressure))
#     def eosCs2(self,pressure):
#         return np.where(pressure<self.pressure_trans,self.eosBPSwithPoly.eosCs2(pressure),self.cs2)
#     def eosChempo(self,pressure):
#         return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
# 
# 
# class EOS_MIT(EOS_BPS): #reference: HYBRID STARS THAT MASQUERADE AS NEUTRON STARS
#     def __init__(self,args):
#         self.ms,self.bag,self.delta,self.a4=args #bag in unit MeVfm-3
#         self.baryon_density_s=0.16
#         self.pressure_s=self.bag
#         self.density_s=self.bag
#         self.pressureMIT,self.densityMIT,self.baryondensityMIT\
#         =self.eos_for_intepolation(np.linspace(1,2000,1000))
#         self.eosPressure = interp1d(self.densityMIT,self.pressureMIT, kind='linear')
#         self.eosDensity  = interp1d(self.pressureMIT,self.densityMIT, kind='linear')
#         self.eosBaryonDensity = interp1d(self.pressureMIT,self.baryondensityMIT, kind='linear')
# 
#     def Omega(self,chempo):
#         return -3./4/np.pi**2*self.a4*chempo**4+3.*self.ms**2*chempo**2/4\
#     /np.pi**2-3.*self.delta**2*chempo**2/np.pi**2\
#     +(12.*np.log(self.ms/2./chempo)-1)*self.ms**4/32/np.pi**2
#     def dOmega_dchempo(self,chempo):
#         return -3./np.pi**2*self.a4*chempo**3+3.*self.ms**2*chempo/2/np.pi**2\
#     -6.*self.delta**2*chempo/np.pi**2-3.*self.ms**4/8/np.pi**4/chempo
#     def eos_for_intepolation(self,chempo):
#         pressure=toMevfm(-self.Omega(chempo),'mev4')-self.bag
#         baryondensity=toMevfm(-self.dOmega_dchempo(chempo),'mev4')
#         energydensity=chempo*baryondensity-pressure
#         return pressure,energydensity,baryondensity
#     def eosCs2(self,pressure):
#         return 1.0/derivative(self.eosDensity,pressure,dx=pressure*dlnx_cs2)
#     def eosChempo(self,pressure):
#         return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
# # =============================================================================
# # a=EOS_MIT([100.,133**4,0,0.61])
# # =============================================================================
#         
# class EOS_FermiGas(EOS_BPS): #reference: THE PHYSICS OF COMPACT OBJECTS BY SHAPIRO  PAGE24 
#     def __init__(self,args):
#         self.m,self.g=args #m in unit MeV
#         self.baryon_density_s=0.16
#         self.pressure_s=toMevfm(self.m**4,'mev4')
#         self.density_s=self.pressure_s
#         x_intepolation=np.linspace(0,2000./self.m,1001)
#         self.pressureFermiGas=self.eos_for_intepolation(x_intepolation)
#         self.eos_x_from_pressure = interp1d(self.pressureFermiGas,x_intepolation, kind='linear')
# 
#     def phi(self,x):
#         return (x*(1+x**2)**0.5*(2*x**2-3)+3*np.log(x+(1+x**2)**0.5))/(24*np.pi**2)
#     def chi(self,x):
#         return (x*(1+x**2)**0.5*(2*x**2+1)-np.log(x+(1+x**2)**0.5))/(8*np.pi**2)
#     def eos_for_intepolation(self,x):
#         pressure=toMevfm(self.g*self.m**4*self.phi(x),'mev4')
#         #energydensity=toMevfm(self.g*self.m**4*self.chi(x),'mev4')
#         return pressure#,energydensity
#     
#     def eosDensity(self,pressure):
#         return toMevfm(self.g*self.m**4*self.chi(self.eos_x_from_pressure(pressure)),'mev4')
#     def eosBaryonDensity(self,pressure):
#         return toMevfm(self.g*self.m**3*self.eos_x_from_pressure(pressure)**3/(6*np.pi**2),'mev4')
#     def eosCs2(self,pressure):
#         return 1.0/derivative(self.eosDensity,pressure,dx=pressure*dlnx_cs2)
#     def eosChempo(self,pressure):
#         return (self.eos_x_from_pressure(pressure)**2+1)**0.5*self.m
# 
# class EOS_FermiGas_eff(EOS_BPS): #reference: THE PHYSICS OF COMPACT OBJECTS BY SHAPIRO  PAGE24 
#     def __init__(self,args):
#         self.m0,self.ms,self.ns,self.g=args #m in unit MeV
#         self.baryon_density_s=self.ns
#         self.pressure_s=toMevfm(self.m0**4,'mev4')
#         self.density_s=self.pressure_s
#         n_intepolation=np.linspace(0,20,1001)
#         self.pressureFermiGas=self.eos_for_intepolation(n_intepolation)
#         self.eos_n_from_pressure = interp1d(self.pressureFermiGas,n_intepolation, kind='linear')
# 
#     def phi(self,x):
#         return (x*(1+x**2)**0.5*(2*x**2-3)+3*np.log(x+(1+x**2)**0.5))/(24*np.pi**2)
#     def chi(self,x):
#         return (x*(1+x**2)**0.5*(2*x**2+1)-np.log(x+(1+x**2)**0.5))/(8*np.pi**2)
#     def m(self,n):
#         return self.m0*self.ms*self.ns/(self.ms*self.ns+(self.m0-self.ms)*n)
#     def x(self,n):
#         return (6*np.pi**2*n)**(1./3.)/(self.m(n)*5.064e-3)    
#     def eos_for_intepolation(self,n):
#         pressure=toMevfm(self.g*self.m(n)**4*self.phi(self.x(n)),'mev4')
#         #energydensity=toMevfm(self.g*self.m**4*self.chi(x),'mev4')
#         return pressure#,energydensity
#     
#     def eosDensity(self,pressure):
#         n=self.eos_n_from_pressure(pressure)
#         return toMevfm(self.g*self.m(n)**4*self.chi(self.x(n)),'mev4')
#     def eosBaryonDensity(self,pressure):
#         n=self.eos_n_from_pressure(pressure)
#         return toMevfm(self.g*self.m(n)**3*self.x(n)**3/(6*np.pi**2),'mev4')
#     def eosCs2(self,pressure):
#         return 1.0/derivative(self.eosDensity,pressure,dx=1e-8)
#     def eosChempo(self,pressure):
#         n=self.eos_n_from_pressure(pressure)
#         return (self.x(n)**2+1)**0.5*self.m(n)
# 
# 
# def match_eos(args):
#     n_s,n1,p1,e1,dpdn1,n2,p2,e2,dpdn2,match_init=args
#     u1=n1/n_s
#     u2=n2/n_s
#     d=dpdn1
#     b=p1/n_s-d*u1
#     a=e1/n1+b/u1
#     #c*(u2-u1)**gamma=e2/n2+b/u2-a-d*np.log(u2/u1)
#     #c*gamma*(u2-u1)**(gamma)*u2**2=(p2/n_s-d*u2-b)*(u2-u1)
#     gamma=(p2/n_s-d*u2-b)*(u2-u1)/((e2/n2+b/u2-a-d*np.log(u2/u1))*u2**2)
#     c=(e2/n2+b/u2-a-d*np.log(u2/u1))/((u2/u1-1)**gamma)
#     dpdn_match=(c*gamma*u2*(u2/u1-1)**(gamma-2))*(2*(u2-u1)+(gamma-1)*u2)/u1**2+d
#     nep2_match=match_get_eos_array(u2,[a,b,c,gamma,d,u1,n_s])
#     dedn2_match=(nep2_match[1]+nep2_match[2])/nep2_match[0]
#     cs2_match=dpdn_match/dedn2_match
#     return [a,b,c,gamma,d,u1,n_s], gamma>2 and cs2_match<1
# def match_get_eos_array(u_array,args):
#     a,b,c,gamma,d,u1,n_s=args
#     e_array=(a-b/u_array+c*(u_array/u1-1)**gamma+d*np.log(u_array/u1))*n_s*u_array
#     p_array=(b+c*gamma*(u_array/u1-1)**(gamma-1)*u_array**2/u1+d*u_array)*n_s
#     return n_s*u_array,e_array,p_array
# 
# 
# def show_eos(eos,pressure,add_togetherwith):
#     density=eos.eosDensity(pressure)
#     chempo=eos.eosChempo(pressure)
#     baryondensity=eos.eosBaryonDensity(pressure)
#     cs2=eos.eosCs2(pressure)
#     
#     #m0=939.5654
#     #unit_MeV4_to_MeVfm3=1.302e-7
#     #A0=m0**4/np.pi**2*unit_MeV4_to_MeVfm3
#     #v=chempo[i]/m0-1
#     #cs2_bound1=1-(A0*(4./45))*(v*(v+2))**(2.5)/((v+1)*(density+pressure))
#     #cs2_bound2=(density-pressure/3)/(density+pressure)
#     
#     import matplotlib.pyplot as plt
#     f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, sharex=True,figsize=(10, 10))
#     ax1.plot(pressure,density)
#     ax1.set_ylabel('density(MeV fm$^{-3}$)')
#     ax2.plot(pressure,chempo)
#     ax2.set_ylabel('chemical potential(MeV)')
#     ax3.plot(pressure,baryondensity)
#     ax3.set_xlabel('pressure(MeV fm$^{-3}$)')
#     ax3.set_ylabel('baryon density(fm$^{-3}$)')
#     ax4.plot(pressure,cs2,label='$c_{s}^2$')
#     plt.ylim(0,1)
#     #ax4.plot(pressure,cs2_bound1,label='original bound')
#     #ax4.plot(pressure,cs2_bound2,label='modified bound')
#     #ax4.legend(loc=4,prop={'size':8},frameon=False)
#     ax4.set_xlabel('pressure(MeV fm$^{-3}$)')
#     ax4.set_ylabel('sound speed square')
#     if(add_togetherwith==True):
#         show_eos_togetherwith(eos,pressure,ax1,ax2,ax3,ax4)
# 
# def show_eos_togetherwith(eos,pressure,ax1,ax2,ax3,ax4):
# # =============================================================================
# #     B=120
# #     g=2*3*3*0.7
# #     ax1.plot(pressure,3*pressure+4*B)
# #     ax2.plot(pressure,3*((pressure+B)*24*np.pi**2/g/1.302e-7)**0.25)
# # =============================================================================
#     ax4.plot(pressure,1-4.*(eos.eosChempo(pressure)**2-eos.m**2)/(eos.eosChempo(pressure)**2*15.),label='$c_{max}^2$')
#     ax4.plot(pressure,[11.0/15]*100,'--',label='$c_{max}^2$ asymptotic value')
#     ax4.plot(pressure,[5.0/15]*100,'--',label='$c_{s}^2$ asymptotic value')
#     ax4.legend(loc=4,prop={'size':8},frameon=False)
# =============================================================================
    

# =============================================================================
# pressure=np.linspace(0.01,8000,100)
# import matplotlib.pyplot as plt
# f = plt.figure(figsize=(6, 6))
# m0=1000
# ms=np.linspace(1,0.8,11)
# a=EOS_FermiGas([m0,2])
# plt.plot(pressure,a.eosCs2(pressure))
# for i in range(len(ms)):
#     a=EOS_FermiGas_eff([m0,ms[i]*m0,0.16,2])
#     plt.plot(pressure,a.eosCs2(pressure),label='$m_s$=%.2f $m_0$'%ms[i])
# plt.legend(loc=4,prop={'size':8},frameon=False)
# plt.xlabel('pressure ($MeV fm^{-3}$)')
# plt.ylabel(' $c_s^2$')
# =============================================================================


# =============================================================================
# baryon_density0=0.16/2.7
# baryon_density1=1.85*0.16
# baryon_density2=3.74*0.16
# baryon_density3=7.4*0.16
# pressure1=10.0
# pressure2=150.
# pressure3=1000.
# pressure_trans=120
# det_density=100
# cs2=1.0/4
# args=[baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3,pressure_trans,det_density,cs2]
# a=EOS_BPSwithPolyCSS(args)
# pressure=np.linspace(10,200,100)
# show_eos(a,pressure,False)
# =============================================================================

# =============================================================================
# args=[0.059259259259259255, 20.0, 0.29600000000000004, 169.32898412566584, 0.5984, 244.0137227866619, 1.1840000000000002]
# a=EOS_BPSwithPoly(args)
# pressure=np.linspace(1,200,100)
# show_eos(a,pressure,True)
# print 'a'
# =============================================================================

# =============================================================================
# args=[100, 100, 0, 1]
# a=EOS_MIT(args)
# pressure=np.linspace(1,200,100)
# show_eos(a,pressure)
# print 'a'
# =============================================================================
