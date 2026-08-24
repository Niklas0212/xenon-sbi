WimPyDD_PATH='WimPyDD/'
#WimPyDD_PATH=''
EXPERIMENTS_PATH=WimPyDD_PATH+'Experiments'
HALO_FUNCTIONS_PATH=WimPyDD_PATH+'Halo_functions'

'''
WimPyDD is a modular, object–oriented and customizable Python code
that calculates accurate predictions for the expected rates in WIMP
direct–detection experiments within the framework of
Galilean–invariant non–relativistic effective theory in virtually any
scenario, including inelastic scattering, an arbitrary WIMP spin and a
generic WIMP velocity distribution in the Galactic halo. WimPyDD
exploits the factorization of the three main components that enter in
the calculation of direct detection signals: i) the Wilson
coefficients of the effective theory, that encode the dependence of
the signals on the theoretical parameters; ii) a response function
that depends on the nuclear physics and on the features of the
experimental detector (acceptance, energy resolution, response to
nuclear recoils); iii) a halo function that depends on the WIMP
velocity distribution and that encodes the astrophysical inputs. In
WimPyDD these three components are calculated and stored separately
for later interpolation, combining them together only as the last step
of the signal evaluation procedure. This makes the phenomenological
study of the direct detection scattering rate with WimPyDD transparent
and fast also when the parameter space of the WIMP model has a large
dimensionality.

WimPyDD can be downloaded from:

https://wimpydd.hepforge.org/
'''

try:
    import numpy as np
except:
    import pip
    pip.main(['install', 'numpy'])
    import numpy as np
try:
    import matplotlib.pyplot as pl
except:
    import pip
    pip.main(['install', 'matplotlib'])
    import matplotlib.pyplot as pl
import os

# Modification for Python 3.11
try:
    from inspect import getfullargspec as getargspec
except ImportError:
    pass  # getargspec not available in newer Python

import pickle
try:
    import scipy.special as sp
except:
    import pip
    pip.main(['install', 'scipy'])
    import scipy.special as sp
try:
    import keyboard
except:
    import pip
    pip.main(['install', 'keyboard'])
    
from functools import reduce
import sys, select
import scipy.optimize as op
import scipy.integrate as integrate
import subprocess
import inspect
#from scipy import interpolate
#import yellin
from scipy.integrate import tplquad
import scipy.optimize as op
from scipy.stats import poisson,norm
import time as win_time
from pathlib import Path as win_path


class documented_float(float):
    def __init__(self,input):
        self=input

class documented_list(list):
    def __init__(self, *args, **kwargs):
        super(documented_list, self).__init__(args[0])



class diff_response_functions(dict):
    '''Dictionary containing the differential response functions for as defined in ...

       Keys: [model_name,spin,coupling1,coupling2,element]
       Output: numpy array of functions with shape (2,2,n_isotopes,4)=(tau,tau_prime,isotope,a) with:
       tau,tau_prime: integers with values 0 or 1 identifying isospin indices
       isotope=integer that identifies isotope of element
       a=integer with possible values 0,1,2(=1E),3(=1E^{-1}) for decomposition of velocity dependence'''
    
    def __init__(self, *args, **kwargs):
        super(diff_response_functions, self).__init__(args[0])

class response_functions(dict):
    '''Dictionary containing the integrated response functions as defined in ... versus the recoil energy.

    Keys: [model,j_chi]

    with model an object belonging to the eft_hamiltonian class and j_chi the spin of the particle

    Output:

    tuple of arrays with indices [a,tau,tau_prime,n_bin,n_element,n_isotope] where:

    a:  integer with possible values 0,1,2(=1E),3(=1E^{-1}) for decomposition of velocity dependenc
    tau,tau_prime: integers with values 0 or 1 identifying isospin indices
    n_bin: integer that identifies one of the experimental bins as initialized in data.tab
    n_element: integer that identifies one of the elements in the target object (target.element[n_element])
    n_isotope: integer starting from 1 that identifies the isotope of target.element[n_element] 
    (n_isotope=0 returns an array of recoil energy values)

    Example:

         er, rbar=exp.response_functions[3,1,0,0,0,0],exp.response_functions[3,1,0,0,0,3]
         pl.plot(er,rbar)

    plots rbar vs. ER for:

         a=3=1E^{-1} (velocity decomposition)
         tau=1, tau_prime=0
         n_bin=0 (firts energy bin)
         n_element=0 (first element in target)
         n_isotope=0  (recoil energy values in keV)
         n_isotope=3  (third isotope of element, exp.target.element[2])  

    '''
    def __init__(self, *args, **kwargs):
        super(response_functions, self).__init__(args[0])

        
        

        
        
class documented_array(np.ndarray):
    def __init__(self,input):
        self=list(input)
        

        
        
        
        

class documented_tuple(tuple):
    def __init__(self,input):
        self=input



n_coeff_vec_spin_0=[0,2,6,9]
n_coeff_vec_spin_1_2=[ 0,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]
n_coeff_vec_spin_1=[0,3,4,7,8,9,10,13,16]

n_coeff_vec={0:n_coeff_vec_spin_0, 0.5:n_coeff_vec_spin_1_2, 1:n_coeff_vec_spin_1}


def mask_spin(j_chi):
    return np.array([np.choose(n in n_coeff_vec[j_chi],[0,1]) for n in range(18)])
        
PATH='WimPyDD'

def get_element(filename):
    f = open(filename, 'r')
    a=np.array([],dtype=int)
    abundance=np.array([])
    spin=np.array([])
    itar=np.array([],dtype=int)    

    name=f.readline().strip()
    symbol=f.readline().strip()
    z=int(f.readline().strip()) 
    
    for line in f:
        line = line.strip()
        columns = line.split()
        a = np.append(a,int(columns[0]))
        abundance = np.append(abundance,float(columns[1]))
        spin = np.append(spin,float(columns[2]))            
        itar = np.append(itar,int(columns[3]))                
    f.close()

    return name,symbol,z,a,abundance,spin,itar

def  get_w_tab():
    filename='nuclear_response_functions_coefficients_table.dat'
    f = open(PATH+'/Targets/Nuclear_response_functions/'+filename, 'r')
    data_inp=np.array([])
    for line in f:
        line = line.strip()
        columns = line.split()
        for i in range(len(columns)):
            data_inp=np.append(data_inp,float(columns[i]))
        data_inp=data_inp.reshape(-1,len(columns))
    f.close()
            
    ii=-1

    ### added fictitious 31th element with vanishing matrix.
    ### for elements with i_tar=0 (missing form factor) the matrix data_mod(itar-1,...)
    ### =data_mod(-1,...) is used. In python this means that the last element
    ### with all vanishing coefficients is used
    #### WARNING!!! tau and tau_prime are inverted because the table contains the output
    #### of Anand et al. code where M-Phi'' and Sigma'-Delta are calculated instead of
    ####  Phi''-M and Delta-Sigma', as indicated in the WIMP response functions.
    #### All other response functions are symmetric in tau, tau_prime
    data_mod=np.zeros(data_inp.shape[0]*data_inp.shape[1]).reshape(31,8,2,2,11)
    for j in range(31):
        for k in range(8):
            for k1 in range(2):
                for k2 in range(2):
                    ii+=1
                    for k3 in range(11):
                        #data_mod[j,k,k1,k2,k3] = data_inp[ii,k3]                        
                        data_mod[j,k,k2,k1,k3] = data_inp[ii,k3]

    return data_mod


data_mod=get_w_tab()


def define_w(element,isotope,verbose=True):

    filename=element.isotopes[isotope]+'_func_w'

    if not os.path.isfile(PATH+'/Targets/Nuclear_response_functions/'+filename+'.py'):
        #if element.name in ['fluorine', 'sodium', 'germanium', 'iodine', 'xenon']:
        #    print('Anand et al. definition for '+element.isotopes[isotope]+' from 1308.6288 used for nuclear W functions as default')
        #else:
        #    print('Catena and Schwabe definition for '+element.isotopes[isotope]+' from 1501.03729 used for nuclear W functions as default')            
        def func_w_default(q):
            '''Definition from Phys.Rev.C 89 (2014) 6, 065501 (e-Print: 1308.6288[hep-ph]) used\nfor nuclear W functions as default''' 
    
            hbarc=197.e-3 #GeV*fm, q is supposed to be in GeV    
            a_nucleus=element.a[isotope]
            b=np.sqrt(41.467e0/(45.e0*a_nucleus**(-1./3.)-25.*a_nucleus**(-2./3.)))    
            y=(b*q/(2.e0*hbarc))**2
            
            return func_w(element.data_mod[isotope],y)


        return func_w_default

    elif os.path.isfile(PATH+'/Targets/Nuclear_response_functions/'+filename+'.py'):
        if verbose:
            print('Nuclear form factor definition for '+element.isotopes[isotope]+' from '+PATH+'/Targets/Nuclear_response_functions/'+filename+'.py')

        func_w1=get_function_dir(PATH+'/Targets/Nuclear_response_functions',filename,verbose=verbose)
        #func_w1=getattr(__import__(PATH+'/Targets/Nuclear_response_functions/'+filename), 'f')
        
        def func_w_custom(q):
            func_w=get_function_dir(PATH+'/Targets/Nuclear_response_functions',filename,verbose=verbose)
            #func_w=getattr(__import__(PATH+'/Targets/Nuclear_response_functions/'+filename), 'f')
            func_w_custom.__doc__=func_w.__doc__
            if len(inspect.signature(func_w).parameters)==3:
                return func_w(element,isotope,q)
            else:
                return func_w(q)
        func_w_custom.__doc__=func_w1.__doc__        
        return func_w_custom


class element(object):
    '''
      Initializes an element of the periodic table
      A set of elements is pre-defined in WimPyDD. Type list_elements() to list them.
      ---------------------------------------------------------------------------------------------------------------------------------
      Parameters: 
		- symbol(str)   - A string starting with capital letter with the symbol of the element. Must match the name of the file 
      		  	           symbol+'.tab' in the Directory WimPyDD/Target. By default uses the nuclear form factor calculated by
				   Anand et al, Phys. Rev.C  89, 065501 (2014) or Catena et al., JCAP04(2015)042. A custom nuclear form factor
				   can be used if the file isotope_name+'_func_w.py' is present in WimPyDD/Target/Nuclear_form_factors/.
				   isotope_name must be one of the strings contained in the isotopes attribute array.
				   
				   Example: Xe=WimPyDD.element('Xe') requires the file WimPyDD/Target/Xe.tab and, 
                                   if a custom form factor is used for
				   133Xe, the file WimPyDD/Target/Nuclear_form_factor/131Xe.tab must exist. 
                                   The array Xe.isotopes must contain the string '131Xe'.
				   
		  - verbose(bool) - If True prints out details of accessed external files. Default=False
      ---------------------------------------------------------------------------------------------------------------------------------
      Attributes:
		  - a(array)		- an array containing the atomic mass numbers of the isotopes of the element
		  - abundance(array)	- array containing the natural fractional abundance of the element isotopes 
		  - average_a(float)	- element atomic mass number averaged over isotopes
		  - average_mass(float) - element mass in GeV averaged over isotopes
		  - isotopes(array)     - array of strings with the symbols of the isotopes of the element
		  - itar(array)         - array containing internal codes to access the default Nuclear functions coefficients in
		                          WimpyDD/Target/Nuclear_response_functions/nuclear_response_functions_coefficients_table.dat  

		  - mass(array)         - an array containing the atomic masses in GeV of the isotopes of the element
		  - n_isotopes(int)     - the number of isotopes
		  - name(str)           - name in full of the element
		  - nt_kg(array)        - array containing the number of targets per kg of the isotopes of the element (for a sample of pure element)
		  - nt_kg_average(float)- number of targets per kg averaged over the isotopes of the target
		  - spin(array)         - an array containing the spins of the isotopes of the element (in multiples of 0.5)
		  - symbol              - symbol of the element
		  - z(int)              - atomic number of the element
		  - func_w(array)       - array of functions w(q) containing the nuclear form factors of the isotopes of the element			
		    			- w(q)   - element of the array func_w
				       ----------------------------------
				       Calculates the 8 nuclear form factors M, Sigma', Sigma'', Phi'', Phi_tilde', Delta, Phi''-M, Delta-Sigma'
				       of the isotope. 

                                        By default WimPyDD uses the response functions calculated in Anand et al., 1308.6288 
                                        and Catena et el., 1501.03729. They are available for the following 30 targets: 
                                       
                                        01)C 02)19F  03)23Na 04)28Si  05)70Ge   06)72Ge   07)73Ge   08)74Ge   09)76Ge   10)127I
                                        11)128Xe 12)129Xe  13)130Xe  14)131Xe  15)132Xe  16)134Xe  17)136Xe 18)16O 19)40Ar 20) 40Ca
                                        21) H 22)3He  23)4He 24)14N 25)20Ne 26)24Mg 27)27Al 28)32S 29)56Fe 30)59Ni   
                                        
                                        Polinomial fits are used to calculated them, whose coefficients are stored in the file:


                                            WimPyDD/Targets/Nuclear_response_functions/nuclear_response_functions_coefficients_table.dat  

				       Such definitions can be overriden, or response functions for missing isotopes can be implemented by
                                       adding to the same directory a file:

                                           isotope_name+'_func_w.py'

                                        with isotope_name one of the entries of the element.isotopes attribute. For instance, for tungsten:

                                             tungsten=element('W')  
                                             tungsten.isotopes ->array(['180W', '182W', '183W', '184W', '186W'])  

                                        The nuclear response functions for 183W can be implemented by adding the file:

                                           183W_func_w.py

                                        to the folder WimPyDD/Targets/Nuclear_response_functions.
                                        The file must contain a function ot the transferred momentum q (in GeV):

                                            def f(q):
                                                ....
                                                return output
                                       
                                       with the same behaviour of the custom functions.  
                                       In particular output.shape=(8,2,2), with
                                       the first index indicating one among the 8 nuclear response functions:

                                           0=M, 1=Sigma'', 2=Sigma', 3=Phi'', 4=Phi_tilde', 5=Delta, 6=Phi''-M, 7=Delta-Sigma'

                                       while the last two indices indicate tau and tau_prime (nuclear isospin indices).
                                       For instance f(0.01)[5,0,1]   -> W_{Delta}^{01}(q=0.01 GeV) 

				       Input:
				       --------------------------------------------------------
		            	       - q - exchanged momentum in GeV
  				       ---------------------------------------------------------
				       Output:
				       array with shape (8,2,2) containing 8 2x2 matrices, each for every form factor X=M, Sigma', Sigma'', Phi'',
				       Phi_tilde', Delta, Phi''-M, Delta-Sigma'. 
                                       Each 2x2 matrix represents W_X^(tau,tau_prime) with tau=0,1, tau+prime=0,1
				       two isospin indices.
    
    '''
    #i_tar : target
    #  01)C 02)19F  03)23Na 04)28Si  05)70Ge   06)72Ge   07)73Ge   08)74Ge   09)76Ge   10)127I
    #  11)128Xe 12)129Xe  13)130Xe  14)131Xe  15)132Xe  16)134Xe  17)136Xe 18)16O 19)40Ar 20) 40Ca
    #  21) H 22)3He  23)4He 24)14N 25)20Ne 26)24Mg 27)27Al 28)32S 29)56Fe 30)59Ni       
    def __init__(self, symbol,verbose=False):
        self.name,self.symbol,self.z,self.a,self.abundance,self.spin,self.itar=get_element(PATH+'/Targets/'+symbol+'.tab')
        self.symbol=symbol
        self.mass=0.931*self.a
        self.average_a=sum(self.a*self.abundance)/sum(self.abundance)
        self.average_mass=self.average_a*0.931
        self.isotopes=np.core.defchararray.add(np.array(["%.i" % x for x in self.a]), self.symbol)
        self.nt_kg_average=1000./self.average_a*6.02e23
        self.nt_kg=self.abundance*1000./self.average_a*6.02e23                

        self.n_isotopes=np.size(self.a)
        self.data_mod=data_mod[self.itar-1,...]

        self.func_w=np.array([])
        for i in range(len(self.isotopes)):
            self.func_w=np.append(self.func_w,define_w(self,i,verbose=verbose))
            #self.func_w=np.append(self.func_w,fix_func_w(define_w(self,i,verbose=verbose))) 

        
        #self.eft=np.array([])

        #for i in np.arange(np.size(self.a)):
         #   self.eft=np.append(self.eft,lambda er: eft_amplitude_squared(coeff,np.sqrt(2*self.mass[i]*er*1e-6),self,i))

    def __add__(self,other):

        if type(other)== element:
            return target(self.symbol+other.symbol)
        elif type(other) == target:
            return target(self.symbol+other.formula)
         
    def __mul__(self,other):

    
        formula=self.symbol+str(other)


        return target(formula)


    def __rmul__(self,other):
        return self.__mul__(other)


         
    def __str__(self):        
        return str(self.name)+", symbol "+self.symbol+", atomic number "+str(self.z)+", average mass "+"{:.3f}".format(self.average_mass)+", "+str(self.isotopes.size)+" isotopes."

def read_formula(formula):
    pos = [i for i,e in enumerate(formula+'A') if e.isupper()]
    parts = [formula[pos[j]:pos[j+1]] for j in range(len(pos)-1)]
    lengths=[len([t for t in p if not t.isdigit()]) for p in parts]
    symbols=[p[:l] for p,l in zip(parts,lengths)]
    stoichiometrics_strings=[p[l:] for p,l in zip(parts,lengths)]
    stoichiometrics_strings_final=[x if x else '1' for x in stoichiometrics_strings]
    stoichiometrics=[int(s) for s in stoichiometrics_strings_final]

    if len(symbols)>0 and formula[0].isupper():
        return np.array(symbols),np.array(stoichiometrics) 
    

class target(element):
    '''
      Initialize the target with given elements combining objects belonging to element class.
      ---------------------------------------------------------------------------------------------------------------------------------
      Parameters: 
		  -formula(str)   - A string starting with combination of elements. The first letter of each element should be capital letter.
				    Each element string must be a valid input for the element class.

				    Example: C3F8=WimPyDD.target('C3F8') uses WimPyDD.element('C') and WimPyDD.element('F').

		  -verbose(bool) - If True prints out details of accessed external files. Default=False
      ---------------------------------------------------------------------------------------------------------------------------------
      Attributes:
		  - element(array)		- an array containing the class elements of the formula.
		  - formula(str)		- name of target formula.
		  - mass(fload)		- sum of total average mass of elements.
		  - n(array)		- an array containing the stoichiometric coefficient of each element.
		  - n_targets(int)		- the number of element types.
		  - nt_kg(array)		- an array containing total mass in kg unit per mol of each element.

      An object belonging to the target class can also be initialized as a linear combination of objects belonging to the element class.

      Ex: c3f8=3*element('C')+8*element('F')
          type(c3f8) -> <class target>
          As a consequence, a since-element target can be obtained by multiplying by 1 an element object:
          ge=element('Ge')
          type(ge) -> <class element>
          type(1*ge) -> <class target>
    '''
    
    def __init__(self, formula,verbose=False):
        symbols,self.n=read_formula(formula)

        self.element=np.array([])
        for s in symbols:
            self.element=np.append(self.element,element(s,verbose=verbose))
            
        
        self.n_targets=self.n.size        

        self.mass=0.
        a_tot=0.

        self.formula=formula

        
        for i,target in enumerate(self.element):
            self.mass+=self.n[i]*target.average_mass            
            a_tot+=self.n[i]*target.average_a

        self.nt_kg=self.n*1000./a_tot*6.02e23
        
        for i in range(self.n_targets):
            self.element[i].nt_kg_average=self.n[i]*1000./a_tot*6.02e23
            self.element[i].nt_kg=self.element[i].nt_kg_average*self.element[i].abundance

    def __add__(self,other):

        if isinstance(other,target):
            return target(clean_formula(self.formula+other.formula))
        else:
            return target(clean_formula(self.formula+other.symbol))
        
    def __mul__(self,other):

        if not isinstance(other, int):
            print(str(other)+' is not an integer. '+self.formula+' left unchanged.')
            return self
            
        symbols,n=read_formula(self.formula)

        n_new=[]
        for nn in n:
            n_new+=[other*nn]
        
        new_formula=''

        for s,nn in zip(symbols,n_new):
            if nn > 0:
                if nn>1:
                    new_formula+=s+str(nn)
                else:
                    new_formula+=s

        return target(new_formula)


    def __rmul__(self,other):
        return self.__mul__(other)

    
            
    def __str__(self):
        output=self.formula+' contains:\n'
        for i in range (self.n_targets):
            output+=str(self.element[i])+'\n'
        output+='Isotope-averaged mass: '+str(self.mass)
        return output


def read_tuple_from_file(file):
    '''Reads a file whose lines are sequences of floats. Returns a tuple
    of arrays, where each array contains the floats of one line.  So
    len(tuple) is equal to the number of lines in the file.  Unreadable lines
    are ignored with exception of the last, which is stored for documentation.'''

    
    f = open(file, 'r')
    nline=0
    tuple=()
    help='No help provided'
    for line in f:
        try:
            line = line.strip()
            columns =line.split()
            values=np.array([])
            for i in range(len(columns)):
                values=np.append(values,float(columns[i]))
            tuple+=((values,))
            nline+=1
        except:
            help=str(line)
    f.close()
    n_points=nline
    return tuple,help


def get_exposure(file,verbose=True):
    if os.path.isfile(file+'.tab'):
        if verbose:
            print('loading '+file+'.tab')

        f = open(file+'.tab', 'r')
        output=()
        info='No help provided'

        for line in f:
            try:
                exec('get_exposure.t='+line)
                if isinstance(get_exposure.t, int) or isinstance(get_exposure.t,float) or isinstance(get_exposure.t,list) or isinstance(get_exposure.t,np.ndarray):
                    output+=get_exposure.t,
                else:
                    info=str(line)
            except:
                info=str(line)        

        output+=info,

        return output
    else:
        if verbose:
            print(file+'.tab missing, using 1. as default. ')                
        return (1,'No input file provided, using 1. as default')

    

    return output
    

def get_data(input,verbose=True):

    if os.path.isfile(input+'.tab'):
        if verbose:
            print('loading '+input+'.tab')        
        return read_tuple_from_file(input+'.tab')
    else:
        if verbose:
            print(input+'.tab missing, using ([1.]) as default. ')                
        return (np.array([1.]),),'No input file provided, using (np.array([1.]),) as default'


    


def get_function_dir(directory,file,verbose=True):
    
    if os.path.isfile(directory+'/'+file+'.py'):

        path=directory.replace('/','.')+'.'+file

        file1=open(directory+'/'+file+'.py','r')

        for line in file1:
            if 'def' in line:
             function_name=line[line.find('def (')+len('def ')+1:line.find('(')]
             break

        
        if verbose:
            print('loading function '+function_name+' from '+directory+'/'+file+'.py')

        #Direct use of __import__() is rare, except
        #in cases where you want to import a module whose
        #name is only known at runtime.
        f=getattr(__import__(path, fromlist=[function_name]), function_name)
        
        
        return f
        #__import__(input)

        #mymodule = sys.modules[input]
        #return mymodule.f

        
    elif os.path.isfile(directory+'/'+file+'.tab'):
        if verbose:
            print('loading '+directory+'/'+file+'.tab')        
        f = open(directory+'/'+file+'.tab', 'r')
        nline=0
        x=np.array([])
        y=np.array([])
        help='No help provided'        
        for line in f:
            try:
                line = line.strip()
                columns = line.split()
                x=np.append(x,float(columns[0]))
                if len(columns)==2:
                    y=np.append(y,float(columns[1]))            
                nline+=1

            except:
                if nline==0:
                    help=str(line)

                
        f.close()
        if nline==1:
            if verbose:
                print('only value '+str(x[0])+' in '+file+'.tab used as constant')

            def f(e):
                return x[0]
            
            f.__doc__ =help
            return f

        else:    
            #tck=interpolate.splrep(x, y, s=0)

            def f(e):
                #return interpolate.splev(e,tck,der=0,ext=3)
                f_out=np.interp(e, x, y)
                return np.choose(f_out>0,[0.,f_out])

            f.__doc__ =help            
            return f
    
    else:
        if verbose:
            print('using 1 as default for '+file)        
        def f(e):
            return 1
        f.__doc__ ='This function was not initialized and automatically set to 1'          
        return f


def overlap(x,y):
    xmin=min(x)
    xmax=max(x)
    ymin=min(y)
    ymax=max(y)

    if xmax<ymin or ymax<xmin:
        return (False, np.array([0.,0.]))

    return (True, np.array([max(xmin,ymin),min(xmax,ymax)]))

    

def print_spin(j_chi):
    if (2*j_chi % 2) == 0:
        return str(int(j_chi))
    else:
        return str(int(j_chi/0.5))+'_2'


def print_coupling(c):
    if isinstance(c, int):
        return str(c)
    else:
        if isinstance(c,tuple):
            if isinstance(c[0], int):
                return str(c[0])+'_'+'_'.join(str(t) for t in c[1:])
            else:
                if len(c)==3:
                    return c[0]+'_'+str(c[1])+'_'+str(c[2])
                else:
                    return c[0]+'_'+str(c[1])+'_'+str(c[2])+'_'+'_'.join(str(t) for t in c[3:])

def print_coupling_short(c):
    if isinstance(c, int):
        return str(c)
    else:
        if isinstance(c,tuple):
            if isinstance(c[0], int):
                return str(c[0])
            else:
                return c[0]+'_'+str(c[1])+'_'+str(c[2])


def get_short_coupling(c):
    if isinstance(c, int):
        return c
    else:
        if isinstance(c,tuple):
            if isinstance(c[0], int):
                return c[0]
            else:
                return c[:3]
                
def print_latex_coupling(c):
    if isinstance(c, int):
        return str(c)
    elif isinstance(c,tuple):
        if isinstance(c[0], int):
            return str(c[0])+','+','.join([t for t in c[1:]])
        else:
            if len(c)==3:
                if c[0]!='M':
                    return '\\'+c[0]+','+str(c[1])+','+str(c[2])
                else:
                    return c[0]+','+str(c[1])+','+str(c[2])
            elif len(c)>3:
                if c[0]!='M':
                    return '\\'+c[0]+','+str(c[1])+','+str(c[2])+','+','.join([t for t in c[3:]])                    
                else:
                    return c[0]+','+str(c[1])+','+str(c[2])+','+','.join([t for t in c[3:]])                    

def print_latex_coupling_short(c):
    if isinstance(c, int):
        return str(c)
    elif isinstance(c,tuple):
        if isinstance(c[0], int):
            return str(c[0])
        else:
            if c[0]!='M':
                return '\\'+c[0]+','+str(c[1])+','+str(c[2])
            else:
                return c[0]+','+str(c[1])+','+str(c[2])


def convert_to_all_spins(c):
    c_out=c
    if isinstance(c, int):
        c_out=haxton_to_all_spins[c]
        return c_out
    else:
        if isinstance(c,tuple):
            if isinstance(c[0], int):
                c_out=haxton_to_all_spins[c[0]]+c[1:]
    return c_out


haxton_to_all_spins={1:('M',0,0), 3:('Phi',0,1), 4:('Sigma',1,0), 5:('Delta',1,1), 6:('Sigma',1,2), 7:('Omega',0,0), 8:('Delta',1,0), 9:('Sigma',1,1), 10:('Sigma',0,1), 11:('M',1,1), 12:('Phi',1,0), 13:('Phi',1,1), 14:('Omega',1,1), 15:('Phi',1,2)}

all_spins_to_haxton={v:k for k,v in list(haxton_to_all_spins.items())}


            
def symmetric_interference(__c1__,__c2__):
    c1_all_spins,c2_all_spins=convert_to_all_spins(__c1__),convert_to_all_spins(__c2__)    
    output=False
    
    x1,l1,s1=c1_all_spins[:3]
    x2,l2,s2=c2_all_spins[:3]

    if l1==l2:
        if x1=='Phi' and x2=='Phi' or x1=='Sigma' and x2=='Sigma':
            if s1==l1-1 and s2==l2+1 or s1==l1+1 and s2==l2-1:
                output=True
            
    return output

def non_symmetric_interference(__c1__,__c2__):
    c1_all_spins,c2_all_spins=convert_to_all_spins(__c1__),convert_to_all_spins(__c2__)
    output=False

    x1,l1,s1=c1_all_spins[:3]
    x2,l2,s2=c2_all_spins[:3]

    if l1==l2:
        if x1=='Phi' and x2=='M':
            if s1==l1-1 and s2==l2 or s1==l1+1 and s2==l2:
                output=True
        if x1=='Delta' and x2=='Sigma':
            if s1==l1 and s2==l2-1 or s1==l1-1 and s2==l2:
                output=True
            
    return output

def is_interference(__c1__,__c2__):
    return non_symmetric_interference(__c1__,__c2__) or symmetric_interference(__c1__,__c2__)


def get_default_args(func):
    signature = inspect.signature(func)
    return {k:v for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def __fix_coupling__(c):
    c_out=c
    if isinstance(c,tuple):
        if not isinstance(c[0],int) and not isinstance(c[0],str):
            c_out=(int(c[0]),)+c[1:]
    else:
        c_out=int(c)
    return c_out



class eft_hamiltonian(object):
    '''
      Sets the effective hamiltonian.
      ---------------------------------------------------------------------------------------------------------------------------------
      Parameters: 
		- name(str)		        - user's choice name for the Hamiltonian

		- wilson_coefficients(dict)	- wc={n1: func1 , n2: func2, ... } containing a list of effective operators 
                                                  and Wilson coefficients. 

                                                  The keys n1,n2,... of the dictionary wc identify the operators in two alternative bases 
                                                  (N.B. WimpyDD does not allow to mix the two different sets in the same Hamiltonian):
                                                
                                                  - one of the 14 integers 1,3,4,...,15, following the convention of 
                                                    Anand et. al, Phys. Rev. C89 (6) (2014) 065501. arXiv:1308.6288

                                                  - a tuple (X,s,l) following the convention of P. Gondolo at al., arXiv:2008.05120.
                                                    X = nuclear current with possible values:
                                                       X = 'M', 'Omega', 'Sigma', 'Delta' and 'Phi' 
                                                    s = rank of the operator with possible values:
                                                       s = 0,1,...,2*J_chi for a WIMP of spin j_chi
                                                    l = power of the transferred momentum q in the operator with possible values: 
                                                       l = s for X = 'M', 'Omega'
                                                       l=s-1,s,s+1 for X='Sigma','Phi'
                                                       l=s-1,s for X='Delta'
                                               
                                                Examples: n1=4, n2=6
                                                          n1=('Sigma',1,0), n2=('Sigma',1,2)
                                                          (These two sets of operators are identical - up to a sign in the Wilson coefficient- 
                                                           see Table 1 of arXiv:2008.05120)

                                                 The values func1, func2, ... of the dictionary wc are functions of arbitrary parameters arg1,arg2,... 
                                                 that return two-dimensional arrays (or lists) containing the Wilson coefficients in isospin base 
                                                 [c0=cp+cn, c1=cp-cn] in GeV^2. 
                        
                                                 Example:
  
                                                         def func1(arg1,arg2):
                                                             ...
                                                             return [c0,c1]

                                                 The argument names mchi, delta and q are reserved for the WIMP mass in GeV, the mass splitting 
                                                 in keV and the transferred momentum in GeV.

                                                 .....
                                                  
                                                 Example:

                                                         def func2(arg2,arg3,q,q0=0.01):
                                                             ...
                                                             return 1/q**2*np.array([c0,c1]) 

                                                 The momentum dependence of a Wilson coefficient is trasferred to the operator, i.e.

                                                 c(q,q0=0.01)O_n=c(q0)*[c(q)/c(q_0)*O_n]=c(q_0=0.01)*O_n'(q), O_n'(q)=c(q)/c(q_0)*O_n

                                                 and the response functions are calculated and stored for O_n'(q) instead of O_n.
                                                 Any argument name starting by q0 (ex: q0_propagator, q0_massless) is reserved to
                                                 fix the momentum normalization in GeV. If in the Wilson coefficient no such argument is present 
                                                 its momentum normalization is assumed to 1 GeV.
                         
                                                 Any argument in the Wilson coefficients can have default values the signal routines do not pass them.
                                                 If the same argument appears in more than one Wilson coefficient it is sufficient to set its default
                                                 value in one of them. Setting different default values to the same argument in different 
                                                 Wilson coefficients leads to unpredictable                                                   
                                                 If an argument has no dafault value and is not passed by the signal routine it is set to 1.
                                                 
                                                 Momentum normalizations (reserved variable starting by q0) are set at instantiation and 
                                                 cannot be changed at run time.                                           

       Example:

                   o4_o6=eft_hamiltonian('o4_o6',{4: lambda M, r=0: [(1+r)/M**2,(1-r)/M**2], 
                         6: lambda r: [(1+r),(1-r)]})   

       In this case the progressive numbers 4 and 6 (possible choices: 1,3,...15) are used for the effective 
       operators following the convention of Anand et al., 1308.6288. The Wilson coefficient functions are 
       defined on-the-fly using the lambda symbol.

       The same effective Hamiltonian can be defined with:

                   o4_o6_all_spins=eft_hamiltonian('o4_o6',{('Sigma',0,1): lambda M, r=0: [(1+r)/M**2,(1-r)/M**2], 
                         ('Sigma',1,2): lambda r: [-(1+r),-(1-r)]}) 

       where the effective operators are defined following the convention (X,s,l) of Gondolo et al., 
       2008.05120. The possible choices of X, s, l for a WIMP of spin j_chi are:

           X='M','Omega','Sigma','Phi','Delta'
           s=0,1,2,...,2*j_chi 
           l=s for X='M','Omega'
           l=s-1,s,s+1 for X='Sigma','Phi'
           l=s-1,s for X='Delta'
    
       Notice that O_6=-O_Sigma_1_2 (see Table 1 of 2008.05120), hence the relative minus sign between 
       the second Wilson coefficients of o4_o6 and o4_o6_all_spins.

       The hamiltonian features can be seen with a print():

       print(o4_o6)
       Hamiltonian name:o4_o6
       Hamiltonian:c_4(M, r=0)* O_4+c_6(r)* O_6
       Squared amplitude contributions:
       O_4*O_4, O_4*O_6, O_6*O_4, O_6*O_6

       or:

      print(o4_o6_all_spins)
      Hamiltonian name:o4_o6
      Hamiltonian:c_Sigma_1_0(M, r=0)* O_Sigma_1_0+c_Sigma_1_2(r)* O_Sigma_1_2
      Squared amplitude contributions:
      O_Sigma_1_0*O_Sigma_1_0, O_Sigma_1_0*O_Sigma_1_2, O_Sigma_1_2*O_Sigma_1_0, O_Sigma_1_2*O_Sigma_1_2

       The parameters of the Wilson coefficient can be passed with the **args argument of signal routines.
       If they are not passed defaut values are used, or 1 if no default values are set.
       For instance:
       - wimp_dd_rate(exp,o4_o6,vmin,delta_eta,mchi) 
         calculates the signals using M=1 (no default value for M) and r=0 (default value for r defined in C_4) 

       - wimp_dd_rate(exp,o4_o6,vmin,delta_eta,mchi,M=1e3,r=1) 
         calculates the signals using M=1e3, r=1    

       A call to the load_response_function routine (see related help) loads or writes a file with the tabulated 
       response functions for each Wilson coefficient combination of the squared amplitude. File names 
       contain the keys used to indicate the operators.

                     For o4_o6:           For o4_o6_all_spins:
                     c_4_c_4.npy          c_Sigma_1_0_c_Sigma_1_0.npy
                     c_4_c_6.npy          c_Sigma_1_0_c_Sigma_1_2.npy
                     c_6_c_4.npy          c_Sigma_1_2_c_Sigma_1_0.npy
                     c_6_c_6.npy          c_Sigma_1_2_c_Sigma_1_2.npy

       Wilson coefficients with explicit momentum dependence are handled as modified operators  
       O_n'(q)=c(q)/c(q_0)*O_n, so require to modify the operator key with one or more specification strings
       in order to read/write in a different file compared to unmodified operators. In particular:

        n     ->  (n,string1,string2,...)
        X,s,l ->  (X,s,l,string1, string2,...)

       For instance:
                                                  
        o4_o6_momentum_dependence=eft_hamiltonian('o4_o6_qm2',{4: lambda M, r=0: [(1+r)/M**2,(1-r)/M**2], 
        (6,'qm2'): lambda r,q,q0=0.01: [(1+r)/q**2,(1-r)/q**2]})   

        o4_o6_all_spins_momentum_dependence=eft_hamiltonian('o4_o6_qm2',{('Sigma',1,0): 
        lambda M, r=0: [(1+r)/M**2,(1-r)/M**2], 
        ('Sigma',1,2,'qm2'): lambda r,q,q0=0.01: [-(1+r)/q**2,-(1-r)/q**2]})   

                           
                    Output/input files:
                     c_4_c_4.npy                    c_Sigma_1_0_c_Sigma_1_0.npy
                     c_4_c_6_qm2.npy                c_Sigma_1_0_c_Sigma_1_2_qm2.npy
                     c_6_qm2_c_4.npy                c_Sigma_1_2_qm2_c_Sigma_1_0.npy
                     c_6_qm2_c_6_qm2.npy            c_Sigma_1_2_qm2_c_Sigma_1_2_qm2.npy

        N.B.:
       - wimp_dd_rate(exp,o4_o6_momentum_dependence,vmin,delta_eta,mchi,M=1e3,r=1,q0=1) 
         calculates the signals using M=1e3, r=1, q0=0.001 (the momentum normalization q0 cannot be set at run time)  
                                                                     
--------------------------------------------------------------------------------------------------------------------------------
      Attributes:
		  - arguments(dict)			- a dictionary {n1 : ['arg1', arg2',...], n2:['arg3', arg4',...], ...} 
                                                          containing a list of strings with the Wilson coefficient arguments names
                                                          for each coupling n1, n2,...
                                                          Ex: o4_o6.arguments = {4: ['M', 'r'], 6: ['r', 'q', 'q0']}

		  - coeff_squared_list(list)		- a list containing all the possible combinations of coefficients in the amplitude 
                                                          including interferences
                                                          Ex: o4_o6.coeff_squared_list = [(4, 4), (4, 6), (6, 4), (6, 6)]

		  - global_arguments(array)		- an array containing a list of all the arguments of the wilson coefficients.
                                                          Ex: o4_o6.global_arguments=array(['M', 'q', 'q0', 'r'])


		  - couplings(list)	                 - returns the list of couplings contained in the dictionary
                                                          Ex: o4_o6.couplings = [4, 6]

                                                          
		  - global_default_args(dict)		- a dictionary containing arguments with a default value.
                                                          Ex: o4_o6.global_default_args = {'r': 0, 'q0': 0.01}

		  - hamiltonian(str)			- a string with the hamiltonian.
                                                          Ex: o4_o6.hamiltonian = 'c_4*O_4+c_6*O_6'

		  - name(str)			        - string with the Hamiltonian name.
                                                          Ex: o4_o6.name = 'o4_o6'

		  - wilson_coefficients(dict)		- the dictionary containing the Wilson coefficients passed as an input at instantiation.

      Methods:
		  - coeff_squared(c1,c2,**args):
			Input:
			- c1,c2			- a pair of couplings of the Hamiltonian
                        - **args                - any list of parameter assignements

                        ex: o4_o6.coeff_squared(4,6,M=1e3)
                            array([[10000., 10000.],
                            [10000., 10000.]])
                            N.B. The variable r has default value in c_4 but not in c_6. If r is not passed 
                                 it is set globally to the default value in c_4.
			--------------------------------------------------------
			Output:
			Returns Kronecker product of two given wilson coefficients for given choice of parameter.
			--------------------------------------------------------

		  - coeff_squared_q_dependence(q,c1,c2):
			Input:
			- q			- momentum.
			- c1,c2			- a pair of couplings of the Hamiltonian
			--------------------------------------------------------
			Output:
			Returns the combination c1(q)/c1(q_0_1)*c2(q)/c2(q_0_2) used to define the operators 
                        O_n'(q)=c_n(q)/c+n(q_0_n)*O_n. Used to calculate the response functions.
			--------------------------------------------------------

		  - print_hamiltonian(self):
			Output: 
                        Prints a string with the hamiltonian
			--------------------------------------------------------

		  - print_hamiltonian_latex(self):
    
			Output:
			Prints a string with the hamiltonian in Latex form.
			--------------------------------------------------------    
    '''

    def print_hamiltonian_latex(self):
        '''Prints a string with the hamiltonian in Latex form.'''
        #return '+'.join([r'$c_{'+print_latex_coupling(c)+'}'+str(inspect.signature(self.wilson_coefficients[c]))+'\\times {\\cal O}_{'+print_latex_coupling_short(c)+'}$\n' for c in self.couplings])
        prime=[]
        for c in self.couplings:
            if 'q' in self.arguments[c]:
                prime.append("'(q)")
            else:
                prime.append("")

        return '+'.join([r'$c_{'+print_latex_coupling(c)+'}'+str(inspect.signature(self.wilson_coefficients[c]))+'\\times {\\cal O}_{'+print_latex_coupling(c)+'}$'+pp+'\n' if ws.__name__=='<lambda>' else ws.__name__+str(inspect.signature(self.wilson_coefficients[c]))+'$\\times {\\cal O}_{'+print_latex_coupling(c)+'}$'+pp+'\n' for pp,ws,c in zip(prime,self.wilson_coefficients.values(),self.couplings)])


    
    
    
    def print_hamiltonian(self):
        '''Prints a string with the hamiltonian'''
        prime=[]
        for c in self.couplings:
            if 'q' in self.arguments[c]:
                prime.append("'(q)")
            else:
                prime.append("")
        return '+'.join(['c_'+print_coupling(c)+str(inspect.signature(self.wilson_coefficients[c]))+'* O_'+print_coupling(c)+pp if ws.__name__=='<lambda>' else ws.__name__+str(inspect.signature(self.wilson_coefficients[c]))+'* O_'+print_coupling(c)+pp for pp,ws,c in zip(prime,self.wilson_coefficients.values(),self.couplings)])
        
    
    def __init__(self,name,wilson_coefficients):
        self.name=name

        '''
        path='WimPyDD_models/'+self.name            
        if not os.path.isdir(path):
            p = subprocess.call("mkdir -p "+path, stdout=subprocess.PIPE, shell=True)
        if not os.path.isfile(os.getcwd()+'/WimPyDD_models/__init__.py'):
            open(os.getcwd()+'/WimPyDD_models/__init__.py','x').close()
        if not os.path.isfile(os.getcwd()+'/WimPyDD_models/'+self.name+'/__init__.py'):
            open(os.getcwd()+'/'+path+'/__init__.py','x').close()

        if not wilson_coefficients:
        #exec('import '+path.replace('/','.')+'.wilson_coefficients'+' as wc')
        #self.wc=wc
        '''
        wilson_coefficients={__fix_coupling__(k):v for k,v in wilson_coefficients.items()}

        self.couplings=list(wilson_coefficients.keys())        
        self.wilson_coefficients=wilson_coefficients
        wilson_coefficients_all_spins=[convert_to_all_spins(c) for c in wilson_coefficients]

        self.arguments={k:inspect.getfullargspec(v)[0] for k,v in list(wilson_coefficients.items())}

        self.global_arguments=np.unique(np.array([e for t in self.arguments.values() for e in t]))

        global_default_args={}

        [global_default_args.update(get_default_args(wilson_coefficients[c])) for c in self.couplings]

        if 'mchi' in global_default_args.keys():
            del global_default_args['mchi']
            print('WARNING: the mchi argument is only positional, defalut value in Wilson coefficient ignored') 

        if 'delta' in global_default_args.keys():
            del global_default_args['delta']            
            print('WARNING: the delta argument is only positional, defalut value in Wilson coefficient ignored') 

            
        self.global_default_args={k:v.default for k,v in global_default_args.items()}


        default_args={c:{k:v.default for k,v in get_default_args(wilson_coefficients[c]).items()} for c in self.couplings}

        self.default_args=default_args
        
        
        random_number=np.random.rand()
        args_c_norm={}
        coeff_squared_list=[]
        for __c1__ in list(wilson_coefficients.keys()):

            args_c_norm[__c1__]={k:random_number for k in {e for e in self.arguments[__c1__]}-{'q'}}
            
            for __c2__ in list(wilson_coefficients.keys()):

                c1_all_spins,c2_all_spins=convert_to_all_spins(__c1__),convert_to_all_spins(__c2__)
                
                if c1_all_spins[:3]==c2_all_spins[:3] or symmetric_interference(c1_all_spins,c2_all_spins) or non_symmetric_interference(c1_all_spins,c2_all_spins):

                    coeff_squared_list.append((__c1__,__c2__))

        self.coeff_squared_list=coeff_squared_list


        def coeff_squared_q_dependence(q,__c1__,__c2__):

            for c in self.default_args.keys():
                for arg in self.default_args[c].keys():
                    args_c_norm[c][arg]=self.default_args[c][arg]

            c1_dep=np.array([1.,1.])
            if 'q' in self.arguments[__c1__]:

                if any([s.startswith('q0') for s in self.arguments[__c1__]]):
                    
                    q_norm=self.global_default_args[self.arguments[__c1__][np.where(['q0' in x for x in self.arguments[__c1__]])[0][0]]]
                    c1_dep=np.array(self.wilson_coefficients[__c1__](**args_c_norm[__c1__],q=q))/np.array(self.wilson_coefficients[__c1__](**args_c_norm[__c1__],q=q_norm))
                else:
                    c1_dep=np.array(self.wilson_coefficients[__c1__](**args_c_norm[__c1__],q=q))/np.array(self.wilson_coefficients[__c1__](**args_c_norm[__c1__],q=1.))
                    
            c2_dep=np.array([1.,1.])        
            if 'q' in self.arguments[__c2__]:
                if any([s.startswith('q0') for s in self.arguments[__c2__]]):
                    
                    q_norm=self.global_default_args[self.arguments[__c2__][np.where(['q0' in x for x in self.arguments[__c2__]])[0][0]]]
                    c2_dep=np.array(self.wilson_coefficients[__c2__](**args_c_norm[__c2__],q=q))/np.array(self.wilson_coefficients[__c2__](**args_c_norm[__c2__],q=q_norm))
                else:
                    c2_dep=np.array(self.wilson_coefficients[__c2__](**args_c_norm[__c2__],q=q))/np.array(self.wilson_coefficients[__c2__](**args_c_norm[__c2__],q=1.))
                    
            q_dep=np.array(c1_dep)*np.array(c2_dep)
        
            return q_dep[0]

        self.coeff_squared_q_dependence=coeff_squared_q_dependence

        self.hamiltonian='+'.join(['c_'+print_coupling(c)+'*O_'+print_coupling_short(c) for c in self.couplings])

        
        

        
        def coeff_squared(__c1__,__c2__,**args):

            input_args={}

            input_args.update({k:1 for k in {e for e in self.global_arguments}-{t for t in list(args.keys())} if k not in list(self.global_default_args.keys()) and k!='mchi' and k!='delta'})

            input_args.update({k:v for k,v in self.global_default_args.items() if k not in list(args.keys())})

            input_args.update(args)

            
            flatten=False
            if flatten:
                return np.kron([1,1],[1,1]).reshape(2,2)
            
            global_arguments=np.unique(np.array([e for t in list(self.arguments.values()) for e in t]))

            #args.update({k:1 for k in {e for e in global_arguments}-{t for t in list(args.keys())}})

            #args.update({k:1 for k in {e for e in global_arguments}-{t for t in list(args.keys())}-{l for l in list(self.global_default_args.keys())}})
            input_args.update({k:1 for k in {e for e in global_arguments}-{t for t in list(input_args.keys())}-{l for l in list(self.global_default_args.keys())}})

            

            args1={}
            
            if 'q' in self.arguments[__c1__]:
                if any([s.startswith('q0') for s in self.default_args[__c1__].keys()]):
                    q_norm=self.default_args[__c1__][list(self.default_args[__c1__].keys())[np.where([s.startswith('q0') for s in self.default_args[__c1__].keys()])[0][0]]]
                    args1['q']=q_norm
                else:
                    args1['q']=1.
                    
            for arg in set(self.arguments[__c1__])-{'q'}-{s for s in self.default_args[__c1__].keys() if s.startswith('q0')}:
                args1[arg]=input_args[arg]
            

            args2={}
            
            if 'q' in self.arguments[__c2__]:
                if any([s.startswith('q0') for s in self.default_args[__c2__].keys()]):
                    q_norm=self.default_args[__c2__][list(self.default_args[__c2__].keys())[np.where([s.startswith('q0') for s in self.default_args[__c2__].keys()])[0][0]]]
                    
                    args2['q']=q_norm
                else:
                    args2['q']=1.                
            
            for arg in set(self.arguments[__c2__])-{'q'}-{s for s in self.default_args[__c2__].keys() if s.startswith('q0')}:
                args2[arg]=input_args[arg]
                

            return np.kron(np.array(self.wilson_coefficients[__c1__](**args1)),np.array(self.wilson_coefficients[__c2__](**args2))).reshape(2,2)
        
        self.coeff_squared=coeff_squared

    def __str__(self):

        print('Hamiltonian name:'+self.name)
        
        print('Hamiltonian:'+self.print_hamiltonian())


        prime={}
        for c in self.couplings:
            if 'q' in self.arguments[c]:
                prime[c]="'(q)"
            else:
                prime[c]=""
        
        c_squared_out=[]
        for __c1__,__c2__ in self.coeff_squared_list:
            c_squared_out.append('O_'+print_coupling(__c1__)+prime[__c1__]+'*'+'O_'+print_coupling(__c2__)+prime[__c2__])

            
        return 'Squared amplitude contributions:\n'+', '.join(c_squared_out) 



    
diff_response_functions_documentation='Dictionary containing the differential response functions as defined in ...\nKeys: [model_name,spin,coupling1,coupling2,element]\nOutput: numpy array of functions with shape (2,2,n_isotopes,4).'

        
def load_response_functions(exp,hamiltonian,j_chi=0.5,reset=False,verbose=True,update_time_stamp=False,n_sampling=100):
    '''For the effective Hamiltonian hamiltonian and the WIMP spin j_chi updates the dictionary 
    exp.response_functions of experiment exp with the corresponding tabulated response functions. 


    In particular, for experiment exp, Hamiltonian hamiltonian, and Wimp spin j_chi
    an entry {(hamiltonian,j_chi):r}  to the dictionary exp.response_functions 
    is added by calling load_response_functions(exp, hamiltonian, j_chi).

    For each Wilson coefficients combination contained in hamiltonian.coeff_squared_list
    the tabulated response functions are first searched in a subfolder of the exp experiment. 
    If missing they are calculated, saved in the subfolder and loaded to the dictionary.  

    In the updated dictionary r=exp.response_functions[hamoltonian, j_chi] is a tuple containing 
    the tabulated response functions for all Wilson coefficient combinations, energy bins, 
    nuclear target etc (se help for experiment class). Each call to load_response_functions 
    adds a (hamiltonian,j_chi) entry to the dictionary unless reset=True.

    Setting:           

    n_cicj=0,...,len(hamiltonian.coeff_squared_list)-1 (couplings product)
    n_vel=0,1,2,3 corresponding to a=0,1,1E,1E^-1 (amplitude decomposition.
    see help on eft_amplitude_squared routine)
    tau=0,1        (nuclear isospin)
    tau_prime=0,1  (nuclear isospin)
    n_bin=0,...,len(exp.data)-1 (energy bin)-1
    n_element=0,...,len(exp.target.element)-1 (element in target)
    n_isotope=1,...,len(exp.target.element[n_element]) (isotope of element)

    the array:

    r[n_cicj][n_vel][tau][tau_prime][n_bin][n_element][0]

    contains a sampling of n_sampling (default: 100) recoil energy values in keV

    and the array:

    r[n_cicj][n_vel][tau][tau_prime][n_bin][n_element][n_isotope+1]

    contains the corresponding tabulated integrated response function.
    -----------------------------------------------------------------
    The routine issues a warning and allows to recalculate existing response functions tables if 
    they are older that the input files in the experiment folder (data.tab, resolution.tab, exposure.tab,
    etc). Setting update_time_stamp=True updates the time stamps of all response functions tables 
    to avoid such warning.
    ------------------------------------------------------------------------------------
    Example:
           xenon1t=experiment('XENON1T')
           o4_o6qm2=eft_hamiltonian('o4_o6qm2',{4: func1, (6,'qm2'): func2})
           o4_o6qm2.coeff_squared_list->
           [(4, 4), (4, (6, 'qm2')), ((6, 'qm2'), 4), ((6, 'qm2'), (6, 'qm2'))]
           j_chi=0.5
           
           Subfolder where response functions are saved/read:
           WimPyDD/Experiments/XENON1T/Response_functions/1_2/

            WIMP spin       Spin subfolder name
                0                 0
                0.5               1_2
                1                 1
                1.5               3_2
           etc.

           Response functions tables that are read (if present) or written (if absent):
           c_4_c_4.npy
           c_4_c_6_qm2.npy
           c_4_qm2_c_6.npy
           c_6_qm2_c_6_qm2.npy

           For xenon1t, o4_o6_qm2 and Wimp spin 0.5
           an entry {(04_06qm2,0.5):r}  to the dictionary xenon1t.response_functions 
           is added by calling load_response_functions(xenon1t, o4_o6_qm2, 0.5).

           n_cicj=3 -> ((6, 'qm2'), (6, 'qm2')) (combination of couplings)

           n_vel=1      (contribution of A_1 to the squared amplitude 
                         A=A0+A1*(v^2-vmin^2), a=1 - see help of eft_amplitude_squared )

           tau=1        
           tau_prime=1     (nuclear isospins)

           len(xenon1t.data) -> 1 (1 energy bin defined in data.tab)
           n_bin=0              (only energy bin)
           xenon1t.data[n_bin][:2] -> array([ 3., 70.]) (3 PE<S1<70PE)

           print(xenon1t.target) 
           Xe contains:
           xenon, symbol Xe, atomic number 54, average mass 122.322, 9 isotopes.
           Isotope-averaged mass: 122.32212558999998

           len(dama.target.element) -> 1 (one element in Xe target)
           n_element=0   (xenon) 
           xenon1t.target.element[0].isotopes ->array(['124Xe', '126Xe', '128Xe', '129Xe', '130Xe', '131Xe', '132Xe',
                                                     '134Xe', '136Xe']) (9 isotopes for xenon)
           n_isotope=3  

           E_R=r[n_cicj][n_vel][tau][tau_prime][n_bin][n_element][0]
           R_bar=r[n_cicj][n_vel][tau][tau_prime][n_bin][n_element][n_isotope+1]

           import matplotlib.pyplot as pl
           pl.plot(E_R, R_bar) 

           plots the integrated response function vs. recoil energy for 
           c_6^1*c_6^1/q**4, 
           a=1
           3 PE<S1<70 PE
           Xenon 129 (129Xe) target 

           The routine plot_all_response_functios(exp,hamiltonian,j_chi) can be used to plot
           all the response funtions of a given experiment, effective Hamiltonian and WIMP spin.
    -------------------------------------------------------------------------
    Input:
    
    exp: object belonging to experiment class

    eft_hamiltonian: object belonging to eft_hamiltonian class

    jchi(float): WIMP spin (default value=0.5)

    reset(bool): (default: True) Empties the exp.response_functions dictionary before adding the output.

    verbose(bool): If True prints out a list of the response function tables that are loaded or written, including 
                   the documentation of the experiment object (default: True). 

    update_time_stamp(bool): (default value: True) If true updates the time stamp of all response functions tables.

    n_sampling(int): (default: 100) Number of points of response functions sampling 
    ----------------------------------------------------------------------------
    Output:

    Adds a tuple containing the tabulated response function to the dictionary entry exp.response_functions[hamiltonian,j_chi]
    ----------------------------------------------------------------------------

    '''
    
    if reset:
        exp.response_functions=response_functions({})

    if isinstance(list(hamiltonian.wilson_coefficients.keys())[0],int) and j_chi>1:
        print('NO RESPONSE FUNCTION LOADED')
        print('Hamiltonian "'+hamiltonian.name+'" has NR couplings in the notation of PHYSICAL REVIEW C 89, 065501 (2014).\n For a WIMP spin higher that 1 use hamiltonian with couplings in all spins format (X, l,s).') 
        return

    if isinstance(list(hamiltonian.wilson_coefficients.keys())[0],tuple):
        if isinstance(list(hamiltonian.wilson_coefficients.keys())[0][0],int) and j_chi>1:
            print('NO RESPONSE FUNCTION LOADED')
            print('Hamiltonian "'+hamiltonian.name+'" has NR couplings in the notation of PHYSICAL REVIEW C 89, 065501 (2014).\n For a WIMP spin higher that 1 use hamiltonian with couplings in all spins format (X, l,s).') 
            return
        
    j_chi_out=print_spin(j_chi)
    response_functions_output=()
    diff_response_functions_output={}    
    for __c1__,__c2__ in hamiltonian.coeff_squared_list:

        outputfile='c_'+print_coupling(__c1__)+'_c_'+print_coupling(__c2__)+'.npy'
        if update_time_stamp:
            path=EXPERIMENTS_PATH+'/'+exp.name+'/Response_functions/spin_'+print_spin(j_chi)+'/'
            if os.path.isfile(path+outputfile):
                win_path(path+outputfile).touch()
                #p = subprocess.call("touch "+path+outputfile, stdout=subprocess.PIPE, shell=True)

        
        #c1=convert_to_all_spins(c1)
        #c2=convert_to_all_spins(c2)        

        
        response_functions_output+=(get_response_functions_interf(exp=exp,n_coeff1=get_short_coupling(__c1__),n_coeff2=get_short_coupling(__c2__),eft_modifier=lambda q: hamiltonian.coeff_squared_q_dependence(q,__c1__=__c1__,__c2__=__c2__) ,j_chi=j_chi,verbose=verbose,outputfile=outputfile,n_sampling=n_sampling),)

        for element in exp.target.element:
            diff_response_functions_output[hamiltonian.name,j_chi,__c1__,__c2__,element.name]=element.response_function_interf

    exp.diff_response_functions.update(diff_response_functions(diff_response_functions_output))
    exp.response_functions[hamiltonian,j_chi]=response_functions_output


    



def flip_tau_tau_prime_in_response_functions_interf(exp,response_functions_interf):
    '''Flips tau and tau_prime in exp.response_functions_interf[n_vel][tau][tau_prime][n_bin][n_element][n_isotope+1]'''

    tuple_vel=()
    for n_vel in range(4):

        tuple_tau=()        
        for tau in range(2):

            tuple_tau_prime=()
            for tau_prime in range(2):

                tuple_bin=()
                for n_bin in range(len(exp.data)):
                    
                    tuple_element=()
                    for n_element in range(exp.target.n_targets) :
                        
                        tuple_element+=(response_functions_interf[n_vel][tau_prime][tau][n_bin][n_element],)
                            
                    tuple_bin+=(tuple_element,)

                tuple_tau_prime+=(tuple_bin,)

            tuple_tau+=(tuple_tau_prime,)                
                
        tuple_vel+=(tuple_tau,)

    tuple_vel+=(response_functions_interf[-1],)
            
    return tuple_vel


def set_c_coeff_interf(c,tau):
    if isinstance(c, int):
        if c<1 or c>18:
            print(20*'=')
            print(20*'=')
            string='Coupling '+str(c)+' outside range from 1 to 18. Insert new index (default=1)'
            default=1
            c=delayed_input(10,string,default)
            print()
            print(20*'=')
            print(20*'=')            
            c=1
        coeff=np.zeros(2*18).reshape(18,2)
        coeff[c-1,tau]=1
        return coeff
    elif isinstance(c,tuple):
        coeff={}
        coeff[c]=[1-tau,tau]
    return coeff




def func_w(a,y):
    f=0.
    for i in range(a.shape[len(a.shape)-1]):
        f+=a[...,i]*y**i
        
    return f*np.exp(-2*y)

def default_nuclear_response_functions(element,isotope,q):
    '''                                                                                                     
    Returns the default response functions from from Phys.Rev.C 89 (2014) 6, 065501 (e-Print: 1308.6288[hep-ph])
    and JCAP 1504 (2015) 04, 042 (e-Print: 1501.03729[hep-ph] for one of the 30 isotopes with the itar
    attribute different from zero.

    When defining a custom nuclear reesponse functions for <isotope> the user must add the file
    WimPyDD/Targets/Nuclear_response_functions/<isotope>_func_w.py
    containing a function func with the same behaviour of the attribute func_w, i.e. returning an array with shape (8,2,2)
    with a 2*2 matrix with the four nuclear isotope combinations tau,tau_prime=00,01,10,11
    for each of the 8 nuclear currents 'M', 'Sigma_prime_prime', 'Sigma_prime', 'Phi_prime_prime',
    'Phi_tilde_prime', 'Delta', 'Phi_prime_prime-M', 'Delta-Sigma_prime' contained in the dictionary WD.nuclear_currents.

    The function func must either have the single input argument q with the transferred momentum in Gev:

    def func(q)                                                                                             
    ...                                                                                                     
    return output                                                                                           
                                                                                                            
    or the three arguments element,isotope,q. In the second case the instructions:                          
                                                                                                            
    import WimPyDD_final as WD

    def func(element,isotope,q):                                                                            

    ...
    output=WD.default_nuclear_response_functions(element,isotope,q)                                         

    n=WD.nuclear_current['Sigma_prime_prime']                                                               

    output[n,0,0]=...                                                                                       
    output[n,0,1]=...                                                                                       
    output[n,1,0]=...                                                                                       
    output[n,1,1]=...                                                                                       
                                                                                                            
    return output

    will overwrite some of default definitions of the response functions from this routine without modifying the others.
                                                                                                            
    Notice that in both cases the functions contained in the attribute <element>.func_w will have the single argument q.
    
    The itar code must be written in the last column of the table containing the                            
    element information in WimPyDD/Target/<element>.tab. Its value must be an integer between 1 and 30 with:

    Isotope   itar                                                                                          
    12C        1                                                                                            
    19F        2                                                                                            
    23Na       3                                                                                            
    28Si       4                                                                                            
    70Ge       5                                                                                            
    72Ge       6                                                                                            
    73Ge       7                                                                                            
    74Ge       8                                                                                            
    76Ge       9 
    127I      10                                                                                            
    128Xe     11                                                                                            
    129Xe     12                                                                                            
    130Xe     13                                                                                            
    131Xe     14                                                                                            
    132Xe     15                                                                                            
    134Xe     16                                                                                            
    136Xe     17                                                                                            
    16O       18                                                                                            
    40Ar      19
    40Ca      20                                                                                            
    1H        21                                                                                            
    3He       22                                                                                            
    4He       23                                                                                            
    14N       24                                                                                            
    20Ne      25                                                                                            
    24Mg      26                                                                                            
    27Al      27                                                                                            
    32S       28                                                                                            
    56Fe      29                                                                                            
    58Ni      30
    in order to read the table:                                                                             
    WimPyDD/Targets/Nuclear_response_functions/nuclear_response_functions_coefficients_table.dat            
                                                                                                            
    If an isotope has itar=0 this function returns all vanishing nuclear response functions.                
    '''
    hbarc=197.e-3 #in GeV*fm, q is in GeV                                                                   
    a_nucleus=element.a[isotope]
    b=np.sqrt(41.467e0/(45.e0*a_nucleus**(-1./3.)-25.*a_nucleus**(-2./3.)))
    y=(b*q/(2.e0*hbarc))**2

    a=element.data_mod[isotope]

    f=0.
    for i in range(a.shape[len(a.shape)-1]):
        f+=a[...,i]*y**i

    return f*np.exp(-2*y)
'''
def fix_func_w(func_w_old):
    #######################                                                                                 
    #0: 'M'                                                                                                 
    #1: 'Sigma_prime_prime'                                                                                 
    #2: 'Sigma_prime'                                                                                       
    #3: 'Phi_prime_prime'                                                                                   
    #4: 'Phi_tilde_prime'                                                                                   
    #5: 'Delta'                                                                                             
    #6: 'Phi_prime_prime-M'                                                                                 
    #7: 'Delta-Sigma_prime'                                                                                 
    #######################
    def func_w(q):
'''
#put indetation if we valid it'''Definition from Phys.Rev.C 89 (2014) 6, 065501 (e-Print: 1308.6288[hep-ph]) used\nfor nuclear W functions as default.'''
'''
        w=func_w_old(q)

        w_fixed=np.array(
         [
         np.array([[w[0,0,0],w[0,0,1]],[w[0,0,1],w[0,0,1]**2/w[0,0,0]]]),
         np.array([[w[1,0,0],w[1,0,1]],[w[1,0,1],w[1,0,1]**2/w[1,0,0]]]),
         np.array([[w[2,0,0],w[2,0,1]],[w[2,0,1],w[2,0,1]**2/w[2,0,0]]]),
         np.array([[w[6,0,0]**2/w[0,0,0],w[6,0,0]*w[6,1,0]/w[0,0,0]],
                   [w[6,1,0]*w[6,0,0]/w[0,0,0],w[6,1,0]**2/w[0,0,0]]]),
         np.array([[w[4,0,0],w[4,0,1]],[w[4,0,1],w[4,0,1]**2/w[4,0,0]]]),
         np.array([[w[7,0,0]**2/w[2,0,0],w[7,0,0]*w[7,1,0]/w[2,0,0]],
                   [w[7,1,0]*w[7,0,0]/w[2,0,0],w[7,1,0]**2/w[2,0,0]]]),
         np.array([[w[6,0,0],w[6,0,0]*w[0,0,1]/w[0,0,0]],
                   [w[6,1,0],w[6,1,0]*w[0,0,1]/w[0,0,0]]]),
         np.array([[w[7,0,0],w[7,0,0]*w[2,0,1]/w[2,0,0]],
                   [w[7,1,0],w[7,1,0]*w[2,0,1]/w[2,0,0]]])
         ]
         )
        return np.choose(np.isnan(w_fixed),[w_fixed,0])

    return func_w
'''

index_dict={('M',0):0,('Sigma',-1):1, ('Sigma',0):2,('Sigma',1):3,('Delta',-1):4,('Delta',0):5,('Phi',-1):6,('Phi',0):7,('Phi',1):8,('Omega',0):9}
nuclear_current={"M":0, "Sigma_prime_prime":1, "Sigma_prime":2, "Phi_prime_prime":3, "Phi_tilde_prime":4, 'Delta':5, "Phi_prime_prime-M": 6, "Delta-Sigma_prime":7}
nuclear_response_list=['M','Sigma','Delta','Phi','Omega']
index_dict_inv={v:k for k,v in list(index_dict.items())}

def set_c_coeff_all_spins(coeff_dict):
    coeff_dict_save={}                                                                                     
    s_max=max([k[1] for k in list(coeff_dict.keys())])                                                           
                                                                                                           
                                                                                                           
    c_coeff=np.array(np.zeros((s_max+1,10,2)),dtype=object)                                                                       
                                                                                                           
    for key in list(coeff_dict.keys()):                                                                          
        interaction,s,s_prime=key
        delta_s=s_prime-s 
        if interaction in ['M','Omega']:                                                                   
            if delta_s==0:
                c_coeff[s,index_dict[interaction,delta_s]]=coeff_dict[interaction,s,s_prime]               
                coeff_dict_save[key]=coeff_dict.pop(key)                                                   
                                                                                                           
        elif interaction in ['Sigma','Phi']:                                                               
            if abs(delta_s)<=1:                                                                          
                c_coeff[s,index_dict[interaction,delta_s]]=coeff_dict[interaction,s,s_prime]               
                coeff_dict_save[key]=coeff_dict.pop(key)                                                   
        elif interaction in ['Delta']:
            if delta_s==-1 or delta_s==0:                                                                  
                c_coeff[s,index_dict[interaction,delta_s]]=coeff_dict[interaction,s,s_prime]               
                coeff_dict_save[key]=coeff_dict.pop(key)                                                   

    if coeff_dict:                                                                                         
        for key in list(coeff_dict.keys()):                                                                      
            print('coupling '+str(key)+' ignored')                                                         
            coeff_dict.pop(key)

    for key in list(coeff_dict_save.keys()):                                                                          
        coeff_dict[key]=coeff_dict_save[key]


    s_max=max([k[1] for k in list(coeff_dict.keys())])                                                           

    return c_coeff[:s_max+1]

def b_coeff(j_chi,s):
    if s==0:
        return 1.
    return float(reduce(int.__mul__,range(s,0,-1)))**2/(float(reduce(int.__mul__,range(2*s+1,0,-2)))*float(reduce(int.__mul__,range(2*s-1,0,-2))))*reduce(float.__mul__,j_chi*(j_chi+1)-np.array(range(s))/2.*(np.array(range(s))/2.+1))


def eft_amplitude_squared(coeff1,q,element,isotope,coeff2=None,j_chi=0.5):
    '''
    Calculates WIMP-nucleus scattering square amplitude
    (velocity-independent part)
    ----------------------------------------------------
    Input:

    coeff1: either a (18,2) dimensional array containing the couplings in GeV^-2
    of the non-relativistic ective theory c^tau_n= coeff1(n-1,tau); n=1,3,...15,tau =0,1
    in the notation of Anand et l., PHYSICAL REVIEW C89, 065501 (2014) (arXiv:1308.6288) 
    and Dent et al., Phys.Rev. D92(2015) no.6, 063515 (arXiv:1505.03117). 

    Example: c^1_4 = coeff1(3, 1); c^0_11=coeff1 (10,0)
    or
    a dictionary with c^tau= coeff1([1-tau,tau]) in the notation of P. Gondolo at al., arXiv:2008.05120.
    Example: coeff1={('M',0,0):[0,1], ('Sigma',1,2):[1,0],...}

    q (float): exchanged momentum in GeV 

    element : object belonging to element class

    isotopes(int) : one of the isotopes of the element object

    coeff2: (default=None). If None, coeff1=coeff2. coeff1 and coeff2 can be set to different values 
    to calculate the contribution to the squared amplitude of a specific combination of couplings   

    jchi(float): WIMP spin (default value=0.5)
    ----------------------------------------------------
    Output:
    Squared amplitude A0 in units of GeV^-4 for the squared amplitude decomposition:

       A=A0+A1*(v^2-vmin^2) 

    including a possible dependence on the incoming WIMP speed v.

    Expanding vmin^2 introduced terms with different dependence on the nuclear recoil energy ER. 
    The corresponding energy-integrated response functions must be calculated separately in order to 
    factorize the signal dependence on mchi, delta. If mt is the target mass and mu_chi_t is the WIMP-target 
    reduced mass

      vmin^2=mt/(2\mu_chi_t^2)*E_R+delta^2/(2*mt)*1/E_R+delta/mu_chi_t

    and the squared amplitude:

    A=A_0+A_1*delta/mu_chi_t+A_1*v^2-(A_1*E_R)*mt/(2*mu_chi-t^2)-(A1/E_R)*delta^2/(2*mt)

    depends on the four combinations:

    A_0, A_1, A_1*E_R, E1/E_R 

    conventionally indicated with 
    
    A_a, a=0,1,1E,1E^-1

    The response functions loaded/calculated by the routine load_response_functions 
    are sampled for each of these 4 cases.
    In particular for experiment exp, effective Hamiltonian hamiltonian and WIMP spin j_chi 
    the index a=0,1,1E,1E^-1 is represented by n_vel in the tuple 

      r=exp.response_functions[hamiltonian,j_chi]
      r[n_cicj][n_vel][tau][tau_prime][n_bin][n_element][0]  

      (see help on load_response_functions or on the experiment class for help on the meaning of the other
       indices of the tuple)
    ----------------------------------------------------
    '''
    
    ###====================================    
    if q==0:
        q=1e-10        
    ###====================================
    if coeff2 is None:
        coeff2=coeff1
    ###====================================
    if isinstance(coeff1,np.ndarray) or isinstance(coeff1,list):    

        coeff1=np.array([x(q) if callable(x) else x for x in coeff1])
        coeff2=np.array([x(q) if callable(x) else x for x in coeff2])                
        return eft_amplitude_squared_max_spin_1(coeff1,q,element,isotope,coeff2,j_chi=j_chi)


    elif isinstance(coeff1,dict):

        coeff1={k: (v(q) if callable(v) else v) for k,v in coeff1.items()}
        coeff2={k: (v(q) if callable(v) else v) for k,v in coeff2.items()}        
        
        coeff1_all_spins=set_c_coeff_all_spins(coeff1)
        coeff2_all_spins=set_c_coeff_all_spins(coeff2)            
        eft_output=0.
        s_max=max(len(coeff1_all_spins),len(coeff2_all_spins))
        for coeff1,coeff2,s in zip(coeff1_all_spins,coeff2_all_spins,range(s_max+1)):
            eft_output+=eft_amplitude_squared_all_spins(coeff1,q,element,isotope,coeff2,j_chi=j_chi,s=s)
        return eft_output

def eft_amplitude_squared1(coeff1,q,element,isotope,coeff2=None,j_chi=0.5):
    '''
    Calculates WIMP-nucleus scattering square amplitude
    (velocity-dependent part)
    ----------------------------------------------------
    Input:

    coeff1: either a (18,2) dimensional array containing the couplings in GeV^-2
    of the non-relativistic ective theory c^tau_n= coeff1(n-1,tau); n=1,3,...15,tau =0,1
    in the notation of Anand et l., PHYSICAL REVIEW C89, 065501 (2014) (arXiv:1308.6288) 
    and Dent et al., Phys.Rev. D92(2015) no.6, 063515 (arXiv:1505.03117). 
    Example: c^1_4 = coeff1(3, 1); c^0_11=coeff1 (10,0)
    or
    a dictionary with c^tau= coeff1([1-tau,tau]) in the notation of P. Gondolo at al., arXiv:2008.05120.
    Example: coeff1={('M',0,0):[0,1], ('Sigma',1,2):[1,0],...}

    q (float): exchanged momentum in GeV 

    element : object belonging to element class

    isotopes(int) : one of the isotopes of the element object

    coeff2: (default=None). If None, coeff1=coeff2. coeff1 and coeff2 can be set to different values 
    to calculate the contribution to the squared amplitude of a specific combination of couplings   

    jchi(float): WIMP spin (default value=0.5)
    ----------------------------------------------------
    Output:
    Squared amplitude A1 in units of GeV^-4 for the squared amplitude decomposion:

       A=A0+A1*(v^2-vmin^2) 

    including a possible dependence on the incoming WIMP speed v.
    ----------------------------------------------------
    ----------------------------------------------------
    '''
    
    ###====================================    
    if q==0:
        q=1e-10
    ###====================================
    if coeff2 is None:
        coeff2=coeff1
    ###====================================
    if isinstance(coeff1,np.ndarray) or isinstance(coeff1,list):

        coeff1=np.array([x(q) if callable(x) else x for x in coeff1])
        coeff2=np.array([x(q) if callable(x) else x for x in coeff2])        
        return eft_amplitude_squared1_max_spin_1(coeff1,q,element,isotope,coeff2,j_chi=j_chi)
    
    elif isinstance(coeff1,dict):

        coeff1={k: (v(q) if callable(v) else v) for k,v in coeff1.items()}
        coeff2={k: (v(q) if callable(v) else v) for k,v in coeff2.items()}        
        
        coeff1_all_spins=set_c_coeff_all_spins(coeff1)
        coeff2_all_spins=set_c_coeff_all_spins(coeff2)        
        eft1_output=0.
        s_max=max(len(coeff1_all_spins),len(coeff2_all_spins))
        for coeff1,coeff2,s in zip(coeff1_all_spins,coeff2_all_spins,range(s_max+1)):
            eft1_output+=eft_amplitude_squared1_all_spins(coeff1,q,element,isotope,coeff2,j_chi=j_chi,s=s)
        return eft1_output

    
def eft_amplitude_squared_all_spins(coeff1,q,element,isotope,coeff2=None,j_chi=0,s=0):
    ###   -------------------------------------------------------------
    ###   i_tar : target
    ###       01)C 02)19F  03)23Na 04)28Si  05)70Ge   06)72Ge   07)73Ge   08)74Ge   09)76Ge   10)127I
    ###       11)128Xe 12)129Xe  13)130Xe  14)131Xe  15)132Xe  16)134Xe  17)136Xe 18)16O 19)40Ar 20) 40Ca
    ###       21) H 22)3He  23)3He 24)14N 25)20Ne 26)24Mg 27)27Al 28)32S 29)56Fe 30)59Ni
    ###====================================================================
    if coeff2 is None:
        coeff2=coeff1
    ###=====================================================================
    ### coeff1 and coeff2 can contain either floats or functions of 
    #coeff1=np.array([[[z(q) if callable(z) else z for z in y] for y in x] for x in coeff1])
    #coeff2=np.array([[[z(q) if callable(z) else z for z in y] for y in x] for x in coeff2])
    ###====================================================================    
    b_j_chi_s=b_coeff(j_chi,s)
    ###====================================    
    c_light=3.e5 #km/sec
    ###==========================================
    ###  nuclear spin
    j_nucleus=element.spin[isotope]
    ###====================================
    ### see Eq. (40) of 1308.6288
    ###====================================
    hbarc=197.e-3 #GeV*fm, q is supposed to be in GeV
    m_n=0.931e0 #nucleon mass, GeV
    q_tilde=q/m_n
    m_nucleus=element.mass[isotope] #GeV
    a_nucleus=element.a[isotope]
    ###========================================
    m_n=0.931e0 # nucleon mass, GeV
    ###==========================================
    c_light=3.e5 #km/sec
    ###====================================
    b=np.sqrt(41.467e0/(45.e0*a_nucleus**(-1./3.)-25.*a_nucleus**(-2./3.)))
        
    y=(b*q/(2.e0*hbarc))**2
    
    ###====================================
    amplitude=np.array([])

    ###       01)M     
    amplitude=np.append(amplitude,
    np.kron(coeff1[0],coeff2[0])*q_tilde**(2*s)
    )
    amplitude=amplitude.reshape(-1,2,2)

                        
    ###C    02)Sigma''
    
    amplitude=np.append(amplitude,
    q_tilde**(2*s-2)/4.*
    np.kron(coeff1[1]-coeff1[3]*q_tilde**2,coeff2[1]-coeff2[3]*q_tilde**2)
    )
    amplitude=amplitude.reshape(-1,2,2)
                        
    ###      03)Sigma'
    if s>0:
        amplitude=np.append(amplitude,
        (s+1.)/(8.*s)*q_tilde**(2*s-2)*(np.kron(coeff1[1],coeff2[1])
        +np.kron(coeff1[2],coeff2[2])*q_tilde**2)
        )
    else:
        amplitude=np.append(amplitude,np.zeros(4))

    amplitude=amplitude.reshape(-1,2,2)
                        
    ###     04)Phi''
    amplitude=np.append(amplitude,
    q_tilde**2*q_tilde**(2*s-2)/4.*np.kron(coeff1[6]-coeff1[8]*q_tilde**2,coeff2[6]-coeff2[8]*q_tilde**2)
    )
                        
    amplitude=amplitude.reshape(-1,2,2)
                        
    ###      05)Phi'
    if s>0:    
        amplitude=np.append(amplitude,
        q_tilde**2*(s+1.)/(8.*s)*q_tilde**(2*s-2)*(np.kron(coeff1[6],coeff2[6])+np.kron(coeff1[7],coeff2[7])*q_tilde**2)
        )
    else:
        amplitude=np.append(amplitude,np.zeros(4))
        
    amplitude=amplitude.reshape(-1,2,2)
                        
    ###      06)Delta
    if s>0:    
        amplitude=np.append(amplitude,
        q_tilde**2*(s+1.)/(2.*s)*q_tilde**(2*s-2)*(np.kron(coeff1[4],coeff2[4])+np.kron(coeff1[5],coeff2[5])*q_tilde**2)
        )
    else:
        amplitude=np.append(amplitude,np.zeros(4))
    
    amplitude=amplitude.reshape(-1,2,2)
    
#######      07Phi''-M

    amplitude=np.append(amplitude,
    q_tilde**2*q_tilde**(2*s-2)*(np.kron(coeff1[6]-coeff1[8]*q_tilde**2,coeff2[0]))
    )
    amplitude=amplitude.reshape(-1,2,2)    
    
#######       08)Delta-Sigma'

    if s>0:
        amplitude=np.append(amplitude,
        q_tilde**2*(s+1.)/(2.*s)*q_tilde**(2*s-2)*(-np.kron(coeff1[5],coeff2[1])-np.kron(coeff1[4],coeff2[2]))
        )

    else:
        amplitude=np.append(amplitude,np.zeros(4))
        
    amplitude=amplitude.reshape(-1,2,2)    

    #return b_j_chi_s*4.e0*np.pi/(2.e0*j_nucleus+1.e0)*\
    #np.sum(amplitude*func_w(element.data_mod[isotope],y))


    return b_j_chi_s*4.e0*np.pi/(2.e0*j_nucleus+1.e0)*\
    np.sum(amplitude*element.func_w[isotope](q))


    eft_amplitude_squared.amplitude=4.e0*np.pi/(2.e0*j_nucleus+1.e0)*amplitude



def eft_amplitude_squared1_all_spins(coeff1,q,element,isotope,coeff2=None,j_chi=0,s=0):          
    ###   -------------------------------------------------------------
    ###   i_tar : target
    ###       01)C 02)19F  03)23Na 04)28Si  05)70Ge   06)72Ge   07)73Ge   08)74Ge   09)76Ge   10)127I
    ###       11)128Xe 12)129Xe  13)130Xe  14)131Xe  15)132Xe  16)134Xe  17)136Xe 18)16O 19)40Ar 20) 40Ca
    ###       21) H 22)3He  23)3He 24)14N 25)20Ne 26)24Mg 27)27Al 28)32S 29)56Fe 30)59Ni   
    ###====================================
    b_j_chi_s=b_coeff(j_chi,s)
    ###====================================                            
    if coeff2 is None:
        coeff2=coeff1
    ###====================================
    c_light=3.e5 #km/sec
    ###==========================================
    ###  nuclear spin
    j_nucleus=element.spin[isotope]
    ###====================================
    ### see Eq. (40) of 1308.6288
    ###====================================
    hbarc=197.e-3 #GeV*fm, q is supposed to be in GeV
    m_n=0.931e0 #nucleon mass, GeV
    q_tilde=q/m_n
    m_nucleus=element.mass[isotope] #GeV
    a_nucleus=element.a[isotope]
    ###========================================
    m_n=0.931e0 # nucleon mass, GeV
    ###==========================================
    c_light=3.e5 #km/sec
    ###====================================
    b=np.sqrt(41.467e0/(45.e0*a_nucleus**(-1./3.)-25.*a_nucleus**(-2./3.)))
        
    y=(b*q/(2.e0*hbarc))**2
    
    ###====================================
    amplitude=np.array([])

    ###       01)M
    if s>0:    
        amplitude=np.append(amplitude,
        (s+1.)/(2.*s)*q_tilde**(2*s-2)*(np.kron(coeff1[4],coeff2[4])+np.kron(coeff1[5],coeff2[5])*q_tilde**2)    
        )
    else:
        amplitude=np.append(amplitude,np.zeros(4))
                        

    amplitude=amplitude.reshape(-1,2,2)

                        
    ###C    02)Sigma''

    if s>0:
        amplitude=np.append(amplitude,
        (s+1.)/(8.*s)*q_tilde**(2*s-2)*(np.kron(coeff1[6],coeff2[6])+np.kron(coeff1[7],coeff2[7])*q_tilde**2)
        )
    else:
        amplitude=np.append(amplitude,np.zeros(4))
        
    amplitude=amplitude.reshape(-1,2,2)
                        
    ###      03)Sigma'   
    amplitude=np.append(amplitude,
    1./8.*q_tilde**(2*s-2)*np.kron((coeff1[6]-coeff1[8]*q_tilde**2),(coeff2[6]-coeff2[8]*q_tilde**2))
    +q_tilde**(2*s)/8.*np.kron(coeff1[9],coeff2[9])                        
    )

    amplitude=amplitude.reshape(-1,2,2)
                        
    ###     04)Phi''
    amplitude=np.append(amplitude,np.zeros(4).reshape(2,2))
    amplitude=amplitude.reshape(-1,2,2)
                        
    ###      05)Phi'
    amplitude=np.append(amplitude,np.zeros(4).reshape(2,2))
    amplitude=amplitude.reshape(-1,2,2)
                        
    ###      06)Delta
    amplitude=np.append(amplitude,np.zeros(4).reshape(2,2))
    amplitude=amplitude.reshape(-1,2,2)
                        
###      07)M-Phi''
    amplitude=np.append(amplitude,np.zeros(4).reshape(2,2))
    amplitude=amplitude.reshape(-1,2,2)
                        
###       08)Sigma'-Delta
    amplitude=np.append(amplitude,np.zeros(4).reshape(2,2))                        
    amplitude=amplitude.reshape(-1,2,2)    

    
    #return b_j_chi_s*4.e0*np.pi/(2.e0*j_nucleus+1.e0)*\
    #np.sum(amplitude*element.func_w[isotope](q))

    return b_j_chi_s*4.e0*np.pi/(2.e0*j_nucleus+1.e0)*\
    np.sum(amplitude*element.func_w[isotope](q))

    

def eft_amplitude_squared_max_spin_1(coeff1,q,element,isotope,coeff2=None,j_chi=0.5):          
    ###   -------------------------------------------------------------
    ###   i_tar : target
    ###       01)C 02)19F  03)23Na 04)28Si  05)70Ge   06)72Ge   07)73Ge   08)74Ge   09)76Ge   10)127I
    ###       11)128Xe 12)129Xe  13)130Xe  14)131Xe  15)132Xe  16)134Xe  17)136Xe 18)16O 19)40Ar 20) 40Ca
    ###       21) H 22)3He  23)3He 24)14N 25)20Ne 26)24Mg 27)27Al 28)32S 29)56Fe 30)59Ni       
    ###====================================
    if coeff2 is None:
        coeff2=coeff1
    ###====================================
    ### use mask to put to zero missing couplings according to j_chi
    coeff1=(coeff1.T*mask_spin(j_chi)).T
    coeff2=(coeff2.T*mask_spin(j_chi)).T    
    ###====================================    
    c_light=3.e5 #km/sec
    ###==========================================
    ###  nuclear spin
    j_nucleus=element.spin[isotope]
    ###====================================
    ### see Eq. (40) of 1308.6288
    ###====================================
    hbarc=197.e-3 #GeV*fm, q is supposed to be in GeV
    m_n=0.931e0 #nucleon mass, GeV
    m_nucleus=element.mass[isotope] #GeV
    a_nucleus=element.a[isotope]
    ###========================================
    m_n=0.931e0 # nucleon mass, GeV
    ###==========================================
    c_light=3.e5 #km/sec
    ###====================================
    b=np.sqrt(41.467e0/(45.e0*a_nucleus**(-1./3.)-25.*a_nucleus**(-2./3.)))
        
    y=(b*q/(2.e0*hbarc))**2
    
    ###====================================
    amplitude=np.array([])

    ###       01)M     
    amplitude=np.append(amplitude,
    np.kron(coeff1[0],coeff2[0])+j_chi*(j_chi+1.e0)/3.e0*
    (q**2/m_n**2*np.kron(coeff1[10],coeff2[10])))

    amplitude=amplitude.reshape(-1,2,2)

                        
    ###C    02)Sigma''

    amplitude=np.append(amplitude,
    q**2/(4.e0*m_n**2)*\
    np.kron(coeff1[9],coeff2[9])+j_chi*(j_chi+1.e0)/12.e0*\
    (np.kron(coeff1[3],coeff2[3])+q**2/m_n**2*\
    (np.kron(coeff1[3],coeff2[5])+np.kron(coeff1[5],coeff2[3]))\
     +q**4/m_n**4*np.kron(coeff1[5],coeff2[5])+q**2/(2.*m_n**2)*np.kron(coeff1[17],coeff2[17])))
    amplitude=amplitude.reshape(-1,2,2)
                        
    ###      03)Sigma'   
    amplitude=np.append(amplitude,
    j_chi*(j_chi+1.e0)/12.e0*
    (
    np.kron(coeff1[3],coeff2[3])+
    q**2/m_n**2*np.kron(coeff1[8],coeff2[8])+q**2/(4.*m_n**2)*np.kron(coeff1[17],coeff2[17])
    )        
    )

    amplitude=amplitude.reshape(-1,2,2)
                        
    ###     04)Phi''
    amplitude=np.append(amplitude,
    q**2/m_n**2*(q**2/(4.*m_n**2)*np.kron(coeff1[2],coeff2[2])+
    j_chi*(j_chi+1.e0)/12.e0*
    np.kron(coeff1[11]-q**2/m_n**2*coeff1[14],coeff2[11]-q**2/m_n**2*coeff2[14])
    )
    )
                        
    amplitude=amplitude.reshape(-1,2,2)
                        
    ###      05)Phi'
    amplitude=np.append(amplitude,
    q**2/m_n**2*j_chi*(j_chi+1.e0)/12.e0*
    (np.kron(coeff1[11],coeff2[11])+
    q**2/m_n**2*np.kron(coeff1[12],coeff2[12]))
    )
    amplitude=amplitude.reshape(-1,2,2)
                        
    ###      06)Delta
    amplitude=np.append(amplitude,                        
    q**2/m_n**2*(j_chi*(j_chi+1.e0)/3.e0*(
    q**2/m_n**2*np.kron(coeff1[4],coeff2[4])+
        np.kron(coeff1[7],coeff2[7])+q**2/(4.*m_n**2)*np.kron(coeff1[16],coeff2[16]))        
    )
    )
    amplitude=amplitude.reshape(-1,2,2)
                        
#######      07Phi''-M
    amplitude=np.append(amplitude,         
    q**2/m_n**2*(
    np.kron(coeff1[2],coeff2[0])
    +
    j_chi*(j_chi+1.e0)/3.e0*
        np.kron(coeff1[11]-q**2/m_n**2*coeff1[14],coeff2[10])))
    

    amplitude=amplitude.reshape(-1,2,2)    
                        
#######       08)Delta-Sigma'
    amplitude=np.append(amplitude,                   
    q**2/m_n**2*j_chi*(j_chi+1.e0)/3.e0*(
        np.kron(coeff1[4],coeff2[3])
        - np.kron(coeff1[7],coeff2[8])))
        
    amplitude=amplitude.reshape(-1,2,2)    


    #return 4.e0*np.pi/(2.e0*j_nucleus+1.e0)*\
    #np.sum(amplitude*func_w(element.data_mod[isotope],y))

    eft_amplitude_squared.amplitude=4.e0*np.pi/(2.e0*j_nucleus+1.e0)*amplitude

    #eft_vec=np.array([np.sum(4.e0*np.pi/(2.e0*j_nucleus+1.e0)*amplitude[([inter[int] for int in interference],)]*element.func_w[isotope](q)[([inter[int] for int in interference],)]) for interference in [interference1,interference2,no_interference]])


    #return np.sum(eft_vec[np.where(eft_vec>0)])
    return 4.e0*np.pi/(2.e0*j_nucleus+1.e0)*\
    np.sum(amplitude*element.func_w[isotope](q))
     


def eft_amplitude_squared1_max_spin_1(coeff1,q,element,isotope,coeff2=None,j_chi=0.5):          
    ###   -------------------------------------------------------------
    ###   i_tar : target
    ###       01)C 02)19F  03)23Na 04)28Si  05)70Ge   06)72Ge   07)73Ge   08)74Ge   09)76Ge   10)127I
    ###       11)128Xe 12)129Xe  13)130Xe  14)131Xe  15)132Xe  16)134Xe  17)136Xe 18)16O 19)40Ar 20) 40Ca
    ###       21) H 22)3He  23)3He 24)14N 25)20Ne 26)24Mg 27)27Al 28)32S 29)56Fe 30)59Ni   
    ###====================================
    if coeff2 is None:
        coeff2=coeff1
    ###====================================
    ### use mask to put to zero missing couplings according to j_chi
    coeff1=(coeff1.T*mask_spin(j_chi)).T
    coeff2=(coeff2.T*mask_spin(j_chi)).T            
    ###====================================
    c_light=3.e5 #km/sec
    ###==========================================
    ###  nuclear spin
    j_nucleus=element.spin[isotope]
    ###====================================
    ### see Eq. (40) of 1308.6288
    ###====================================
    hbarc=197.e-3 #GeV*fm, q is supposed to be in GeV
    m_n=0.931e0 #nucleon mass, GeV
    m_nucleus=element.mass[isotope] #GeV
    a_nucleus=element.a[isotope]
    ###========================================
    m_n=0.931e0 # nucleon mass, GeV
    ###==========================================
    c_light=3.e5 #km/sec
    ###====================================
    b=np.sqrt(41.467e0/(45.e0*a_nucleus**(-1./3.)-25.*a_nucleus**(-2./3.)))
        
    y=(b*q/(2.e0*hbarc))**2
    
    ###====================================
    amplitude=np.array([])

    ###       01)M     
    amplitude=np.append(amplitude,
        j_chi*(j_chi+1.)/3.*(q**2/m_n**2*np.kron(coeff1[4],coeff2[4])+np.kron(coeff1[7],coeff2[7])+
        q**2/(4.*m_n**2)*np.kron(coeff1[16],coeff2[16])))
                        

    amplitude=amplitude.reshape(-1,2,2)

                        
    ###C    02)Sigma''

    amplitude=np.append(amplitude,
    j_chi*(j_chi+1.)/12.*(
    np.kron(coeff1[11],coeff2[11])+q**2/m_n**2*np.kron(coeff1[12],coeff2[12])
    ))
    amplitude=amplitude.reshape(-1,2,2)
                        
    ###      03)Sigma'   
    amplitude=np.append(amplitude,
    1./8.*(
    q**2/m_n**2*np.kron(coeff1[2],coeff2[2])+np.kron(coeff1[6],coeff2[6])
    )+
    j_chi*(j_chi+1.)/12.*
    (
    1./2.*        
    np.kron(coeff1[11]-q**2/m_n**2*coeff1[14],coeff2[11]-q**2/m_n**2*coeff2[14])
    +q**2/(2.*m_n**2)*np.kron(coeff1[13],coeff2[13])
    )
    )

    amplitude=amplitude.reshape(-1,2,2)
                        
    ###     04)Phi''
    amplitude=np.append(amplitude,np.zeros(4).reshape(2,2))
    amplitude=amplitude.reshape(-1,2,2)
                        
    ###      05)Phi'
    amplitude=np.append(amplitude,np.zeros(4).reshape(2,2))
    amplitude=amplitude.reshape(-1,2,2)
                        
    ###      06)Delta
    amplitude=np.append(amplitude,np.zeros(4).reshape(2,2))
    amplitude=amplitude.reshape(-1,2,2)
                        
###      07)M-Phi''
    amplitude=np.append(amplitude,np.zeros(4).reshape(2,2))
    amplitude=amplitude.reshape(-1,2,2)
                        
###       08)Sigma'-Delta
    amplitude=np.append(amplitude,np.zeros(4).reshape(2,2))                        
    amplitude=amplitude.reshape(-1,2,2)    


    #return 4.e0*np.pi/(2.e0*j_nucleus+1.e0)*\
    #np.sum(amplitude*func_w(element.data_mod[isotope],y))

    
    return 4.e0*np.pi/(2.e0*j_nucleus+1.e0)*\
    np.sum(amplitude*element.func_w[isotope](q))




def eft_amplitude_squared1_em1(coeff1,q,element,isotope,coeff2=None,j_chi=0.5):
    ###====================================
    if coeff2 is None:
        coeff2=coeff1
    ###====================================    
    m_nucleus=element.mass[isotope]
    # q and er are both in GeV    
    er=q**2/(2.*m_nucleus)
    return eft_amplitude_squared1(coeff1,q,element,isotope,coeff2,j_chi=j_chi)/er


def eft_amplitude_squared1_e(coeff1,q,element,isotope,coeff2=None,j_chi=0.5):         
    ###====================================
    if coeff2 is None:
        coeff2=coeff1
    ###====================================    
    m_nucleus=element.mass[isotope]
    # q and er are both in GeV
    er=q**2/(2.*m_nucleus) 
    return eft_amplitude_squared1(coeff1,q,element,isotope,coeff2,j_chi=j_chi)*er



def def_response_function(coeff1,element,isotope,efficiency,quenching,resolution,exposure=1,eft=eft_amplitude_squared,coeff2=None):
    if coeff2 is None:
        coeff2=coeff1

    if np.size(exposure)>1:
        exposure=1

    
    def response_function(er,eprime):

        a_nucleus=element.a[isotope]
        m_nucleus=a_nucleus*0.931
        q=np.sqrt(2.*m_nucleus*er*1e-6)# in GeV
        nt=element.nt_kg[isotope]

        
        return 10**(-6)*exposure*nt*m_nucleus/2.*efficiency(eprime)*resolution(eprime,quenching(er)*er)*\
        eft(coeff1,q,element,isotope,coeff2)
                                   
    return response_function



def def_response_function_nores(coeff1,element,isotope,efficiency,quenching,exposure=1,eft=eft_amplitude_squared,coeff2=None):
    if coeff2 is None:
        coeff2=coeff1

    if np.size(exposure)>1:
        exposure=1

    
    def response_function(er,eprime):

        a_nucleus=element.a[isotope]
        m_nucleus=a_nucleus*0.931
        q=np.sqrt(2.*m_nucleus*er*1e-6)# in GeV
        nt=element.nt_kg[isotope]

        
        return 10**(-6)*exposure*nt*m_nucleus/2.*efficiency(quenching(er)*er)*\
        eft(coeff1,q,element,isotope,coeff2)
    
    return response_function


    
def convert_events_to_data(event_file,data_file):

    if not os.path.isfile(event_file):                                                                                   
        return                                                                                                          

                                                                                                                         
    if os.path.isfile(event_file) and os.path.isfile(data_file):                                                         
        if os.stat(data_file).st_mtime>os.stat(event_file).st_mtime:                                                     
            return                                                                                                      

                                                                                                                         
    tuple=read_tuple_from_file(event_file)                                                                               
                                                                                                                         
    string=tuple[1]                                                                                                      
                                                                                                                         
    events=np.array([])                                                                                                  
    for n in range(len(tuple[0])):                                                                                       
        if len(tuple[0][n])==3:                                                                                          
            emin=tuple[0][n][0]                                                                                          
            emax=tuple[0][n][1]                                                                                          
            nbin=tuple[0][n][2]                                                                                          
        elif len(tuple[0][n])==1:                                                                                        
            events=np.append(events,tuple[0][n])                                                                         
                                                                                                                         


    events=np.sort(events)        
            
                                                                                                                         
    try:                                                                                                                 
                                                                                                                         
        bins=np.linspace(emin,emax,nbin+1)                                                                                 
        histogram=pl.hist(events,bins)                                                                                   
        output_table=np.transpose(np.append(np.append(bins[:-1],bins[1:]),histogram[0]).reshape(3,-1))                   
                                                                                                                         
        output_string=string+' Converted from events.tab to '+str(nbin)+' bins in the range '+str(emin)+'<energy<'+str(emax)
    
        if events.shape[0]>=2:                                                                                           
            with open(data_file,'wd') as f:                                                                              
                np.savetxt(data_file,output_table,delimiter="  ",fmt="%s", header=output_string)                         
                                                                                                                         
    except:                                                                                                                  

        output_string=string+' Converted from events.tab file for use in Optimal Interval method.'
        output_table=np.transpose(np.append(events[:-1],events[1:]).reshape(2,-1))
                                                                                                                         
        if output_table.shape[0]>=2:                                                                                     
            with open(data_file,'w') as f:                                                                              
                np.savetxt(data_file,output_table,delimiter="  ",fmt="%s", header=output_string)                         



                
class experiment(target):
    '''
        Instantiates an experiment object containing the experimental information required to calculate 
        response functions: nuclear targets, energy bins, energy resolution, efficiency, quenching factors 
        for each target, exposure. All such experimental information is collected from a 
        subfolder of the Experiments directory in the working folder that can contains any of the 
        following user-provided files:

           target.tab
           resolution.py
           efficiency.tab or efficiency.py
           data.tab
           exposure.tab
           xenon_quenching.tab or xenon_quenching.py
           modifier.py           

        Example:
           xenon1t=experiment('XENON1T')

        Subfolder:
           WimPyDD/Experiments/XENON1T

        With the exception of target.tab the first line of .tab input files can contain documentation info that is 
        shown by issuing print(xenon1t) and is stored in the exp.response_functions tuple when load_response_functions 
        calculates the response functions. The same happens for the documentation of the functions defined in the .py 
        input files.
	----------------------------------------------------------------------------------------------------------------
	Parameters: 
		- name(str)		        - name of experiment. Corresponds to the name of the subfolder in
                                                  the Experiments folder containing the files with the experimental 
                                                  input. (Ex: if name='XENON1T' the input folder is set to 
                                                          WimPyDD/Experiments/XENON1T)

		- target(target obj)		- (default value: None). Object instantiated with the target class. 
                                                  If None the target object is instantiated with the target class 
                                                  using as input the string read from target.tab   
                                                  
                                                  Examples of acceptable target.tab contents:
                                                  echo 'NaI' > target.tab
                                                  echo 'Xe'  > target.tab
                                                  echo 'C3F8' > target.tab

		- verbose(bool)		- (default:False) If True prints out the details of the loading procedure 
                                          from the input directory.
	 ----------------------------------------------------------------------------------------------------------------
	 Attributes:


		- exposure(float)		-  (default value:1) Exposure of experiment in kg day. Loaded from the 
                                                   file exposure.tab, if present. exposure.tab can contain a single float 
                                                   or a list of floats (each for every energy bin contained in exp.data.
 
                                                   Examples:
                                                   12.5 > exposure.tab
                                                   [12.5,23.5,14] > exposure.tab (in this case len(exp.data)=3)


		- name(str)		       -  experiment name.

		- target(class)		       -  experimental target object (see help on target class).

		- data(tuple):                 - (default: (array([1.]), if data.tab not present). 
                                                 A tuple of arrays, each array containing one line of the file data.tab 
                                                 Each line in data.tab must represent a bin in the visible 
                                                 energy e_prime.

                                                 Example of data.tab content, with the interpretation for the calculation 
                                                 of the response functions:
 
                                                 Line content    |            WimPyDD interpretation                     |
                                                 -------------------------------------------------------------------------
                                                 10              |    e_prime=10 keVee (no binning -> differential rate) |
                                                 10    20        |    10 keV<e_prime<20 keV                              |

                                                 Each line can contain the observed count rate for each energy bin
                                                 used by the routine mchi_vs_exclusion to calculate exclusion plots.
                                                 Indicating with N_exp the observed rate and with sigma_exp the 
                                                 fluctuation:

                                                 Line content    |            WimPyDD interpretation                     |
                                                 -------------------------------------------------------------------------
                                                 10 20 2         |10 keV<e_prime<20 keV, N_exp=2  (Poisson fluctuation)  | 
                                                 10 20 3.2  0.4  |    10 keV<e_prime<20 keV, N_exp=3.2 sigma_exp=0.4     |



		- response_functions(dict)	- a dictionary containing the tabulated response functions.
                                                  For experiment exp, Hamiltonian hamiltonian and Wimp spin j_chi
                                                  nn entry {(hamiltonian,j_chi):r}  to the dictionary exp.response_functions 
                                                  is added by calling load_response_functions(exp, hamiltonian, j_chi).
                                                  ------------------------------------------------------------------------
                                                  Setting:           

                                                  n_cicj=0,...,len(hamiltonian.coeff_squared_list)-1 (couplings product)
                                                  n_vel=0,1,2,3 corresponding to a=0,1,1E,1E^-1 (amplitude decomposition.
                                                  see help on eft_amplitude_squared routine)
                                                  tau=0,1        (nuclear isospin)
                                                  tau_prime=0,1  (nuclear isospin)
                                                  n_bin=0,...,len(exp.data)-1 (energy bin)-1
                                                  n_element=0,...,len(exp.target.element)-1 (element in target)
                                                  n_isotope=1,...,len(exp.target.element[n_element]) (isotope of element)

                                                  the array:

                                                  r[n_cicj][n_vel][tau][tau_prime][n_bin][n_element][0]

                                                  contains a sampling of recoil energy values in keV

                                                  and the array:

                                                  r[n_cicj][n_vel][tau][tau_prime][n_bin][n_element][n_isotope+1]
 
                                                  contains the corresponding tabulated integrated response function.
                                                  -----------------------------------------------------------------

                                                  Example: 

                                                  o4_o6=eft_hamiltonian('o4_o4',{4: func1, 6: func2})
                                                  o4_o6.coeff_squared_list-> [(4, 4), (4, 6), (6, 4), (6, 6)]
                                                 
                                                  n_cicj=2 -> (6,4) (combination of couplings)

                                                  n_vel=2      (contribution of A_1*E_R to the squared amplitude 
                                                                A=A0+A1*(v^2-vmin^2), a=1E )
 
                                                  tau=0         
                                                  tau_prime=1     (nuclear isospins)

                                                  dama=experiment('dama')

                                                  len(dama.data) -> 12 (12 energy bins defined in data.tab)
                                                  n_bin=3              (4th energy bin)
                                                  dama.data[n_bin][:2] -> array([3.5, 4. ]) (3.5 keV<E_prime<4 keV)

                                                  print(dama_2018.target)
                                                  NaI contains:
                                                  sodium, symbol Na, atomic number 11, average mass 21.413, 1 isotopes.
                                                  iodine, symbol I, atomic number 53, average mass 118.237, 1 isotopes.
                                                  Isotope-averaged mass: 139.65
                                                  
                                                  len(dama.target.element) -> 2 (two elements in NaI target)
                                                  n_element=1   (iodine) 
                                                  dama.target.element[1].isotopes -> array(['127I'] (one isotope for iodine)
                                                  n_isotope=0

                                                  E_R=r[n_cicj][n_vel][tau][tau_prime][n_bin][n_element][0]
                                                  R_bar=r[n_cicj][n_vel][tau][tau_prime][n_bin][n_element][n_isotope+1]

                                                  import matplotlib.pyplot as pl
                                                  pl.plot(E_R, R_bar) 

                                                  plots the integrated response function vs. recoil energy for 
                                                  c_4^0*c_6^1, 
                                                  a=1E
                                                  3.5 keV<E_prime<4 keV
                                                  Iodine 127 (127I) target 

	      Methods:
			--------------------------------------------------------

		- efficiency(e_prime):
			Input:
			- e_prime	 - visible energy.

                         exp.efficiency is loaded from the first valid Python function parsed in the file efficiency.py,
                         if present, or interpolated from the content of the file efficiency.tab, which must contain
                         two columns with the visible energy and the efficiency.
                         If neither file is present returns 1.
			--------------------------------------------------------
			Output:
			Returns efficiency at visible energy e_prime. 
			--------------------------------------------------------

                - resolution(e_prime, e_ee)
                  --------------------------------------------------------
			Input:
			- e_prime(float)
			- e_ee(float)	
                  --------------------------------------------------------
                  Calculates the energy resolution in terms of the visible energy e_prime and of the 
                  electron-equivalent energy e_ee, where e_prime is the visible energy whose binning is 
                  initialized in data.tab while for the n-th element of exp.target:

                             e_ee=exp.target.element[n].quenching(E_R)*E_R 

                  is the electron-equivalent energy, with E_R the recoil energy in kev.  

                  The two arguments of the exp.resolution function need not be in keV. For example, 
                  for dual-phase xenon detectors:

                                    e_prime-> S1 

                                    e_ee -> <S1>=exp.target.element[n].quenching(E_R)*E_R 

                  In this case exp.target.element[n].quenching is the light yield, both S1 and <S1> 
                  are in Photo Electrons (PE), and the binning in data.tab refers to S1.
  
                  exp.resolution is loaded from the first valid Python function parsed in the file resolution.py,
                  if present.
		  --------------------------------------------------------
		  Output:
		  Returns the energy resolution function normalized to 1 upon integration over e_prime and e_ee. 
                  If the file resolution.py is missing exp.resolution is set to a function of a single argument 
                  that returns 1; detecting that exp.resolution has a single argument instead of two
                  load_response_functions calculates the response functions without the energy resolution.
		  --------------------------------------------------------

                  - quenching(E_R)

                   quenching is an attribute of the n-th element of the target:

                          target.element[n].quenching(E_R)

                  It converts the nuclear recoil energy to the electron-equivalent energy:

                        e_ee=quenching(E_R)*E_R

                  and smeared out to the visible energy E_prime whose binning is given in data.tab.
                  More generally, in coverts E_R to a quantity S that is used experimentally instead of the energy,
                  and that is eventually smeared out to a visible S'
                  by exp.resolution(S',S), and whose binning is given in data.tab. 

                  For instance in dual-phase xenon detectors the expected number nu of photoelectrons is used (S=nu),
                  and the quenching corresponds to the light-yield Ly:

                       nu=Ly*E_R

                 nu is smeared out to S'=S1 by xenon1t.resolution(S1,nu), and data.tab contains the binning 
                 in "visible energy" S1.

                  Quenching is loaded from the first valid Python function parsed in the file 

                   exp.target.element[n].name+'_quenching.py'

                  if present, or interpolated from the content of the file 

                   exp.target.element[n].name+'_quenching.tab'

                  which must contain two columns with the recoil energy and the quenching.
                  If neither file is present returns 1.
                  
                  Example:

                      xenon1t.target.element[0].quenching

                  is either loaded from the Python library:

                      xenon_quenching.py

                  or interpolated from the table:

                      xenon_quenching.tab
		  --------------------------------------------------------

		- modifier(er,eprime,exp,n_element,n_isotope,n_bin):

                  A user-defined function that modifies the expected rate in an arbitrary multiplicative way.
                  For costumization purposes. 
                  exp.modifier is loaded from the first valid Python function parsed in the file 

                   modifier.py

                  if present. If modifier.py is missing it is set to 1.


			Input:
			- er(float)			- recoil energy.
			- eprime(float)	         	- visible energy.
			- n_element(int)		- the number of elements.
			- n_isotope(int)		- the number of isotopes.
			- n_bin(int)		        - the number of energy bin.
			--------------------------------------------------------
			Output:
                        (default: 1) Multiplicative factor for the expected rate.
			--------------------------------------------------------




    '''
    
    def __init__(self, name,target_obj=None,verbose=False):

        if isinstance(target_obj,element):
            target_obj=1*target_obj

        
        self.name=name
        path=EXPERIMENTS_PATH+'/'+self.name            
        if not os.path.isdir(path):
            os.makedirs(os.path.join(path))
            #p = subprocess.call("mkdir -p "+path, stdout=subprocess.PIPE, shell=True)
        if not os.path.isfile(os.getcwd()+'/'+EXPERIMENTS_PATH+'/__init__.py'):
            open(os.getcwd()+'/'+EXPERIMENTS_PATH+'/__init__.py','x').close()
        if not os.path.isfile(os.getcwd()+'/'+EXPERIMENTS_PATH+'/'+self.name+'/__init__.py'):
            open(os.getcwd()+'/'+path+'/__init__.py','x').close()

        exposure=get_exposure(path+'/exposure',verbose=verbose)
        if isinstance(exposure[0], int) or isinstance(exposure[0],float):
            self.exposure=documented_float(exposure[0])
        else:
            self.exposure=documented_list(exposure[0])            
        self.exposure.info =exposure[1]

            
        #self.exposure=exposure
        self.efficiency=get_function_dir(path,'efficiency',verbose=verbose)
        
        self.modifier=get_function_dir(path,'modifier',verbose=verbose)
        #self.target=target(target_name,verbose)
        
        if os.path.isfile(path+'/target.tab'):
            f=open(path+'/target.tab','r')
            for line in f:
                target_name=line
            if verbose:
                print('loaded '+target_name+' from target.tab')

            try:
                
                self.target=target(target_name,verbose)
                
            except:
                
                if target_obj is None:
                    print('Could not load target, instantiation of experiment object coud not be finalized. Please try again providing a target.')
                    return None
                else:
                    target_name=target_obj.formula
                    self.target=target_obj
                    f.close()
                    f=open(path+'/target.tab','w')
                    f.write(target_obj.formula)
                    

                    
            if not target_obj is None:
                if target_name!=target_obj.formula:
                    print('WARNING: the target in '+path+'/target.tab is '+target_name+' and DOES NOT MATCH with '+target_obj.formula+'.')
                    print('To use '+target_obj.formula+' delete or modify target.tab') 
                    xx=input('Press enter to continue with '+target_name+':\n')
            
        else:
            f=open(path+'/target.tab','w+')
            if target_obj is None:                

                print('Target not provided. Could not find '+path+'/target.tab.')
                successful_target=False
                target_input=''
                while successful_target==False or target_input=='':
                    target_input=input('Enter now a valid target name (no quotation marks):\n')
                    try:
                        self.target=target(target_input,verbose)
                        successful_target=True
                    except:
                        pass
                    
                        
                if target_input:
                
                    self.target=target(target_input,verbose)

                else:
                    #print('WARNING!!! No target input provided. Arbitrarily assuming a Germanium target!')
                    print('No target input provided, instantiation of experiment objet could not be finalized. Please try again providing a target.')
                    #self.target=target('Ge',verbose)
                    return None
            else:
                self.target=target_obj
            f.write(self.target.formula)
            f.close()
        #self.target=target
        
        for i in range(self.target.n_targets):            
            self.target.element[i].quenching=get_function_dir(path,self.target.element[i].name+'_quenching',verbose=verbose)

        self.resolution=get_function_dir(path,'resolution',verbose=verbose)

        convert_events_to_data(path+'/events.tab',path+'/data.tab')

        data=get_data(path+'/data',verbose=verbose)
        self.data=documented_tuple(data[0])
        self.data.info =data[1]

        try:
            self.binned_background=documented_tuple(binned_background(self,verbose=verbose)[0])
            self.binned_background.__doc__=binned_background(self)[1]
        except:
            pass

            
        if np.size(self.exposure)>1 and np.size(self.exposure) != len(self.data):
            print('WARNING!: exposure provided as '+str(np.size(self.exposure))+'-dimensional array. If provided as array its\ndimension must be the same of the number of data points ('+str(len(self.data))+').\n All values besides the first('+str(self.exposure[0])+') ignored.')

            info=self.exposure.info
            self.exposure=documented_float(self.exposure[0])
            self.exposure.info=info
            
            
        self.response_functions=response_functions({})
        self.diff_response_functions=diff_response_functions({})        


    def __str__(self):
        #return "==================================================\n"+self.name+" uses "+str(self.exposure)+" kg*day of "+str(self.target.formula)+"\nAvaliable output:\nEnergy resolution function: "+self.name+".resolution(e_prime,e_ee)\nQuenching factor for each element: "+self.name+".target.element[i].quenching(er)\nEfficiency: "+self.name+".efficiency(eprime)\nData bins: "+self.name+".data\nDifferential response function for each element i and isotope j:\n"+self.name+".target.element[i].response_function[j](er,eprime)\nTables of integrated response functions Rbar vs. Er in tuple\n "+self.name+".response_functions[i_exp_point][i_element][0]-"+self.name+".response_functions[i_exp_point][i_element][i_isotope+1]\nAll indices start from 0 so first isotope correspond to 0 index etc.\n"+"===================================="
        return '\n'.join([t for t in get_info(self)])


    

def get_info(exp):
    info=()
    info+=('=========================================================================================',)
    info+=('Experiment name : '+str(exp.name),)
    info+=('******************',)
    info+=('Target :',)
    info+=(str(exp.target),)
    info+=('Exposure (kg-days) : '+str(exp.exposure)+', '+str(exp.exposure.info),)
    for i in range(exp.target.n_targets):
         info+=('Quenching for '+str(exp.target.element[i].name)+ ' : '+str(exp.target.element[i].quenching.__doc__),)
         info+=('******************',)
         for j in range(exp.target.element[i].n_isotopes):
             info+=('Nuclear response functions for '+exp.target.element[i].isotopes[j]+'. '+exp.target.element[i].func_w[j].__doc__,)
    info+=('******************',)             
    info+=('Energy bins : '+str(exp.data.info),)     
    info+=('Efficiency : '+str(exp.efficiency.__doc__),)
    info+=('Energy resolution for : '+ str(exp.resolution.__doc__),) 
    info+=('Modifier : ' + str(exp.modifier.__doc__),)
    try:
        info+=('Background estimation for '+str(exp.name)+' : ' + str(exp.binned_background.__doc__),)
    except:
        pass
    info+=('=========================================================================================',)
    return info




def print_string(string):
    for i in range(len(string)):
        print(string[i])

        

def last_modified_file(path,inputfile,exceptions=['*']):
    file=''
    latest_time=-1.

    list=os.listdir(path)

    npy_file_list=[f for f in os.listdir(path) if '.npy' in f]
    
    ####WARNING! this only works in linux and mac/os
    hidden_file_list=[f for f in os.listdir(path) if f.startswith('.')]

    
    exceptions=np.append(exceptions,npy_file_list)
    exceptions=np.append(exceptions,hidden_file_list)    
    
    
    for i in np.array(list)[np.where(np.array(['.pyc' not in list[i] for i in range(len(list))]))]:
        #if not any (s in i for s in exceptions):
        if not any (s==i for s in exceptions):             
            time=os.stat(os.path.join(path,i)).st_mtime
            if  time > latest_time:
                latest_time=time
                file=i
            
    return (file,latest_time)

def is_last_modified_file(file,exp,j_chi):

    path_exp=EXPERIMENTS_PATH+'/'+exp.name
    path_response_functions=path_exp+'/Response_functions/spin_'+print_spin(j_chi)

    time=os.stat(os.path.join(path_response_functions,file)).st_mtime

    expected_files=set(['target.tab','exposure.tab','efficiency.tab','efficiency.py','resolution.py','xenon_quenching.py','xenon_quenching.tab','data.tab','modifier.tab'])

    list_exp=set(os.listdir(path_exp)).intersection(expected_files)

    latest_time=-1.
    for f in list_exp:
        t=os.stat(os.path.join(path_exp,f)).st_mtime
        if  t > latest_time:
            latest_time=t
            latest_file=f
    last=time>latest_time
    if last:
        return last,file
    else:
        return last,latest_file



def is_last_modified_file_old(path,file,exceptions=['*']):
    time1=os.stat(file).st_mtime
    file,time2=last_modified_file(path,file,exceptions=exceptions)
    return time1>time2

    
def delayed_input(timeout,string,default):
    print(string)
    start_time=win_time.time()
    while True:
        try:
            if keyboard.is_pressed('q'):
                print("input:")
                return input()
            else:
                pass
            if (win_time.time()- start_time) > timeout:
                return default
            else:
                pass
        except:
            pass


def primitive_table_resolution_bin_interf(exp,n,tau,tau_prime,response_function,n_sampling=100,verbose=True):

    path=EXPERIMENTS_PATH+'/'+exp.name            
    
    eprime1=exp.data[n][0]
    eprime2=exp.data[n][1]

    exposure_bin=1.
    if np.size(exp.exposure)>1:
        exposure_bin=exp.exposure[n]
    
    print(('calculating energy bin:',eprime1,eprime2))
    
    n_elements=exp.target.element.shape[0]
    tuple=()
    res=exp.resolution

    
    for n_element in range(n_elements):

        n_isotopes=exp.target.element[n_element].a.shape[0]        
        q=exp.target.element[n_element].quenching
        eff=exp.efficiency
        
        table=np.array([])

        er1=op.bisect(lambda er,eprime:q(er)*er-eprime,1e-3,1e3,eprime1)
        er2=op.bisect(lambda er,eprime:q(er)*er-eprime,1e-3,1e3,eprime2)

        
        result=op.minimize(lambda eprime,er:-res(eprime,q(er)*er),q(er1)*er1,er1,method='Nelder-Mead')
        xmax=result.x
        #print('xmax=',xmax,' er1=',er1,' er2=',er2)
        
        x2=op.bisect(lambda eprime,er:res(eprime,q(er)*er)/res(q(er)*er,q(er)*er)-0.5,xmax,10*xmax,er1)

        if res(1e-3,q(er1)*er1)/res(q(er1)*er1,q(er1)*er1)<0.5:

            x1=op.bisect(lambda eprime,er:res(eprime,q(er)*er)/res(q(er)*er,q(er)*er)-0.5,1e-3,xmax,er1)

        else:
    
            x1=1e-3
        #estimation of the energy resolution    
        delta_eprime=(x2-x1)/2.

        #convert into er
        y1=op.bisect(lambda er,eprime:q(er)*er-eprime,1e-3,1e3,x1)
        y2=op.bisect(lambda er,eprime:q(er)*er-eprime,1e-3,1e3,x2)

        delta_er=(y2-y1)/2.

        print('Er resolution estimation:',delta_er)
        print('xmax=',xmax,' x1=',x1,' x2=',x2)
    
        nstep=n_sampling
        if (eprime2-eprime1)/delta_eprime>nstep:
            nstep=int((eprime2-eprime1)/delta_eprime)
        
        er1=max(1e-3,er1-3*delta_er)
        er2=er2+3*delta_er
        print(('Er interval:',er1,er2))
        er_vec=np.linspace(er1,er2,nstep)

        n_add=20
        er_add=np.linspace(0,er_vec[0],n_add)
        er_vec_out=np.append(er_add[:-1],er_vec)
        er_add=np.linspace(er_vec[-1],er_vec[-1]*1.2,n_add+2)        
        er_vec_out=np.append(er_vec_out,er_add[1:])
        
        table=np.append(table,er_vec_out).reshape(-1,nstep+2*n_add)
        
        print('nstep=',nstep)

        
        for n_isotope in range(n_isotopes):

            
            f_vec=np.array([])        
            print('calculating: ',exp.target.element[n_element].isotopes[n_isotope])


            #modifier=get_function('modifier')            
            modifier=get_function_dir(path,'modifier',verbose=verbose)

            if len(getargspec(modifier).args)==1:                

                r=response_function[n_element][tau,tau_prime,n_isotope]
                
            else:                
                def r(er,eprime):

                    return modifier(er,eprime,exp,n_element,n_isotope,n)*response_function[n_element][tau,tau_prime,n_isotope](er,eprime)

            frac=1e-3
            i=0
            n_show=int(nstep/10)
            
            for er in er_vec:
                i+=1

                x1=0
                x2=0
                if q(er)>0 and eff(q(er)*er)>0:
                    if res(q(er)*er,q(er)*er)>0:
                        result=op.minimize(lambda eprime,er:-res(eprime,q(er)*er),q(er)*er,er,method='Nelder-Mead')
                        xmax=result.x
                        
                        x_large=10*xmax
                        while res(x_large,q(er)*er)/res(q(er)*er,q(er)*er)>frac:
                            x_large=2.*x_large
                        
                        x2=op.bisect(lambda eprime,er:res(eprime,q(er)*er)/res(q(er)*er,q(er)*er)-frac,xmax,x_large,er)

                
                        if res(1e-3,q(er)*er)/res(q(er)*er,q(er)*er)<frac:

                            x1=op.bisect(lambda eprime,er:res(eprime,q(er)*er)/res(q(er)*er,q(er)*er)-frac,1e-3,xmax,er)

                        else:

                            x1=1e-3

                integral=0.
                intersect, e_prime_interval=overlap([x1,x2],[eprime1,eprime2])
                if intersect:
                    integral=integrate.quadrature(lambda e_prime, er:r(er,e_prime),e_prime_interval[0],e_prime_interval[1], args=(er), rtol=1.e-03, maxiter=100, vec_func=False)[0]
                    
                f_vec=np.append(f_vec,integral)
                
                if i%n_show == 0:
                    print(str(int(10*i/nstep))+'0% evaluated')

            #tck=interpolate.splrep(er_vec,f_vec, s=0)    

            #np.save('test_'+str(n_element)+'_'+str(n_isotope)+'_'+str(eprime1)+'_'+str(eprime2)+'.npy',np.append(er_vec,f_vec).reshape(2,-1))

            def f_inter(x):
                #return interpolate.splev(x,tck,der=0,ext=1)
                return np.interp(x, er_vec, f_vec)
            

            
            integral=0.
            final=np.array([])
            for i in range(nstep)[:-1]:
                integral+=integrate.quadrature(f_inter,er_vec[i],er_vec[i+1], rtol=1e-3, maxiter=100, vec_func=False)[0]
                final=np.append(final,exposure_bin*integral)


                
            final=np.append(np.zeros(n_add),final)
            final=np.append(final,final[-1]*np.ones(n_add+1))

            table=np.append(table,final).reshape(-1,nstep+2*n_add)
        tuple+=((table,))
    
    return tuple

def primitive_table_resolution_diff_interf(exp,n,tau,tau_prime,response_function,n_sampling=100,verbose=True):

    path=EXPERIMENTS_PATH+'/'+exp.name            
    exposure_bin=1.
    if np.size(exp.exposure)>1:
        exposure_bin=exp.exposure[n]

                  
    eprime=exp.data[n][0]     
    print(('calculating energy value:',eprime))


                  
    n_elements=exp.target.element.shape[0]
    tuple=()
    res=exp.resolution
    
    for n_element in range(n_elements):

        n_isotopes=exp.target.element[n_element].a.shape[0]        
        q=exp.target.element[n_element].quenching
        
        table=np.array([])

        er=op.bisect(lambda er,eprime:q(er)*er-eprime,1e-3,1e3,eprime)

        result=op.minimize(lambda eprime,er:-res(eprime,q(er)*er),q(er)*er,er,method='Nelder-Mead')
        xmax=result.x

        x2=op.bisect(lambda eprime,er:res(eprime,q(er)*er)/res(q(er)*er,q(er)*er)-1e-3,xmax,10*xmax,er)

        if res(1e-3,q(er)*er)/res(q(er)*er,q(er)*er)<1e-3:

            x1=op.bisect(lambda eprime,er:res(eprime,q(er)*er)/res(q(er)*er,q(er)*er)-1e-3,1e-3,xmax,er)

        else:
    
            x1=1e-3

        er1=op.bisect(lambda er,eprime:q(er)*er-eprime,1e-3,1e3,x1)
        er2=op.bisect(lambda er,eprime:q(er)*er-eprime,1e-3,1e3,x2)
            

        nstep=n_sampling
        er_vec=np.linspace(er1,er2,nstep)

        n_add=20
        er_add=np.linspace(0,er_vec[0],n_add)
        er_vec_out=np.append(er_add[:-1],er_vec)
        er_vec_out=np.append(er_vec_out,er_vec[-1]*1.2)
        
        
        table=np.append(table,er_vec_out).reshape(-1,nstep+n_add)
        
        print('nstep=',nstep)

        for n_isotope in range(n_isotopes):
            f_vec=np.array([])        
            print('calculating: ',exp.target.element[n_element].isotopes[n_isotope])

            #modifier=get_function('modifier')
            modifier=get_function_dir(path,'modifier',verbose=verbose)                        

            if len(getargspec(modifier).args)==1:                            
                modifier(1)
                r=response_function[n_element][tau,tau_prime,n_isotope]
                
            else:                
                def r(er,eprime):
                    return modifier(er,eprime,exp,n_element,n_isotope,n)*response_function[n_element][tau,tau_prime,n_isotope](er,eprime)

            i=0
            n_show=int(nstep/10)
            for er in er_vec:
                i+=1
                if i%n_show == 0:
                    print(str(int(10*i/nstep))+'0% evaluated')

                f_vec=np.append(f_vec,r(er,eprime))

            #tck=interpolate.splrep(er_vec,f_vec, s=0)    

            def f_inter(x):
                #return interpolate.splev(x,tck,der=0,ext=3)
                return np.interp(x,er_vec,f_vec)

            
            integral=0.
            final=np.array([])
            for i in range(nstep)[:-1]:
                integral+=integrate.quadrature(f_inter,er_vec[i],er_vec[i+1],tol=1.49e-08, rtol=1.49e-08, maxiter=50, vec_func=False)[0]
                final=np.append(final,exposure_bin*integral)

            final=np.append(np.zeros(n_add),final)
            final=np.append(final,final[-1])

            table=np.append(table,final).reshape(-1,nstep+n_add)
        tuple+=((table,))

    return tuple

def primitive_table_no_resolution_diff_interf(exp,n,tau,tau_prime,response_function,n_sampling=100,verbose=True):

    path=EXPERIMENTS_PATH+'/'+exp.name                
    exposure_bin=1.
    if np.size(exp.exposure)>1:
        exposure_bin=exp.exposure[n]
                  
    eprime=exp.data[n][0] 
    
    print(('calculating energy value:',eprime))
    
    n_elements=exp.target.element.shape[0]
    tuple=()
    
    for n_element in range(n_elements):

        n_isotopes=exp.target.element[n_element].a.shape[0]        
        q=exp.target.element[n_element].quenching
        
        table=np.array([])

        er0=op.bisect(lambda er,eprime:q(er)*er-eprime,1e-3,1e3,eprime)
            

        nstep=n_sampling
        er_vec=np.linspace(0.5*er0,2*er0,nstep)
                
        table=np.append(table,er_vec).reshape(-1,nstep)
        
        print('nstep=',nstep)

        for n_isotope in range(n_isotopes):
            f_vec=np.array([])        
            print('calculating: ',exp.target.element[n_element].isotopes[n_isotope])

            #modifier=get_function('modifier')
            modifier=get_function_dir(path,'modifier',verbose=verbose)            

            if len(getargspec(modifier).args)==1:                            
                modifier(1)
                r=response_function[n_element][tau,tau_prime,n_isotope]
                
            else:                
                def r(er,eprime):
                    return modifier(er,eprime,exp,n_element,n_isotope,n)*response_function[n_element][tau,tau_prime,n_isotope](er,eprime)


                
            i=0
            n_show=int(nstep/10)
            for er in er_vec:
                i+=1
                if i%n_show == 0:
                    print((str(int(10*i/nstep))+'0% evaluated'))

                if er>er0:
                    f_vec=np.append(f_vec,exposure_bin*r(er0,eprime))

                else:

                    f_vec=np.append(f_vec,0.)
                
            table=np.append(table,f_vec).reshape(-1,nstep)
        tuple+=((table,))

    return tuple

def primitive_table_no_resolution_bin_interf(exp,n,tau,tau_prime,response_function,n_sampling=100,verbose=True):

    path=EXPERIMENTS_PATH+'/'+exp.name                
    eprime1=exp.data[n][0]
    eprime2=exp.data[n][1]

    exposure_bin=1.
    if np.size(exp.exposure)>1:
        exposure_bin=exp.exposure[n]
                  
    print(('calculating energy bin:',eprime1,eprime2))
    
    n_elements=exp.target.element.shape[0]
    tuple=()

    for n_element in range(n_elements):

        n_isotopes=exp.target.element[n_element].a.shape[0]        
        q=exp.target.element[n_element].quenching
        
        table=np.array([])

        er01=op.bisect(lambda er,eprime:q(er)*er-eprime,1e-3,1e3,eprime1)
        er02=op.bisect(lambda er,eprime:q(er)*er-eprime,1e-3,1e3,eprime2)
        
        nstep=n_sampling
            
        print(('Er interval:',er01,er02))            
        er_vec=np.linspace(er01,er02,nstep)

        n_add=20
        er_add=np.linspace(0,er_vec[0],n_add)
        er_vec_out=np.append(er_add[:-1],er_vec)
        er_vec_out=np.append(er_vec_out,er_vec[-1]*1.2)
        
        table=np.append(table,er_vec_out).reshape(-1,nstep+n_add)
        
        print('nstep=',nstep)

        for n_isotope in range(n_isotopes):
            final=np.array([])        
            print('calculating: ',exp.target.element[n_element].isotopes[n_isotope])

            #modifier=get_function('modifier')
            modifier=get_function_dir(path,'modifier',verbose=verbose)            

            if len(getargspec(modifier).args)==1:
                r=response_function[n_element][tau,tau_prime,n_isotope]
                
            else:                
                def r(er,eprime):
                    return modifier(er,eprime,exp,n_element,n_isotope,n)*response_function[n_element][tau,tau_prime,n_isotope](er,eprime)

            n_show=int(nstep/10)
            integral_sum=0.                        
            for i in range(nstep)[:-1]:
                if i%n_show == 0:
                    print(str(int(10*i/nstep))+'0% evaluated')

                integral=0.
                intersect, er_interval=overlap([er01,er02],[er_vec[i],er_vec[i+1]])
                if intersect:
                    integral=integrate.quadrature(r,er_interval[0],er_interval[1],0.,rtol=1.e-03, maxiter=100, vec_func=False)[0]

                    
                integral_sum+=integral
                final=np.append(final,exposure_bin*integral_sum)

            final=np.append(np.zeros(n_add),final)                
            final=np.append(final,final[-1])
            
            table=np.append(table,final).reshape(-1,nstep+n_add)
        tuple+=((table,))
        
        
    return tuple

def primitive_table_interf(exp,tau,tau_prime,response_function,n_sampling=100,verbose=True):

    res=exp.resolution
    tuple=()
    
    n_exp_points=len(exp.data)

    for n in range(n_exp_points):
        if exp.data[n].shape[0] > 1:
            print('analyzing interval:',exp.data[n][0],'<eprime<',exp.data[n][1])


            if len(getargspec(exp.resolution).args)==2:
                print('energy resolution available')
                tuple+=((primitive_table_resolution_bin_interf(exp,n,tau,tau_prime,response_function,n_sampling=n_sampling,verbose=verbose),))
            else:
                print('energy resolution not available')                
                tuple+=((primitive_table_no_resolution_bin_interf(exp,n,tau,tau_prime,response_function,n_sampling=n_sampling,verbose=verbose),))                
        elif exp.data[n].shape[0] == 1:

            print('analyzing eprime=',exp.data[n][0])
            if len(getargspec(exp.resolution).args)==2:            
                tuple+=((primitive_table_resolution_diff_interf(exp,n,tau,tau_prime,response_function,n_sampling=n_sampling,verbose=verbose),))
            else:
                tuple+=((primitive_table_no_resolution_diff_interf(exp,n,tau,tau_prime,response_function,n_sampling=n_sampling,verbose=verbose),))
                
        else:
            print('data point ',exp.data[n],' is neither an energy value nor an energy interval')

    return tuple

    


def get_response_functions_interf(exp,n_coeff1,n_coeff2=None,coeff1_input=None,coeff2_input=None, eft_amplitude_squared=eft_amplitude_squared,eft_amplitude_squared1=eft_amplitude_squared1,eft_modifier=lambda q:1.,j_chi=0.5,outputfile=None,verbose=True,n_sampling=100):

    if n_coeff2 is None:
        n_coeff2=n_coeff1

    if get_short_coupling(convert_to_all_spins(n_coeff1))!=get_short_coupling(convert_to_all_spins(n_coeff2)) and not symmetric_interference(convert_to_all_spins(n_coeff1),convert_to_all_spins(n_coeff2)) and not non_symmetric_interference(convert_to_all_spins(n_coeff1),convert_to_all_spins(n_coeff2)):
        # all response functions are zero. fills in a tuple with the correct stucture containing all zeros

        response_functions_interf=vanishing_response_functions(exp)
        if verbose:
            print('response functions for c_'+print_coupling(n_coeff1)+'-c_'+print_coupling(n_coeff2)+' are vanishing')
        return response_functions_interf
        
    if coeff1_input is not None:
        if coeff2_input is None:
            coeff2_input=coeff1_input

    if not os.path.isfile(os.getcwd()+'/'+EXPERIMENTS_PATH+'/__init__.py'):
        open(os.getcwd()+'/'+EXPERIMENTS_PATH+'/__init__.py','a').close()
            
    path_exp=EXPERIMENTS_PATH+'/'+exp.name            
    if not os.path.isdir(path_exp):
        os.makedirs(os.path.join(path_exp))
       #p = subprocess.call("mkdir -p "+path_exp, stdout=subprocess.PIPE, shell=True)
    if not os.path.isfile(os.getcwd()+path_exp+'/__init__.py'):
        open(os.getcwd()+'/'+path_exp+'/__init__.py','a+').close()

    path_response_functions=path_exp+'/Response_functions'
    if not os.path.isdir(path_response_functions):
        os.makedirs(os.path.join(path_response_functions))
       #p = subprocess.call("mkdir -p "+path_response_functions, stdout=subprocess.PIPE, shell=True)        
    if not os.path.isfile(os.getcwd()+'/'+path_response_functions+'/__init__.py'):
        open(os.getcwd()+'/'+path_response_functions+'/__init__.py','a+').close()

        
    path_spin=path_response_functions+'/spin_'+print_spin(j_chi)            
    if not os.path.isdir(path_spin):
        os.makedirs(os.path.join(path_spin))
       #p = subprocess.call("mkdir -p "+path_spin, stdout=subprocess.PIPE, shell=True)                
    if not os.path.isfile(os.getcwd()+'/'+path_spin+'/__init__.py'):
        open(os.getcwd()+'/'+path_spin+'/__init__.py','a+').close()
        
    path=path_spin
        
                
    time_w_func=0
    for i in range(exp.target.n_targets):
        #exp.target.element[i].func_w=np.array([])
        for j in range (exp.target.element[i].n_isotopes):            

            #func_w=define_w(exp.target.element[i],j,verbose)

            #exp.target.element[i].func_w=np.append(exp.target.element[i].func_w,func_w)

            try:
                time_w_func=max(time_w_func,os.stat(exp.target.element[i].isotopes[j]+'_func_w.py').st_mtime)
            except:
                pass

        
    for i in range(exp.target.n_targets):

        exp.target.element[i].response_function_interf=np.array([])
        for tau in range(2):
            response_function_interf_tau_prime=()
            for tau_prime in range(2):
                if coeff1_input is None:
                    coeff1=set_c_coeff_interf(n_coeff1,tau)
                else:
                    coeff1=coeff1_input
                    
                if coeff2_input is None:
                    coeff2=set_c_coeff_interf(n_coeff2,tau_prime)
                else:
                    coeff2=coeff2_input

                    
                r=np.array([])                
                for j in range (exp.target.element[i].n_isotopes):


                
                    if len(getargspec(exp.resolution).args)==2:                
                        r=np.append(r,def_response_function(coeff1,exp.target.element[i],j,exp.efficiency,exp.target.element[i].quenching,exp.resolution,exp.exposure,eft=lambda coeff1,q,element,isotope,coeff2: eft_modifier(q)*eft_amplitude_squared(coeff1,q,element,isotope,coeff2,j_chi),coeff2=coeff2))

                        r=np.append(r,def_response_function(coeff1,exp.target.element[i],j,exp.efficiency,exp.target.element[i].quenching,exp.resolution,exp.exposure,eft=lambda coeff1,q,element,isotope,coeff2: eft_modifier(q)*eft_amplitude_squared1(coeff1,q,element,isotope,coeff2,j_chi),coeff2=coeff2))

                        r=np.append(r,def_response_function(coeff1,exp.target.element[i],j,exp.efficiency,exp.target.element[i].quenching,exp.resolution,exp.exposure,eft=lambda coeff1,q,element,isotope,coeff2: eft_modifier(q)*eft_amplitude_squared1(coeff1,q,element,isotope,coeff2,j_chi)*q**2/(2.*element.mass[isotope]),coeff2=coeff2))

                        r=np.append(r,def_response_function(coeff1,exp.target.element[i],j,exp.efficiency,exp.target.element[i].quenching,exp.resolution,exp.exposure,eft=lambda coeff1,q,element,isotope,coeff2: eft_modifier(q)*eft_amplitude_squared1(coeff1,q,element,isotope,coeff2,j_chi)*2.*element.mass[isotope]/q**2,coeff2=coeff2))
                    
                    else:
                        r=np.append(r,def_response_function_nores(coeff1,exp.target.element[i],j,exp.efficiency,exp.target.element[i].quenching,exp.exposure,eft=lambda coeff1,q,element,isotope,coeff2: eft_modifier(q)*eft_amplitude_squared(coeff1,q,element,isotope,coeff2,j_chi),coeff2=coeff2))

                        r=np.append(r,def_response_function_nores(coeff1,exp.target.element[i],j,exp.efficiency,exp.target.element[i].quenching,exp.exposure,eft=lambda coeff1,q,element,isotope,coeff2: eft_modifier(q)*eft_amplitude_squared1(coeff1,q,element,isotope,coeff2,j_chi),coeff2=coeff2))

                        r=np.append(r,def_response_function_nores(coeff1,exp.target.element[i],j,exp.efficiency,exp.target.element[i].quenching,exp.exposure,eft=lambda coeff1,q,element,isotope,coeff2: eft_modifier(q)*eft_amplitude_squared1_e(coeff1,q,element,isotope,coeff2,j_chi),coeff2=coeff2))

                        r=np.append(r,def_response_function_nores(coeff1,exp.target.element[i],j,exp.efficiency,exp.target.element[i].quenching,exp.exposure,eft=lambda coeff1,q,element,isotope,coeff2: eft_modifier(q)*eft_amplitude_squared1_em1(coeff1,q,element,isotope,coeff2,j_chi),coeff2=coeff2))
   
                response_function_interf_tau_prime=np.append(response_function_interf_tau_prime,r)
            exp.target.element[i].response_function_interf=np.append(exp.target.element[i].response_function_interf,response_function_interf_tau_prime)
                            
        exp.target.element[i].response_function_interf=exp.target.element[i].response_function_interf.reshape(2,2,-1,4)

    info_new=get_info(exp)

    vel_dep_string=np.array(['R0','R1','R1*E_R','R1/E_R'])

    if n_coeff1 is not None and outputfile is None:
        inputfile='c_'+print_coupling(n_coeff1)+'_c_'+print_coupling(n_coeff2)+filename_suffix+'.npy'
        inputfile2='c_'+print_coupling(n_coeff2)+'_c_'+print_coupling(n_coeff1)+filename_suffix+'.npy'
    else:

        if outputfile is None:
            inputfile='custom_response_functions_'+filename_suffix+'.npy'
        else:
            inputfile=outputfile

        inputfile2=inputfile
        
    flag_tau_tau_prime_flip=0

    '''
    if not os.path.isfile(path+'/'+inputfile) and os.path.isfile(path+'/'+inputfile2):
        inputfile=inputfile2
        n_coeff3=n_coeff1
        n_coeff1=n_coeff2
        n_coeff2=n_coeff3
        flag_tau_tau_prime_flip=1
    '''   
    
    if os.path.isfile(path+'/'+inputfile):
        if verbose:
            print('loading '+inputfile)
            print('type: print_string(np.load(\''+path+'/'+inputfile+'\',allow_pickle=True)[-1]) to get info about input used')

        
        with open(path+'/'+inputfile, 'rb') as file:
            response_functions_interf=pickle.load(file,encoding='latin1')            
                
        info_old=response_functions_interf[-1]

        time_response_functions=os.stat(path+'/'+inputfile).st_mtime

        n_count=0
        if info_old !=info_new and verbose:
            print(20*'*')
            print('WARNING: some input changed:')
            for (string_old,string_new) in zip(info_old,info_new):
                if string_old!=string_new:
                    n_count+=1
                    print(10*'-'+str(n_count)+10*'-')
                    print('info from response function file '+inputfile+':')
                    print(string_old)
                    print('info from '+path_exp+':')
                    print(string_new)
            print(20*'*'+'Warning'+20*'*'+'\nThe documentation collected from the directory '+path_exp+' does not match with that contained in the file '+path+'/'+inputfile+'.\nIf any modification in the input of the '+path_exp+' directory is more recent than the file '+path+'/'+inputfile+' you need to recalculate it. If this is not the case consider to use update_exp_info to update the documentation in '+inputfile+' to stop this warning.')
            print(20*'*')

        last,latest_file=is_last_modified_file(inputfile,exp,j_chi)
            
        #if not is_last_modified_file(path_exp,path+'/'+inputfile,exceptions=get_exceptions_to_last_modified_path_interf(n_coeff1,n_coeff2)) or time_w_func>time_response_functions:
        if not last or time_w_func>time_response_functions:
            print('**********************************************')
            print('*******************WARNING********************')
            print('**********************************************')


            #if not is_last_modified_file(path_exp,path+'/'+inputfile,exceptions=get_exceptions_to_last_modified_path_interf(n_coeff1,n_coeff2)):
            if not last:
                #print('in '+path_exp+' the file '+last_modified_file(path_exp,path+'/'+inputfile,exceptions=get_exceptions_to_last_modified_path_interf(n_coeff1,n_coeff2))[0]+' is more recent than '+inputfile)
                print('in '+path_exp+' the file '+latest_file+' is more recent than '+inputfile)

                
                
            if time_w_func>time_response_functions:
                print('a custom nuclear form factor for one of the targets is more recent then '+inputfile)

            print('if response functions need not to be updated use update_time_stamp=True ')   
            print('when calling load_response functions to avoid this warning in the future') 
            reply=int(delayed_input(10,'Type 1 if you want to recalculate the response functions, 2 if you want to use the response functions stored in '+path+'/'+inputfile+', 0 otherwise (default answer is 2 after 10 seconds). To type integer, press "q"',2))
            #reply=1
            if reply==1:
                response_functions_interf=()
                
                for n_vel in range(4):
                    if verbose:
                        print('calculating '+vel_dep_string[n_vel])
                    response_functions_tau=()        
                    for tau in range(2):
                        response_functions_tau_prime=()
                        for tau_prime in range(2):

                            if verbose:
                                print('calculating tau='+str(tau)+', tau_prime='+str(tau_prime))

                            if tau>tau_prime and n_coeff1==n_coeff2:

                                response_functions_tau_prime+=(response_functions_tau[0][1],)                                

                            else:
                                
                                response_functions_tau_prime+=(primitive_table_interf(exp,tau,tau_prime,[exp.target.element[i].response_function_interf[:,:,:,n_vel] for i in range(exp.target.n_targets)],n_sampling=n_sampling,verbose=verbose),)


                        response_functions_tau+=(response_functions_tau_prime,)
                    response_functions_interf+=(response_functions_tau,)    
                response_functions_interf+=(info_new,)
                
                with open(path+'/'+inputfile, 'wb') as file:
                    pickle.dump(response_functions_interf, file)
                    
            elif reply==2:
                pass
            
            elif reply==0:                    
                
                print('Response functions for '+exp.name+' have not been loaded and '+exp.name+'.response_functions has been set to the empty tuple (). Also the info in the file '+inputfile+' has not been updated.')
                response_functions_interf=()

        else:
            if flag_tau_tau_prime_flip==1:
                flip_tau_tau_prime_in_response_functions_interf(exp,response_functions_interf)
            
            
    if not os.path.isfile(path+'/'+inputfile):

        if coeff1_input is None:
            reply=int(delayed_input(10,inputfile+' not available for '+exp.name+' in '+path+'. Type 1 if you want to calculate the response functions, 0 otherwise (default answer is 1 after 10 seconds). To type integer, press "q"',1))
        else:
            reply=int(delayed_input(10,'Response functions not available for '+inputfile+' in '+path+'. Type 1 if you want to calculate the response functions, 0 otherwise (default answer is 1 after 10 seconds). To type integer, press "q"',1))

        
        #reply=input('response functions not available for '+name+' and c'+str(n_coeff+1)+' in '+path+'. Type 1 if you want to calculate them, 0 otherwise.\n')
        #reply=1
        
        if reply==1:
            response_functions_interf=()

            for n_vel in range(4):
                if verbose:
                    print('calculating '+vel_dep_string[n_vel])
                response_functions_tau=()        
                for tau in range(2):
                    response_functions_tau_prime=()
                    for tau_prime in range(2):

                        if verbose:
                            print('calculating tau='+str(tau)+', tau_prime='+str(tau_prime))

                        if tau>tau_prime and n_coeff1==n_coeff2:

                            response_functions_tau_prime+=(response_functions_tau[0][1],)                                
                        else:
                            
                            response_functions_tau_prime+=(primitive_table_interf(exp,tau,tau_prime,[exp.target.element[i].response_function_interf[:,:,:,n_vel] for i in range(exp.target.n_targets)],n_sampling=n_sampling,verbose=verbose),)


                    response_functions_tau+=(response_functions_tau_prime,)

                            
                response_functions_interf+=(response_functions_tau,)    
                
            response_functions_interf+=(info_new,)

            if flag_tau_tau_prime_flip==1:
                flip_tau_tau_prime_in_response_functions_interf(exp,response_functions_interf)
            
            
            with open(path+'/'+inputfile, 'wb') as file:
                pickle.dump(response_functions_interf, file)

            
        else:
            print('Response functions for '+exp.name+' have not been calculated. '+exp.name+'.response_functions has been set to the empty tuple ().')
            response_functions_interf=()

    return response_functions_interf



def get_exceptions_to_last_modified_path_interf(n_coeff1,n_coeff2):

    exceptions=[]


    exceptions=np.append(exceptions,'__init__.pyc')
    exceptions=np.append(exceptions,'__init__.py')    
    exceptions=np.append(exceptions,'background.tab')
    exceptions=np.append(exceptions,'binned_background.tab')    
    exceptions=np.append(exceptions,'resolution.pyc')
    exceptions=np.append(exceptions,'efficiency.pyc')
    exceptions=np.append(exceptions,'Response_functions')    
        
    
    return exceptions        


                                                                
def vanishing_response_functions(exp):

    try:
        info=exp.response_functions_interf[-1]
    except:
        info=()
    
    exp.response_functions_interf=()
    
    for n_vel in range(4):
        response_functions_tau=()        
        for tau in range(2):
            response_functions_tau_prime=()
            for tau_prime in range(2):
                response_functions_n_bin=()        
                for n_bin in range(len(exp.data)):
                    response_functions_n_element=()
                    for n_element in range(exp.target.n_targets):
                        response_functions_n_element+=(np.zeros(100*(exp.target.element[n_element].n_isotopes+1)).reshape(exp.target.element[n_element].n_isotopes+1,-1),)
                    response_functions_n_bin+=(response_functions_n_element,)

                response_functions_tau_prime+=(response_functions_n_bin,)
            response_functions_tau+=(response_functions_tau_prime,)

                            
        exp.response_functions_interf+=(response_functions_tau,)    
                
    exp.response_functions_interf+=(info,)

    return exp.response_functions_interf


def get_vstar(mn,mchi,delta):
    '''
    Calculates minimal WIMP incoming speed in the lab frame for inelastic scattering
    --------------------------------------------------------------------------------
    Input:

    mn: Nuclear mass in GeV

    mchi: WIMP mass in GeV

    delta: mass splitting for inelastic scattering, in keV
    --------------------------------------------------------------------------------
    Output:

    velocity in the units of km/s.
    --------------------------------------------------------------------------------
    '''
    
    mu=mchi*mn/(mchi+mn)

    vstar=np.choose(delta>0,[0.,np.sqrt(2*abs(delta)/mu)*300])

    return vstar


def get_estar(mn,mchi,delta):
    '''
    Calculates recoil energy corresponding to minimal WIMP incoming speed in 
    the lab frame for inelastic scattering
    ------------------------------------------------------------------------
    Input:

    mn: Nuclear mass in GeV

    mchi: WIMP mass in GeV

    delta: mass splitting for inelastic scattering, in keV
    -----------------------------------------------------------------------
    Output:

    recoil energy in keV
    -----------------------------------------------------------------------

    '''
    mu=mchi*mn/(mchi+mn)

    estar=abs(delta)*mu/mn

    return estar

def er_max_old(mn,mchi,delta,vmin):
    '''
    Calculates the maximum value of recoil energy for a target nuclei
    ---------------------------------------------------------------------------
    Input:

    mn: Nuclear mass in GeV

    mchi: WIMP mass in GeV

    delta: mass splitting for inelastic scattering, in keV

    vmin: minimum WIMP velocity (in lab frame) required to deposit a recoil energy in km/s
    ---------------------------------------------------------------------------
    Output:

    maximum recoil energy in keV
    -------------------------------------------------------------------------- 
    '''
    
    mu=mchi*mn/(mchi+mn)      
    vstar=get_vstar(mn,mchi,delta)
    vmin1=np.choose(vmin>vstar,[vstar,vmin])
    return (mu/np.sqrt(2.*mn)*(vmin1+np.sqrt(vmin1**2-vstar**2))/300.)**2

def er_min_old(mn,mchi,delta,vmin):
    '''
    Calculates the minimum value of recoil energy for a target nuclei
    ---------------------------------------------------------------------------
    Input:

    mn: Nuclear mass in GeV
  
    mchi: WIMP mass in GeV

    delta: mass splitting for inelastic scattering in keV

    vmin: minimum WIMP velocity (in lab frame) required to deposit a recoil energy in km/s
    -----------------------------------------------------------------------------
    Output:

    minimum recoil energy in keV
    -----------------------------------------------------------------------------
    '''
    
    mu=mchi*mn/(mchi+mn)      
    vstar=get_vstar(mn,mchi,delta)
    vmin1=np.choose(vmin>vstar,[vstar,vmin])    
    return (mu/np.sqrt(2.*mn)*(vmin1-np.sqrt(vmin1**2-vstar**2))/300.)**2




def er_max(mn,mchi,delta,vmin):
    '''
    Calculates the maximum value of recoil energy for a target nuclei
    ---------------------------------------------------------------------------
    Input:

    mn: Nuclear mass in GeV

    mchi: WIMP mass in GeV

    delta: mass splitting for inelastic scattering, in keV

    vmin: minimum WIMP velocity (in lab frame) required to deposit a recoil energy in km/s
    ---------------------------------------------------------------------------
    Output:

    maximum recoil energy in keV
    -------------------------------------------------------------------------- 
    '''
    
    mu=mchi*mn/(mchi+mn)    
    return mu**2/mn*(vmin/300)**2-delta*mu/mn+(vmin/300)*mu**2/mn*np.sqrt((vmin/300)**2-2*delta/mu)

def er_min(mn,mchi,delta,vmin):
    '''
    Calculates the minimum value of recoil energy for a target nuclei
    ---------------------------------------------------------------------------
    Input:

    mn: Nuclear mass in GeV
  
    mchi: WIMP mass in GeV

    delta: mass splitting for inelastic scattering in keV

    vmin: minimum WIMP velocity (in lab frame) required to deposit a recoil energy in km/s
    -----------------------------------------------------------------------------
    Output:

    minimum recoil energy in keV
    -----------------------------------------------------------------------------
    '''
    
    mu=mchi*mn/(mchi+mn)      
    return mu**2/mn*(vmin/300)**2-delta*mu/mn-(vmin/300)*mu**2/mn*np.sqrt((vmin/300)**2-2*delta/mu)



def print_wimp_rate_contributions(exp,cont,bin_list=None):
    print('contributions available in wimp_dd_rate.contributions')
    if bin_list is None:
        bin_list=set([c[0] for c in list(cont.keys())])
    coeff_squared_list=set([c[1] for c in list(cont.keys())])
    for n_bin in bin_list:
        print(20*'-')        
        print('n_bin:'+str(n_bin))
        print('Contributions from squares of operators:')
        for __c1__,__c2__ in coeff_squared_list:
            print(__c1__,__c2__,cont[n_bin,(__c1__,__c2__)])
        print(20*'-')


def wimp_dd_rate(exp,hamiltonian,vmin,delta_eta,mchi,j_chi=0.5,delta=0,rho_loc=0.3,velocity_dependence=True,response_functions=None,energy_bins_vec=None, elements_list=None,isotopes_list=None,reset_response_functions=False,print_contributions=False,tau_range=range(2),tau_prime_range=range(2),sum_over_streams=True,verbose=False,**args):
    '''
    calculates the expected number of events in each energy bin defined in exp.data for the set of parameters 
    **args of the Wilson coefficients in the effective Hamiltonian.
    Calls load_response_functions if exp.response_functions does not contain the required response functions for hamiltonian and j_chi.
    --------------------------------------------------------------------------------------
    Input:

    exp: object belonging to experiment class

    hamiltonian: object belonging to eft_hamiltonian class

    vmin: array containing a list of WIMP stream speed velocities in the lab frame in km/s.

    delta_eta: array containing the contribution of each stream to the halo function eta(v) in (km/sec)^-1
               (the routine streamed_halo_function can be used to calculate it)

    mchi: WIMP mass in GeV

    delta (default value=0): Mass splitting for inelastic scattering in keV 

    j_chi (default value=0.5): WIMP spin

    rho_loc (default value=0.3): local dark matter density in units of GeV/cm^3 

    velocity_dependence (default=True): If False calculates the rate without including the terms of the the squared amplitude 
                                        with an explicit velocity dependence in the cross-section (n_vel=0, see help on eft_amplitude_squared,
                                        load_response_functions or experiment)

    response_functions (default=None): if passed, overrides the set of response functions. Must be a tuple with the same format of 
                                       exp.response_functions[hamiltonian,j_chi] 

    energy_bins_vec (default=None): if passed, a list or array to override the set of energy bins normally given by range(len(exp.data)) 

    reset_response_functions (Default: False): If True empties the dictionary exp.response_functions and reloads the response functions
                                               for hamiltonian and j_chi.

    print_contributions (default value: False): if True, prints out the contributions to the expected rate from each combination of 
                                                the Wilson coefficients. After each call to wimp_dd_rate such contributions can be access 
                                                via the dictionary:

                                                    wimp_dd_rate.contributions[n_bin,(ci,cj)] 
                         
                                                for the energy bin n_bin and any of the effective coupling combinations (ci,cj)
                                                contained in hamiltonian.coeff_squared_combinations
                                              
                                                  

    tau_range, tau_prime_range (defaul value: range(2)->[0,1]): overrides the values of tau, tau_prime in the double sum over nuclear isospins 

                                                                   sum_{tau} sum_{tau_prime} R^{tau tau_prime}W^{tau tau_prime} 

                                                                  that enters the calculation of the rate. It allows to calculate the 
                                                                  contribution to the rate of specific isospin combinations.


    sum_over_streams(default: True): if False does not sum over velocity streams and returns for each energy bin the contributions                              
    to the rate from each stream, in an array with shape energy.shape+vmin.shape.

    verbose (default value: False): Passed to load_response_functions, if called.
                                    
    **args: passes the parameters of the Wilson coefficients defined in hamiltonian (see hamiltonian.global_arguments). 
            Parameters that are not passed are set to their default values, if defined (see hamiltonian.global_defaul_args), otherwise 
            to 1.  
    ---------------------------------------------------------------------------------------
    Output:

    Two arrays contaning the centers of the bins contained in data.tab and the predicted number of events in each energy bin 
    If a line of the data.tab file contains a single energy value returns the corresponding differential rate 
    (see help on data.dat in experiment class)
    ---------------------------------------------------------------------------------------
    Example:

    dama=experiment('dama') 

    12 bins of width 0.5 keV:
    [(x[0],x[1]) for x in damaz.data] -> [(2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 5.0), 
                                         (5.0, 5.5), (5.5, 6.0), (6.0, 6.5), (6.5, 7.0), (7.0, 7.5), (7.5, 8.0)]

    dama.exposure -> 2 (divided by the common bin width to get exposure equal to 1kg*day*keV, 
                        output of wimp_dd_rate in events/kg/day/keV)

    o1=eft_hamiltonian('o1',{1: lambda M, r=1: [(1+r)/M**2,(1-r)/M**2]})   

    load_response_functions(dama,o1,0.5) (load response functions for WIMP spin=0.5)

    vmin,delta_eta0=streamed_halo_function()  (time-independent component of halo function for Maxwellian with default 
                                               values of parameters - see help on streamed_halo_function)
    
    e,r0=WD.wimp_dd_rate(dama,o1,vmin,delta_eta0,mchi=100,M=1e3)
    
    e-> vector with centers of energy bins in data.tab
    r0-> vector with expected integrated rate in each energy bin for mchi=100, M=1e3,r=1

    vmin,delta_eta1=streamed_halo_function(yearly_modulation=True)  (yearly-modulated component of halo function for Maxwellian with default 
                                                                      values of parameters - see help on streamed_halo_function)
    
    e,r1=WD.wimp_dd_rate(dama,o1,vmin,delta_eta1,mchi=100,M=1e3)

    r1-> vector with modulated amplitudes in each energy bin for mchi=100, M=1e3,r=1
    ---------------------------------------------------------------------------------------
    '''
    # converts elements_list
    
    flatten=False
    
    # puts to one or to their default values all arguments not used in the function call
    input_args={}

    input_args.update({k:1 for k in {e for e in hamiltonian.global_arguments}-{t for t in list(args.keys())} if k not in list(hamiltonian.global_default_args.keys()) and k!='mchi' and k!='delta'})


    
    input_args.update({k:v for k,v in hamiltonian.global_default_args.items() if k not in list(args.keys())})

    input_args.update(args)

    
    
    if not flatten:
        response_function_index=hamiltonian,j_chi
        if response_function_index in list(exp.response_functions.keys()) and reset_response_functions:
            del exp.response_functions[hamiltonian,j_chi]
        
        coeff_squared_list=hamiltonian.coeff_squared_list

        if  response_function_index not in list(exp.response_functions.keys()):
            if verbose:
                print('response functions not available for')
                print(hamiltonian)
                print(' and j_chi='+str(j_chi))
            load_response_functions(exp,hamiltonian,reset=reset_response_functions,j_chi=j_chi,verbose=verbose)
            #return np.zeros(len(exp.data))
    else:
        if reset_response_functions:
            del exp.response_functions[hamiltonian,j_chi,'flatten']
        
        coeff_squared_list=[(0,0)]        
        response_function_index=hamiltonian,j_chi,'flatten'
        tau_range=range(1)
        tau_prime_range=range(1)        

        if response_function_index not in list(exp.response_functions.keys()):
            print('response functions not available for')
            print('flattened '+hamiltonian.name)
            print(' with j_chi='+str(j_chi))

            
            
            load_flattened_response_functions(exp,hamiltonian,mchi,delta,j_chi,**input_args)
            #return np.zeros(len(exp.data))
        
    
    # initialize this routine calling custom_coefficients() and load_custom_response_functions(exp)

    if energy_bins_vec is None:
        energy_bins_vec=range(len(exp.data))

    if elements_list is None:
        element_list=range(exp.target.n_targets)

    else:
        element_list=[n for n in range(exp.target.n_targets) if exp.target.element[n].name in [e.name for e in elements_list]]


        
    isotopes_list1={n:range(exp.target.element[n].n_isotopes) for n in range(exp.target.n_targets)}

    if isotopes_list is not None:
        for key in isotopes_list.keys():
            isotopes_list1[key]=isotopes_list[key]

    vmin_is_float=False
    vmin_is_vector=False

    try:
        if (len(vmin.shape))==1:
            vmin_is_vector=True

    except:
        vmin_is_float=True
        vmin=np.array([vmin]).reshape(1,1)
        delta_eta=np.array([delta_eta]).reshape(1,1)

    '''
    len_test=len(vmin.shape)

    if len_test==1:
        
        vmin=vmin.reshape(1,vmin.shape[0])

    len_test=len(delta_eta.shape)
    if len_test==1:
        
        delta_eta=delta_eta.reshape(1,delta_eta.shape[0])
    '''
    
    c_light=3.e5
    hbarc2=0.389e-27 # in GeV^2 cm^2
    
    sm_out=np.array([])                                                                      
    mu_p=1./(1./0.931+1./mchi)

    contributions={}
    for n_bin in energy_bins_vec:                                                       
        sm_bin=0.

        
        for n_coeff_squared,(__c1__,__c2__) in enumerate(coeff_squared_list):

            if flatten:
                c1_c2_arguments=np.array([])
            else:
                c1_c2_arguments=np.unique(np.array([e for t in [hamiltonian.arguments[__c1__],hamiltonian.arguments[__c2__]] for e in t]))
            
            args_c1_c2={}
            for arg in c1_c2_arguments:
                if arg!='mchi' and arg!='delta':
                    args_c1_c2[arg]=input_args[arg]
            
            if 'mchi' in c1_c2_arguments: args_c1_c2['mchi']=mchi
            if 'delta' in c1_c2_arguments: args_c1_c2['delta']=delta
            
            coeff_contribution=0
            for n_element in element_list:                                       
                for n_isotope in range(exp.target.element[n_element].n_isotopes):                
                                                                                             
                    mn=exp.target.element[n_element].mass[n_isotope]                             
                    vstar=get_vstar(mn,mchi,delta)                                            
                    estar=get_estar(mn,mchi,delta)                                            

                    vmin_larger_than_vstar=np.choose(vmin.transpose()>vstar,[vstar,vmin.transpose()]).transpose()

                    delta_eta_larger_than_vstar=np.choose(vmin.transpose()>vstar,[0,delta_eta.transpose()]).transpose()


                    mu_n=mchi*mn/(mchi+mn)                          

                    one=np.ones(np.prod(vmin.shape)).reshape(*vmin.shape)
                    p_vdep=np.array([one, vmin_larger_than_vstar**2/c_light**2-delta*1e-6/mu_n*one, -mn/(2.*mu_n**2)*one, -(delta*1e-6)**2/(2.*mn)*one])
                    #one=np.ones(vmin.shape[0]*vmin.shape[1]).reshape(vmin.shape[0],vmin.shape[1])
                    #ones=np.ones(vmin.shape[1]).reshape(1,-1)
                
                    #p_vdep=np.array([one, vmin_larger_than_vstar**2/c_light**2-delta*1e-6/mu_n*ones, -mn/(2.*mu_n**2)*ones, -(delta*1e-6)**2/(2.*mn)*ones])

                    
                    er_1=er_min(mn,mchi,delta,vmin_larger_than_vstar)               
                    er_2=er_max(mn,mchi,delta,vmin_larger_than_vstar)               

                
                    if velocity_dependence:
                        n_vel_max=4
                    else:
                        n_vel_max=1                        
                    
                    for n_vel in range(n_vel_max):

                        for tau in tau_range:
                            for tau_prime in tau_prime_range:
                                coeff_contribution+=\
                                hbarc2*9.e20*86400./1.e5*rho_loc/mchi*\
                                hamiltonian.coeff_squared(__c1__,__c2__,**args_c1_c2)[tau,tau_prime]*p_vdep[n_vel]*\
                                np.choose(vmin>vstar.reshape(-1,1),[0.,np.transpose(np.transpose(\
                                (np.interp(er_2,exp.response_functions[response_function_index][n_coeff_squared][n_vel][tau][tau_prime][n_bin][n_element][0],\
                                exp.response_functions[response_function_index][n_coeff_squared][n_vel][tau][tau_prime][n_bin][n_element][n_isotope+1])-\
                                np.interp(er_1,exp.response_functions[response_function_index][n_coeff_squared][n_vel][tau][tau_prime][n_bin][n_element][0],\
                                exp.response_functions[response_function_index][n_coeff_squared][n_vel][tau][tau_prime][n_bin][n_element][n_isotope+1]))*delta_eta_larger_than_vstar)*1./np.pi)])
                                #*delta_eta)*1./mu_p**2)]),1)

            if sum_over_streams:
                coeff_contribution=np.sum(coeff_contribution,-1)
            contributions[n_bin,(__c1__,__c2__)]=coeff_contribution

            # sums over contributions 
            sm_bin+=coeff_contribution
            
        sm_out=np.append(sm_out,sm_bin).reshape(-1,*sm_bin.shape)

    wimp_dd_rate.contributions=contributions
    if print_contributions:
        print_wimp_rate_contributions(exp,contributions,bin_list=energy_bins_vec)

    energy_values=np.array([(exp.data[i][0]+exp.data[i][1])/2. if len(exp.data[i])>1 else exp.data[i] for i in energy_bins_vec])

    #rate=np.transpose(sm_out.reshape(len(energy_bins_vec),delta_eta.shape[0]))[0]

    if vmin_is_float or vmin_is_vector and sum_over_streams:
        sm_out=sm_out.reshape(len(exp.data),)
    if vmin_is_vector and not sum_over_streams:
        sm_out=sm_out.reshape(len(exp.data),len(vmin))

    return energy_values,sm_out#rate







def plot_flattened_diff_response_functions(exp,eft_hamiltonian,mchi,delta,j_chi=0.5,er=None,eprime=None,e_min=None,e_max=None,coeff_squared_list=None,element_range=None,tau_range=[0,1],tau_prime_range=[0,1],n_vel_range=range(4),frac=0.5,n_points=100,style='',linewidth=1,**args):

    coeff_squared_args={k:v for k,v in args.items()}
    
    if 'mchi' in eft_hamiltonian.global_arguments:
        coeff_squared_args['mchi']=mchi
    if 'delta' in eft_hamiltonian.global_arguments:
        coeff_squared_args['delta']=delta


        
    if er is None and eprime is None:
        print('Nothing plotted')
        print('Need to fix er or eprime value.')        
        print('Provide e_min,e_max and:')
        print('- er to plot G(er,prime) for e_min<eprime<e_max')
        print('- eprime to plot G(er,prime) for e_min<er<e_max')

        return

    response_function_index=eft_hamiltonian,j_chi
    if response_function_index not in [k[:2] for k in exp.diff_response_functions.keys()]:
        load_response_functions(exp,eft_hamiltonian,j_chi)

    
    response_function_index=eft_hamiltonian,j_chi,'flatten'

    if response_function_index not in [k[:3] for k in exp.diff_response_functions.keys()]:
        print('response functions not available for')
        print('flattened '+eft_hamiltonian.name)
        print(' with j_chi='+str(j_chi))
        load_flattened_response_functions(exp,eft_hamiltonian,mchi,delta,j_chi,tau_range=tau_range,tau_prime_range=tau_prime_range,**args)
    
    if coeff_squared_list is None:
        coeff_squared_list=eft_hamiltonian.coeff_squared_list

    n_elements=exp.target.n_targets        
    if element_range is None:
        element_range=range(n_elements)

    isotope_range=()
    for n_element in range(n_elements):
        isotope_range+=(range(exp.target.element[n_element].n_isotopes),)

    n_plotted=0
    n_plotted_flattened=0    
    for n_element in element_range:
        for n_isotope in isotope_range[n_element]:
            for n_vel in n_vel_range:
                if er is None:
                    er0=op.bisect(lambda er,eprime:exp.target.element[n_element].quenching(er)*er-eprime,1e-3,1e3, eprime)
                        
                    if e_min is None:
                        e_min=er0*(1.-frac)                            

                    if e_max is None:
                        e_max=er0*(1.+frac)                        

                    e_vec=np.linspace(e_min,e_max,n_points)

                    r_sum=np.zeros(n_points)                    
                    for tau,tau_prime in [(x,y) for x in tau_range for y in tau_prime_range]:                                        
                    
                        r=lambda er: exp.diff_response_functions[eft_hamiltonian.name, j_chi, 'flattened', exp.target.element[n_element].name][tau,tau_prime,n_isotope,n_vel](er,eprime)
                                 
                        
                        r_vec=[r(er) for er in e_vec]

                        if max(abs(np.array(r_vec)))>0:
                        
                            n_plotted_flattened+=1
                            pl.plot(e_vec,r_vec,'-k',linewidth=1)



                        for __c1__,__c2__ in coeff_squared_list:

                            r=lambda er: exp.diff_response_functions[eft_hamiltonian.name, j_chi, __c1__, __c2__, exp.target.element[n_element].name][tau,tau_prime,n_isotope,n_vel](er,eprime)

                            r_vec=[r(er) for er in e_vec]
                            if max(abs(np.array(r_vec)))>0:
                                n_plotted+=1                                
                                pl.plot(e_vec,r_vec,':r',linewidth=2) 


                            
                            r_sum+=eft_hamiltonian.coeff_squared(__c1__,__c2__,**coeff_squared_args)[tau,tau_prime]*np.array(r_vec)
                            
                    if max(abs(np.array(r_sum)))>0:
                        pl.plot(e_vec,r_sum,':b',linewidth=2)

                            
                elif eprime is None:

                    e_ee=exp.target.element[n_element].quenching(er)*er
                    
                        
                    if e_min is None:
                        e_min=e_ee*(1.-frac)                            

                    if e_max is None:
                        e_max=e_ee*(1.+frac)                        

                    e_vec=np.linspace(e_min,e_max,n_points)
                    
                    r_sum=np.zeros(n_points)
                    for tau,tau_prime in [(x,y) for x in tau_range for y in tau_prime_range]:                    
                        r=lambda eprime: exp.diff_response_functions[eft_hamiltonian.name, j_chi, 'flattened', exp.target.element[n_element].name][tau,tau_prime,n_isotope,n_vel](er,eprime)
                                 

                        r_vec=[r(er) for er in e_vec]
                        if max(abs(np.array(r_vec)))>0:

                            
                            n_plotted_flattened+=1
                            pl.plot(e_vec,r_vec,'-k',linewidth=1)
                        
                    


                        for __c1__,__c2__ in coeff_squared_list:

                            
                            r=lambda eprime: exp.diff_response_functions[eft_hamiltonian.name, j_chi, __c1__, __c2__, exp.target.element[n_element].name][tau,tau_prime,n_isotope,n_vel](er,eprime)                        
                            
                            r_vec=[r(eprime) for eprime in e_vec]
                            
                            if max(abs(np.array(r_vec)))>0:
                                n_plotted+=1
                                pl.plot(e_vec,r_vec,':r',linewidth=2) 
                            

                            r_sum+=eft_hamiltonian.coeff_squared(__c1__,__c2__,**coeff_squared_args)[tau,tau_prime]*np.array(r_vec)
                            

                    if max(abs(np.array(r_sum)))>0:                        
                        pl.plot(e_vec,r_sum,':b',linewidth=2) 
    print(str(n_plotted)+' non-vanishing differential response functions plotted')
    print(str(n_plotted_flattened)+' non-vanishing flattened differential response functions plotted')    
    

        
def plot_diff_response_functions(exp,eft_hamiltonian,j_chi=0.5,er=None,eprime=None,e_min=None,e_max=None,coeff_squared_list=None,element_range=None,tau_range=[0,1],tau_prime_range=[0,1],n_vel_range=range(4),frac=0.5,n_points=100,style='',linewidth=1):
    
    if er is None and eprime is None:
        print('Nothing plotted')
        print('Need to fix er or eprime value.')        
        print('Provide e_min,e_max and:')
        print('- er to plot G(er,prime) for e_min<eprime<e_max')
        print('- eprime to plot G(er,prime) for e_min<er<e_max')

        return


    if coeff_squared_list is None:
        coeff_squared_list=eft_hamiltonian.coeff_squared_list

    n_elements=exp.target.n_targets        
    if element_range is None:
        element_range=range(n_elements)

    isotope_range=()
    for n_element in range(n_elements):
        isotope_range+=(range(exp.target.element[n_element].n_isotopes),)

    for n_element in element_range:

        for __c1__,__c2__ in coeff_squared_list:

            if not (eft_hamiltonian.name, j_chi, __c1__, __c2__, exp.target.element[n_element].name) in exp.diff_response_functions.keys():
                print('Use load_response_functions to load the differential response functions for '+exp.name+', '+eft_hamiltonian.name+' and j_chi='+str(j_chi)+' (you do not need to calculate the integrated response functions if missing)')
                return None
            



        
    n_plotted=0
    for n_element in element_range:
        for n_isotope in isotope_range[n_element]:
            for n_vel in n_vel_range:
                if er is None:
            
                    er0=op.bisect(lambda er,eprime:exp.target.element[n_element].quenching(er)*er-eprime,1e-3,1e3, eprime)
                        
                    if e_min is None:
                        e_min=er0*(1.-frac)                            

                    if e_max is None:
                        e_max=er0*(1.+frac)                        


                    e_vec=np.linspace(e_min,e_max,n_points)

                    
                    for __c1__,__c2__ in coeff_squared_list:
                        for tau,tau_prime in [(x,y) for x in tau_range for y in tau_prime_range]:

                            
                            r=lambda er: exp.diff_response_functions[eft_hamiltonian.name, j_chi, __c1__, __c2__, exp.target.element[n_element].name][tau,tau_prime,n_isotope,n_vel](er,eprime)

                            

                            r_vec=[r(er) for er in e_vec]

                            if max(abs(np.array(r_vec)))>0:
                                n_plotted+=1
                                pl.plot(e_vec,r_vec,style,linewidth=linewidth)   
                            
                            
                elif eprime is None:

                    e_ee=exp.target.element[n_element].quenching(er)*er
                    
                        
                    if e_min is None:
                        e_min=e_ee*(1.-frac)                            

                    if e_max is None:
                        e_max=e_ee*(1.+frac)                        

                    e_vec=np.linspace(e_min,e_max,n_points)

                    for __c1__,__c2__ in coeff_squared_list:
                        for tau,tau_prime in [(x,y) for x in tau_range for y in tau_prime_range]:
                            
                            r=lambda eprime: exp.diff_response_functions[eft_hamiltonian.name, j_chi, __c1__, __c2__, exp.target.element[n_element].name][tau,tau_prime,n_isotope,n_vel](er,eprime)
                        
                            
                            r_vec=[r(eprime) for eprime in e_vec]

                            if max(abs(np.array(r_vec)))>0:
                                n_plotted+=1
                                pl.plot(e_vec,r_vec,style,linewidth=linewidth)   

    print(str(n_plotted)+' non-vanishing differential response functions plotted')    

                            
                    
                        

def plot_response_functions(exp,eft_hamiltonian,j_chi=0.5,coeff_squared_list=None,style='',linewidth=1,filename_suffix='',tuple=None,n_vel_list=None,tau_list=None,tau_prime_list=None,n_bin_list=None,n_element_list=None,n_isotope_list=None,rescaling_factor=1.,scatter_plot=False, scatter_plot_point_size=20, scatter_plot_color=None):
    '''
    Plots the response functions contained in exp.response_functions[eft_hamiltonian,j_chi]
    ----------------------------------------------------------------------------
    Input:

    exp: object belonging to experiment class

    eft_hamiltonian: object belonging to eft_hamiltonian class

    j_chi: WIMP spin


    coeff_squared_list (default: None): A list of the Wilson coefficients combinations for which the response functions are plotted 
                                        (among those in hamiltonian.coeff_squared_list).  
                                        By default all are plotted. 
                                        Example: [(('Omega', 0, 0), ('Omega', 0, 0))]
                                                 [(4,4), (6,6)]

    response_functions (default: None): overrides exp.response_functions(hamiltonian,j_chi).

    rescaling_factor: factor to rescale response functions (default=1)


    Example: if o4_o6.coeff_squared_list -> [(4, 4), (4, 6), (6, 4), (6, 6)], 
    
    plot_response_functions(xenon_2018,o4_o6,j_chi=0.5)

    Plots 16 non-vanishing response functions out of 576 

    plot_response_functions(xenon_2018,o4_o6,n_vel_list=[0],tau_list=[0],tau_prime_list=[1],n_bin_list=[0],
                                        n_element_list=[0],n_isotope_list=([3],),coeff_squared_list=[[6,6]])
    
    plots a single response function for c_6,c_6, n_vel=0 (velocity-independent term in squared amplitude), 
    tau=0, tau_prime=1, first energy bin, first (and only) xenon target, 4th isotope (xenon_2018.target.element[0].isotopes[3]-> '129Xe'


    ----------------------------------------------------------------------------------
    Output:

    Figure showing response functions for defined experiment exp as a function of recoil energy
    -----------------------------------------------------------------------------------
    '''
    #tuple=response_functions
    flatten=False
    
    if coeff_squared_list is None:
        coeff_squared_list=eft_hamiltonian.coeff_squared_list
    
    if not flatten:
        response_function_index=eft_hamiltonian,j_chi
    else:
        coeff_squared_list=[(0,0)]        
        response_function_index=eft_hamiltonian,j_chi,'flatten'

    if n_vel_list is None:
        n_vel_list=range(4)
    if tau_list is None:        
        tau_list=range(2)
    if tau_prime_list is None:                
        tau_prime_list=range(2)
    if n_bin_list is None:                        
        n_data_points=len(exp.data)
        n_bin_list=range(n_data_points)

    if n_element_list is None:                                
        n_elements=exp.target.n_targets
        n_element_list=range(n_elements)

    if n_isotope_list is None:
        
        n_isotope_list=()
        for n_element in n_element_list:
            n_isotope_list+=(range(exp.target.element[n_element].n_isotopes),)
            
    try:
        
        tuple_list=exp.response_functions[response_function_index]
        
    except:
        if not flatten:
            print(exp.name+' response functions for '+eft_hamiltonian.name+' and spin '+str(j_chi)+':')
            print(eft_hamiltonian)
            print('are not present in '+eft_hamiltonian.name+'.response_functions. Do you want to load/calculate them?')
            answer=input('No (0); Yes (1)? ')
            if int(answer)==1:
                load_response_functions(exp,eft_hamiltonian,j_chi,reset=False)            
                tuple_list=exp.response_functions[response_function_index]
            else:
                print('Response functions not avalable. Nothing is plotted')
                return
        else:
            print(exp.name+' response functions for '+eft_hamiltonian.name+', spin '+str(j_chi)+' and flattened:')
            print(eft_hamiltonian)
            print('are not available. Do you want to calculate them?')
            answer=input('No (0); Yes (1)? ')
            if int(answer)==1:
                load_flattened_response_functions(exp,eft_hamiltonian,j_chi)            
                tuple_list=exp.response_functions[response_function_index]
            else:
                print('Response functions not avalable. Nothing is plotted')
                return


    ntot=0
    n_non_vanishing=0
    for __c1__,__c2__ in coeff_squared_list:
        if (__c1__,__c2__) in eft_hamiltonian.coeff_squared_list:
            n=eft_hamiltonian.coeff_squared_list.index((__c1__,__c2__))
            if tuple is None:
                tuple=tuple_list[n]

            for n_vel in n_vel_list:
                for n_bin in n_bin_list:
                    for n_element in n_element_list:
                        for n_isotope in n_isotope_list[n_element]:

                            for tau in tau_list:
                                for tau_prime in tau_prime_list:
                                    er=tuple[n_vel][tau][tau_prime][n_bin][n_element][0]
                                    r=tuple[n_vel][tau][tau_prime][n_bin][n_element][n_isotope+1]
                                    ntot+=1
                                    
                                    if max(abs(r))!=0:
                                        n_non_vanishing+=1
                                    
                                        if scatter_plot:
                                            
                                            pl.scatter(er,rescaling_factor*r,s=scatter_plot_point_size,color=scatter_plot_color)
                                        else:
                                            pl.plot(er,rescaling_factor*r,style,linewidth=linewidth)

        else:
            print('Squared coupling '+str((__c1__,__c2__))+' is not present in '+eft_hamiltonian.name)
    print('total of '+str(ntot)+' response functions')
    print(n_non_vanishing,' non-vanishing response functions plotted')
            
def response_functions_list(exp):
    '''
    Lists response functions for defined experiment exp
    -------------------------------------------------------------------
    Input:

    exp: object belonging to experiment class
    --------------------------------------------------------------------
    Output:

    information about the loaded response functions for defined experiment exp
    --------------------------------------------------------------------------
    '''
    
    if exp.response_functions:
        print('Currently loaded response functions for '+str(exp.name)+':')
        for m,spin in list(exp.response_functions.keys()):
            print(40*'=')
            print('WIMP spin:'+str(spin))                                    
            print(m.name+' EFT Hamiltonian: '+m.print_hamiltonian())
            print(40*'=')            
    else:
            print('No response functions are currently loaded for '+exp.name+'. Load them using load_response_functions(exp,eft_hamiltonian,j_chi).')


            
def collapse_wilson_coefficients_old(hamiltonian,mchi,delta,**args):
    output={}
    global_arguments=np.unique(np.array([e for t in list(hamiltonian.arguments.values()) for e in t]))
    args.update({k:1 for k in {e for e in global_arguments}-{t for t in list(args.keys())} if k!='q'})

    if 'mchi' in global_arguments: args['mchi']=mchi
    if 'delta' in global_arguments: args['delta']=delta

    
    collapsed_wilson_coefficients={}
    for __c1__ in set([get_short_coupling(convert_to_all_spins(c)) for c in list(hamiltonian.wilson_coefficients.keys())]):
        coeff_list=[]
        for __c2__ in list(hamiltonian.wilson_coefficients.keys()):
            if get_short_coupling(convert_to_all_spins(__c2__))==__c1__:
                args_c={}

                for arg in hamiltonian.arguments[__c2__]:
                    if arg!='q':
                        args_c[arg]=args[arg]

                if 'mchi' in hamiltonian.arguments[__c2__]: args_c['mchi']=mchi
                if 'delta' in hamiltonian.arguments[__c2__]: args_c['delta']=delta

                if 'q' in hamiltonian.arguments[__c2__]:

                    sign=1
                    if __c2__!=convert_to_all_spins(__c2__):
                        sign=haxton_to_all_spins_sign(all_spins_to_haxton[convert_to_all_spins(get_short_coupling(__c2__))])
                        
                    def coeff_func(q,__c2__=__c2__,args1=args_c,sign=sign):
                        args1['q']=q
                        
                        return sign*np.array(hamiltonian.wilson_coefficients[__c2__](**args1))
                    
                    coeff_list.append(coeff_func)
                else:

                    sign=1
                    if __c2__!=convert_to_all_spins(__c2__):
                        sign=haxton_to_all_spins_sign(all_spins_to_haxton[convert_to_all_spins(get_short_coupling(__c2__))])
                    
                    coeff_list.append(lambda q, __c2__=__c2__, args1=args_c, sign=sign:  sign*np.array(hamiltonian.wilson_coefficients[__c2__](**args1)))
                    
        collapsed_wilson_coefficients[__c1__]=coeff_list


    
    for c in set([get_short_coupling(convert_to_all_spins(c)) for c in hamiltonian.couplings]):
        output[c]=lambda q, coeff_list=collapsed_wilson_coefficients[c]: sum(np.array([f(q) for f in coeff_list]))

        
    return output



def collapse_wilson_coefficients(hamiltonian,mchi,delta,tau_range=[0,1],**args):
    output={}
    global_arguments=np.unique(np.array([e for t in list(hamiltonian.arguments.values()) for e in t]))
    args.update({k:1 for k in {e for e in global_arguments}-{t for t in list(args.keys())} if k!='q'})

    if 'mchi' in global_arguments: args['mchi']=mchi
    if 'delta' in global_arguments: args['delta']=delta

    
    collapsed_wilson_coefficients={}
    for __c1__ in set([get_short_coupling(convert_to_all_spins(c)) for c in list(hamiltonian.wilson_coefficients.keys())]):
        coeff_list=[]
        for __c2__ in list(hamiltonian.wilson_coefficients.keys()):
            if get_short_coupling(convert_to_all_spins(__c2__))==__c1__:
                args_c={}

                for arg in hamiltonian.arguments[__c2__]:
                    if arg!='q':
                        args_c[arg]=args[arg]

                if 'mchi' in hamiltonian.arguments[__c2__]: args_c['mchi']=mchi
                if 'delta' in hamiltonian.arguments[__c2__]: args_c['delta']=delta

                if 'q' in hamiltonian.arguments[__c2__]:

                    sign=1
                    if __c2__!=convert_to_all_spins(__c2__):
                        sign=haxton_to_all_spins_sign(all_spins_to_haxton[convert_to_all_spins(get_short_coupling(__c2__))])
                        
                    def coeff_func(q,__c2__=__c2__,args1=args_c,sign=sign,tau_range=tau_range):
                        args1['q']=q
                        out=[0,0]
                        for tau in tau_range:
                            out[tau]=sign*hamiltonian.wilson_coefficients[__c2__](**args1)[tau]
                        return out
                    
                    coeff_list.append(coeff_func)
                else:
                    
                    sign=1
                    if __c2__!=convert_to_all_spins(__c2__):
                        sign=haxton_to_all_spins_sign(all_spins_to_haxton[convert_to_all_spins(get_short_coupling(__c2__))])

                    def coeff_func(q,__c2__=__c2__,args1=args_c,sign=sign,tau_range=tau_range):
                        out=[0,0]
                        for tau in tau_range:
                            out[tau]=sign*hamiltonian.wilson_coefficients[__c2__](**args1)[tau]
                        return out

                    coeff_list.append(coeff_func)                    
                        
                    
        collapsed_wilson_coefficients[__c1__]=coeff_list


    
    for c in set([get_short_coupling(convert_to_all_spins(c)) for c in hamiltonian.couplings]):
        output[c]=lambda q, coeff_list=collapsed_wilson_coefficients[c]: sum(np.array([f(q) for f in coeff_list]))

        
    return output



            

def load_flattened_response_functions(exp,hamiltonian,mchi,delta,j_chi=0.5,outputfile=None,verbose=True,tau_range=[0,1],tau_prime_range=[0,1],**args):
    '''
    Loads the attened response functions for defined experiment exp
    -----------------------------------------------------------------
    Input:
 
    exp: object belonging to experiment class

    hamiltonian: object belonging to eft_hamiltonian class

    mchi: WIMP mass in GeV

    delta: mass splitting for inelastic scattering in keV (default value=0)

    j_chi: WIMP spin (default value=0.5)

    outputfile: If None, creates output file in npy format

    verbose(bool): If True prints out details of accessed external files (default=True)

    **args: passes the parameters of the Wilson coefficients defined in hamiltonian (see hamiltonian.global_arguments). 
            Parameters that are not passed are set to their default values, if defined (see hamiltonian.global_defaul_args), otherwise 
            to 1.
    -------------------------------------------------------------------------
    Output:

    A dictionary of response functions which could be accessed by 
    exp.response_functions[hamiltonian,j_chi,'flatten']
    --------------------------------------------------------------------------
    '''
    
    if outputfile is None:
        outputfile=hamiltonian.name+'_j_chi_'+str(j_chi)+'_flattened.npy'
    #response_functions=get_response_functions_interf(exp,n_coeff1=None,coeff1_input=collapse_wilson_coefficients(hamiltonian,mchi,delta,**args),j_chi=j_chi,outputfile=outputfile,verbose=verbose)
    response_functions=get_response_functions_interf(exp,n_coeff1=None,coeff1_input=collapse_wilson_coefficients(hamiltonian,mchi,delta,tau_range=tau_range,**args),coeff2_input=collapse_wilson_coefficients(hamiltonian,mchi,delta,tau_range=tau_prime_range,**args),j_chi=j_chi,outputfile=outputfile,verbose=verbose)


    for element in exp.target.element:
        exp.diff_response_functions[hamiltonian.name,j_chi,'flattened',element.name]=element.response_function_interf


            



def get_vmin(er,mn,mchi,delta):
    '''
    Calculates minimal velocity in lab frame which is required to deposit a given 
    recoil energy in the detector
    -----------------------------------------------------------------------------
    Input:

    er: nuclear recoil energy in keV

    mn: Nuclear mass in GeV

    mchi: WIMP mass in GeV

    delta: mass splitting for inelastic scattering, in keV
    -----------------------------------------------------------------------------
    Output:

    minimum velocity in the units of km/s
    -----------------------------------------------------------------------------
    '''
    
    mu=mchi*mn/(mchi+mn)

    return 1.0/np.sqrt(2.0*mn*er)*abs(mn*er/mu+delta)*300.




def dsigma_der(element,hamiltonian,mchi,v,er,j_chi=0.5,delta=0,velocity_dependence=True,**args):
    '''
    Calculates the differential cross section for WIMP-nucleus scattering.
    ---------------------------------------------------------------------
    Input:

    element: object belonging element class

    hamiltonian: object belonging to eft_hamiltonian class

    mchi: WIMP mass in GeV

    v: WIMP incoming velocity in km/s.

    er: nuclear recoil energy in keV

    delta (default value=0): mass splitting for inelastic scattering, in keV
    
    j_chi (default:0.5): WIMP spin

    velocity_dependence: if False the velocity-dependent part of the 
    the cross-section is neglected (default=True)
    
    **args: passes the parameters of the Wilson coefficients defined in hamiltonian (see hamiltonian.global_arguments). 
            Parameters that are not passed are set to their default values, if defined (see hamiltonian.global_defaul_args), otherwise 
            to 1.

    --------------------------------------------------------- 
    output:

    An array containing the differential cross section normalized to cm^2/keV for each of 
    the ni isotopes contained in element.isotopes. If v is a one-dimensional array 
    of nv values the shape of the output is (ni,nv)

    Example:

    If o4_o6.global_arguments -> ['M', 'q', 'r']

    xe=element('Xe')
    xe.isotopes->['124Xe', '126Xe', '128Xe', '129Xe', '130Xe', '131Xe', '132Xe',
       '134Xe', '136Xe'] (9 isotopes)  (9 isotopes)

    v=[200,300,400,500] (4 velocity values)

    diff_sigma==dsigma_der(xe,o4_o6,mchi=100,j_chi=0.5,delta=0,v=[200,300,400],er=10,r=0.5,M=1e3)
    
    diff_sigma.shape->(9, 3)

    calculates the differential cross section off the 9 xenon isotopes for 4 values of the WIMP 
    incoming speed, WIMP mass 100 GeV, elastic scattering (vanishing mass splitting), nuclear 
    recoil energy 10 keV and fixing the two parameters of the effective Hamiltonian to r=10 and M=1e3. 
    --------------------------------------------------------- 
    '''

    ### if v is not iterable put v in np.array([v])
    try:
        iterator=iter(v)
        if not isinstance(v,np.ndarray):
            v=np.array(v)
    except:
        v=np.array([v])
        
    #if not isinstance(v,np.ndarray):
    #    if isinstance(v, int) or isinstance(v,float):
    #        v=np.array([v])
    #    elif isinstance(v,list):
    #        v=np.array(v)


    # puts to one or to their default values all arguments not used in the function call
    input_args={}

    input_args.update({key:1 for key in {e for e in hamiltonian.global_arguments}-{t for t in list(args.keys())} if key not in list(hamiltonian.global_default_args.keys()) and key!='mchi' and key!='delta'})

    input_args.update({key:value for key,value in hamiltonian.global_default_args.items() if key not in list(args.keys())})

    input_args.update(args)

    c_coeff_vec=collapse_wilson_coefficients(hamiltonian,mchi=mchi,delta=delta,**input_args)

    
    dsigma_der_output=np.array([])

    c_light=3.e5 # in km/sec
    hbarc2=0.389e-27 # in GeV^2 cm^2
    
    for isotope in range(len(element.isotopes)): 
        a_nucleus=element.a[isotope]
        m_nucleus=a_nucleus*0.931
        q=np.sqrt(2.*m_nucleus*er*1e-6)# in GeV
        vmin=get_vmin(er,m_nucleus,mchi,delta)

        eft0=eft_amplitude_squared(c_coeff_vec,q,element,isotope,j_chi=j_chi)

        eft1=0
        if velocity_dependence:
            eft1=eft_amplitude_squared1(c_coeff_vec,q,element,isotope,j_chi=j_chi)


        eft=np.choose(v>vmin,[0,eft0+eft1*(v**2-vmin**2)/c_light**2])

        dsigma_der_isotope=hbarc2*10**(-6)*2*m_nucleus/(4.*np.pi)*\
                            c_light**2/v**2*eft

        if len(dsigma_der_isotope)==1:
            dsigma_der_isotope=dsigma_der_isotope[0]
    
            dsigma_der_output=np.append(dsigma_der_output,dsigma_der_isotope)
        else:
            dsigma_der_output=np.append(dsigma_der_output,dsigma_der_isotope).reshape(-1,*v.shape)


    return dsigma_der_output



def diff_rate_old(target,hamiltonian,mchi,j_chi,er,vmin,delta_eta,delta=0.,exposure=1.,rho=0.3,element_list=None,**args):
    '''
    Calculates the differential rate for WIMP-nucleus scattering
    -------------------------------------------------------------------
    Input:

    target: object belonging to target class

    hamiltonian: object belonging to eft_hamiltonian class mchi: WIMP mass in GeV

    j_chi: WIMP spin

    er: nuclear recoil energy in keV

    vmin: array with values of vmin in km/sec where delta_eta is sampled

    delta_eta: array containing the contributions of streames vmin to the halo function in (km/sec)^-1

    delta: mass splitting for inelastic scattering in keV (default value=0)

    v0: most probable WIMP speed (default value=220 km/s)

    u_esc: escape velocity in the Galactic rest frame (default value=550 km/s)

    rho_loc: local dark matter density in GeV/cm3 (default value=0.3)

    v_rms: WIMP rms speed, if None then relates to v0 (default=None)

    velocity_dependence(default=True): if False the velocity-dependent part of the 
    the cross-section is neglected 

    **args: passes the parameters of the Wilson coefficients defined in hamiltonian (see hamiltonian.global_arguments). 
            Parameters that are not passed are set to their default values, if defined (see hamiltonian.global_defaul_args), otherwise 
            to 1.
    --------------------------------------------------------------------------
    Output:

    diffential rate in units of events/kg/day/keV
    ---------------------------------------------------------------------------
    '''
    if type(target) != 'target':
        target=1*target
    
    if element_list is None:
        element_list=target.element
        
    
    hbarc2=0.389e-27
    mu_p=1./(1./0.931+1./mchi)

    dr_der=0.
    for element in element_list:
        for i,(nt_kg,mn) in enumerate(zip(element.nt_kg,element.mass)):

            vmin_er=get_vmin(er, mn, mchi, delta)

            dr_der+=np.sum(nt_kg*vmin**2*dsigma_der(element,hamiltonian,mchi,j_chi,delta,vmin,er,**args)[i]*delta_eta)


    dr_der=exposure*86400.*1e5*rho/mchi*dr_der#*np.pi/mu_p**2.*1/hbarc2
    

    return dr_der 



def diff_rate(target,hamiltonian,mchi,energy,vmin,delta_eta,j_chi=0.5,delta=0.,exposure=1.,rho_loc=0.3,elements_list=None,velocity_dependence=True,sum_over_streams=True,**args):
    '''Calculates the differential rate for WIMP-nucleus scattering
    -------------------------------------------------------------------
    Input:

    target: object belonging to the "target" or to the "element"
    class.  if belonging to "target" class the differential rate is
    summed upon the elements of the target, or on a subset of them specified
    by the elements_list argument.

    hamiltonian: object belonging to eft_hamiltonian class 
    mchi: WIMP mass in GeV


    energy: nuclear recoil energy in keV. For an element object for which the quenching method is defined it is interpreted 
    as the electron equivalent energy (i.e. the quenching is used to calculate the true recoil energy).

    vmin: array with values of vmin in km/sec where delta_eta is sampled

    delta_eta: array containing the contributions of streames vmin to the 
    halo function in (km/sec)^-1

    j_chi (default=0.5): WIMP spin


    delta: mass splitting for inelastic scattering in keV (default value=0)

    rho_loc: local dark matter density in GeV/cm3 (default value=0.3)

    exposure: exposure in kg*day (default value=1)

    velocity_dependence(default=True): if False the velocity-dependent part of the 
    the cross-section is neglected 

    elements_list: a list containing the elements to be included
    in the calculation (for the default value None all the elements in 
    the target are included). Ignored if target belongs to 
    the "element" class.

    sum_over_streams(default:True): if False does not sum over velocity streams and
    returns an array with the contributions of each velocity stream to the      
    differential rate

    **args: passes the parameters of the Wilson coefficients defined in hamiltonian (see hamiltonian.global_arguments). 
            Parameters that are not passed are set to their default values, if defined (see hamiltonian.global_defaul_args), otherwise 
            to 1.
    --------------------------------------------------------------------------
    Output:

    diffential rate in units of events/keV (events/kg/day/kev if exposure=1)
    ---------------------------------------------------------------------------
    Example:
    
    set target:
    nai=target('NaI')

    ['quenching' in dir(element) for element in nai.element] -> [False, False]
    Quenching is not defined for any element so energy is interpreted as nuclear recoil energy (quenching=1) 

    set halo function
    vmin,delta_eta0=streamed_halo_function() (time-independent component of halo function for Maxwellian with default 
                                               values of parameters - see help on streamed_halo_function)

    set effective Hamiltonian:
    o1=eft_hamiltonian('o1',{1: lambda M, r: [(1+r)/M**2,(1-r)/M**2]})  
    
    diff_rate(nai,o1,100,10,vmin,delta_eta0,M=1e3) -> differential rate in events/kg/day/keV for mchi=100GeV (WIMP mass),
                                                          energy=10 keVnr (nuclear recoil energy), M=1e3 and r=1 
                                                          (effective Hamiltonian parameters)


    dama=experiment('dama')
    ['quenching' in dir(element) for element in dama.target.element] -> [True, True]
    Quenching is defined for each element of the target of the dama experiment.

    diff_rate(dama.target.element,o1,100,10,vmin,delta_eta0,M=1e3)
    In this case energy=10 is interpreted in keVee (electron-equivalent energy).
    ---------------------------------------------------------------------------
    '''
    ## for a vanishing speed the differential rate is proportional to 1/v^2*dsigma_der
    #   with dsigma_der proportional to v^2. so if v->0 take the limit.
    vmin=np.choose(vmin==0,[vmin,1e-6])
    
    if 'target' in str(type(target)):
        if elements_list is None:
            elements_list=target.element
    elif 'element' in str(type(target)):
        elements_list=[target]
    
    hbarc2=0.389e-27
    mu_p=1./(1./0.931+1./mchi)

    dr_der=np.zeros(vmin.shape)

    for element in elements_list:

        dr_der_vec_element=np.array([])
        
        if 'quenching' in dir(element):
            er=op.bisect(lambda er:element.quenching(er)*er-energy,1e-9,1e9)
            quenching=element.quenching(er)
        else:
            er=energy
            quenching=1.

        
        # sum over isotopes    
        for i,(nt_kg,mn) in enumerate(zip(element.nt_kg,element.mass)):
                
            dr_der+=nt_kg*vmin**2*dsigma_der(element,hamiltonian,mchi,vmin,er,j_chi,delta,velocity_dependence=velocity_dependence,**args)[i]*delta_eta/quenching


    dr_der=exposure*86400.*1e5*rho_loc/mchi*dr_der#*np.pi/mu_p**2.*1/hbarc2

    if sum_over_streams:
        dr_der=np.sum(dr_der,-1)

    return dr_der 



def diff_rate_maxwellian(target,hamiltonian,mchi,er,j_chi=0.5,delta=0.,exposure=1.,v0=220.,u_esc=550.,rho=0.3,v_rms=None,**args):

    if v_rms is None:
        v_rms=np.sqrt(3./2.)*v0
    
    hbarc2=0.389e-27
    mu_p=1./(1./0.931+1./mchi)
    v_earth=v0+12
    z=np.sqrt(3./2.)*u_esc/v_rms
    eta=np.sqrt(3./2.)*v_earth/v_rms

    dr_der=0.
    for element in target.element:
        for i,(nt_kg,mn) in enumerate(zip(element.nt_kg,element.mass)):

            vmin=get_vmin(er, mn, mchi, delta)

            xmin=np.sqrt(3./2.)*vmin/v_rms


            if xmin<z-eta:

                dr_der+=nt_kg*integrate.quadrature(lambda x: x**2.*dsigma_der(element,hamiltonian,mchi,j_chi,delta,np.sqrt(2./3.)*v_rms*x,er,**args)[i]*(np.exp(-(x-eta)**2)-np.exp(-(x+eta)**2))/(eta*np.sqrt(np.pi)),xmin,z-eta,rtol=1e-3,vec_func=False)[0]


                dr_der+=nt_kg*integrate.quadrature(lambda x: x**2.*dsigma_der(element,hamiltonian,mchi,j_chi,delta,np.sqrt(2./3.)*v_rms*x,er,**args)[i]*(np.exp(-(x-eta)**2)-np.exp(-z**2))/(eta*np.sqrt(np.pi)),z-eta,z+eta,rtol=1e-3,vec_func=False)[0]


            elif xmin>z-eta and xmin<z+eta:

                dr_der+=nt_kg*integrate.quadrature(lambda x: x**2.*dsigma_der(element,hamiltonian,mchi,j_chi,delta,np.sqrt(2./3.)*v_rms*x,er,**args)[i]*(np.exp(-(x-eta)**2)-np.exp(-z**2))/(eta*np.sqrt(np.pi)),xmin,z+eta,rtol=1e-3,vec_func=False)[0]




    dr_der=exposure*86400.*1e5*v_rms*rho/mchi*np.sqrt(2./3.)*dr_der
    

    return dr_der 



def mchi_vs_exclusion(exp, hamiltonian, vmin, delta_eta, j_chi=0.5,delta=0.,
mchi_vec=None, mchi_min=0.1,mchi_max=1000,mchi_scale='linear',n_points=1000,rho_loc=0.3,
velocity_dependence=True, energy_bins_vec=None, elements_vec=None,
 response_functions=None, wimp_dd_rate_function=None,**args):

    '''
       If the expected rate calculated by wimp_dd_rate is proportional to an effective Hamiltonian parameter 
       x calculates the 90%C.L. upper bound on x as a function of the WIMP mass (in GeV), if x is set to 1 
       through **args, and for the the halo function vmin,delta_eta calculated by streamed_halo_function. 
       Calculates the strongest constraint among the energy bins in exp.data for which an upper bound of the 
       count rate is provided.
       If mchi_scale='log' a log scale is used for mchi. If the array bin_vec is provided only the 
       corresponding energy bins in exp.data are used to calculate the bound.
    --------------------------------------------------------------------------------------
    Inputs:

    exp: object belonging to experiment class

    hamiltonian: object belonging to eft_hamiltonian class

    vmin: array containing a list of WIMP stream speed velocities in the lab frame in km/s.

    delta_eta: array containing the contribution of each stream to the halo function eta(v) in (km/sec)^-1
               (the routine streamed_halo_function can be used to calculate it)

    delta (default value=0): Mass splitting for inelastic scattering in keV 

    mchi_vec (default: None): list or array with the WIMP mass values on which the bound is calculated. 
                              If not passed np.linspace(mchi_min,mchi_max,n_points) is used if mchi_scale='linear'
                              or np.logspace(mchi_min,mchi_max,n_points) if mchi_scale='log'
    
    mchi_min (default value: 0.1)

    mchi_max (default value: 1000)

    mchi_scale (defaul value: 'linear')
    
    n_points (default value: 1000)

    rho_loc(default value: 0.3): WIMP density in the neighbourghood of the Sun in GeV/cm^3. 

    velocity_dependence (default value: True): if False the velocity-dependent part of the 
    the cross-section is neglected.


    energy_bins_vec (default value: None) 

    elements_vec (dafaul value: None): A list or array of integers n with 0<n<len(exp.target.elements) that identify the different
                                       elements in a target. If provided calculates the bound including only the corresponding elements.
                                       If None all elements in target are included.
                                       
    
    j_chi (dafault value: 0.5): spin of the WIMP.

    response_functions(default value: None): If passed overrides the set of response functions exp.response_funtions[hamiltonian,j_chi]
                                             
    wimp_dd_rate_function(default value: None): overrides the function wimp_dd_rate to calculate the bound.

    **args: passes the parameters of the Wilson coefficients defined in hamiltonian (see hamiltonian.global_arguments). 
            Parameters that are not passed are set to their default values, if defined (see hamiltonian.global_defaul_args), otherwise 
            to 1. The parameter for which the bound is calculated must be set to 1, and the output of wimp_dd_rate must be proportional to it.
    --------------------------------------------------------------------------------------
    Example:

    set up experiment:
    xenon1t=experiment('xenon1t')

    set up halo function:
    vmin,delta_eta0=streamed_halo_function()  (time-independent component of halo function for Maxwellian with default 
                                               values of parameters - see help on streamed_halo_function)

    set up effective Hamiltonian:

    o1=eft_hamiltonian('o1',{1: lambda cp, r: [cp, r*cp]})

    in this way the expected rate is proportional to cp**2, while r=cn/cp (cp,cn=WIMP-proton and WIMO-neutron couplings)

    To calculate the bound on cp for r=0 need to set cp=1 (it is enough not to pass it, since arguments that are not passed and 
                                                           do not have a default value are set to 1)
    mchi,cp2_lim = mchi_vs_exclusion(xenon1t, o1, vmin, delta_eta0, r=0)

    cp_lim=cp2_lim**(1/2)

    The same Hamiltonian can be parameterized in terms of an effective mass scape M setting cp=1/M**2:

    o1=eft_hamiltonian('o1',{1: lambda M,r: [[(1+r)/M**2, (1-r)/M**2]})

    in this way the expected rate is proportional to 1/M**4.

    To calculate the bound on M for r=0.5 need to set M=1 (it is enough not to pass it, since arguments that are not passed and 
                                                           do not have a default value are set to 1)

    mchi,M_lim_m4= mchi_vs_exclusion(xenon1t, o1, vmin, delta_eta0, r=0.5)
    
    M_lim=M_lim**(-1/4)

    It is also possible to trade cp for an effetive cross section in cm^2:

    sigma_eff=cp**2*mu_chi_n**2/pi*hbarc2

    with mu_chi_n=WIMP-nucleon reduced mass in GeV and hbarc2=(hbar*c)**2 in GeV/cm. This implies to express the Wilson coefficient
    in terms of sigma_ref and r:

    hbarc2=0.389e-27 #(hbar*c)^2 in GeV^2 * cm^2

    import numpy as np
    def c_tau_sigma(sigma_ref,mchi,r=1):
        mn=0.931
        mu=mchi*mn/(mchi+mn)
        return np.sqrt(np.pi*sigma_ref/hbarc2)/mu*np.array([1+r,1-r])
    
    o1=eft_hamiltonian('o1',{1: c_tau_sigma})

    In this way the output of wimp_dd_rate is proportional to sigma_ref.
    To calculate the bound on sigma_ref for r=-1 need to set sigma_ref=1 (it is enough not to pass it, since arguments that are not passed and 
                                                           do not have a default value are set to 1)

    mchi,sigma_ref_lim= mchi_vs_exclusion(xenon1t, o1, vmin, delta_eta0, r=-1)

    Limit from C3F8 in pico including only fluorine and second run:

    pico60=experiment('pico60')

    in this case each energy bin correspond to a different run. Pico is a threshold detector 
    so in pico60.data the energy ranges must be large enough to include all the signal.
    Moreover each "energy bin"=run has a different exposure:

    len(pico60.data) -> 2 (an "energy bin" for each run)

    pico60_2019.exposure -> [1404.0, 1167.0] (exposure can be a flot or an aray/list with same length as exp.data

    [element.name for element in pico60_2019.target.element] -> ['carbon', 'fluorine']


    To calculate the bound on sigma_ref for r=0.5 including only fluorine (exp.target.elements=[1]) and the second run (exp.data[1]):

    mchi,sigma_ref_lim= mchi_vs_exclusion(xenon1t, o1, vmin, delta_eta0, energy_bins_vec=[1], elements_vec=[1], r=0.5)

    '''
    flatten=False
    
    if wimp_dd_rate_function is None:
        wimp_dd_rate_function=wimp_dd_rate
    
    # puts to one or to their default values all arguments not used in the function call
    input_args={}

    input_args.update({k:1 for k in {e for e in hamiltonian.global_arguments}-{t for t in list(args.keys())} if k not in list(hamiltonian.global_default_args.keys()) and k!='mchi' and k!='delta'})



    input_args.update({k:v for k,v in hamiltonian.global_default_args.items() if k not in list(args.keys())})

    input_args.update(args)


    if elements_vec is None:
        elements_vec=range(len(exp.target.element))


    if energy_bins_vec is None:
        n_data_points=len(exp.data)
        energy_bins_vec=range(n_data_points)


    lambda_max_90CL=np.array([2.3026,3.8897,5.3223,6.6808,7.9936,9.2747,10.5321,11.7709])

    if mchi_vec is None:
        if mchi_scale=='log':
            mchi_vec=np.logspace(np.log10(mchi_min),np.log10(mchi_max),n_points)
        else:
            mchi_vec=np.linspace(mchi_min,mchi_max,n_points)
    else:
        mchi_vec=np.array(mchi_vec)
            
    sigma_lim_vec=np.array([])
    for mchi in mchi_vec:

        energy,rate=wimp_dd_rate_function(exp,hamiltonian,vmin,delta_eta,mchi,j_chi=j_chi,delta=delta,rho_loc=rho_loc,velocity_dependence=velocity_dependence,response_functions=response_functions, elements_vec=elements_vec,**input_args)

        rate_th_vec=rate         

        sigma_lim=1.e10
        for n_bin in energy_bins_vec:
            try:
                try:
                    count_upper_bound=exp.data[n_bin][2]+1.28*exp.data[n_bin][3]
                except:
                    n_counts=int(exp.data[n_bin][2])
                    if n_counts<8:
                        count_upper_bound=lambda_max_90CL[n_counts]
                    else:
                        count_upper_bound=n_counts+1.28*np.sqrt(n_counts)

                rate_th=rate_th_vec[n_bin]

                if rate_th>0:
                    sigma_lim=min(sigma_lim,count_upper_bound/rate_th)

            except:
                pass

        # after trying all lines in the data file with at least three entries and ignorining those with two
        # checks if Optimal Interval method applies.
        if  'Optimal Interval' in exp.data.info:
            rate_th1=rate_th_vec[energy_bins_vec]
            rate_th=np.choose(rate_th1>0,[0.,rate_th1])
            mu=np.sum(rate_th)
            fc=np.append(0.,np.cumsum(rate_th)/mu)
            cl=0.9
            iflag=0
            mub=0.
            if_min=1
            upper_limit=yellin.upperlim(cl,if_min,fc,mub,fc,iflag)
            if upper_limit>0:
                sigma_lim=min(sigma_lim,upper_limit/mu)


        # tries also background subtraction, if exp.binned_background is defined
        try:
            if  'background subtraction' in exp.binned_background.__doc__:
                bck=np.array([exp.binned_background()[n][2] for n in energy_bins_vec])
                x=np.array([exp.data[n][2] for n in energy_bins_vec])
                sigma_x=np.array([])

                for n in energy_bins_vec:
                    if len(exp.data[n])==3:
                        sigma_x=np.append(sigma_x,np.sqrt(exp.data[n][2]))
                    else:
                        sigma_x=np.append(sigma_x,exp.data[n][3])

                #s=maxwellian_rate(exp,1.,mchi,delta,r,v0=v0,u_esc=u_esc)[0]
                s1=rate_th_vec[energy_bins_vec]

                s=np.choose(s1>0,[0.,s1])

                if  'physical_minimum=False' in exp.binned_background.__doc__:
                    upper_limit=cross_section_exclusion_bck_subtraction(s,bck,x,sigma_x,physical_minimum=False)
                else:
                    upper_limit=cross_section_exclusion_bck_subtraction(s,bck,x,sigma_x,physical_minimum=True)

                sigma_lim=min(sigma_lim,upper_limit)

        except:
            #print(exp.name+' background subtraction failed!')
            pass

        sigma_lim_vec=np.append(sigma_lim_vec,sigma_lim)

    nn=np.where(sigma_lim_vec<1.e10)
    return (mchi_vec[nn],sigma_lim_vec[nn])





def mchi_sigma_exclusion(exp, hamiltonian, vmin, delta_eta, delta=0.,
mchi_vec=None, mchi_min=0.1,mchi_max=1000,mchi_scale='linear',n_points=1000,rho_loc=0.3,
velocity_dependence=True, data_points_range=None, elements_vec=None,
j_chi=0.5, response_functions=None, cross_section_rescaling=1., wimp_dd_rate_function=None,**args):

    '''Calculates the 90%C.L. upper bound of the conventional WIMP-nucleon
    cross section sigma_p=c^2*mu*2/pi (in cm^2) as a function of the
    WIMP mass (in GeV) for the the halo function delta_eta calculated by
    streamed_halo_function. If mchi_scale='log' a log scale is used
    for mchi. If the array bin_vec=[] is
    provided only the corresponding bins are used to calculate the
    bound.

    '''
    flatten=False
    
    if wimp_dd_rate_function is None:
        wimp_dd_rate_function=wimp_dd_rate
    
    # puts to one or to their default values all arguments not used in the function call
    input_args={}

    input_args.update({k:1 for k in {e for e in hamiltonian.global_arguments}-{t for t in list(args.keys())} if k not in list(hamiltonian.global_default_args.keys()) and k!='mchi' and k!='delta'})



    input_args.update({k:v for k,v in hamiltonian.global_default_args.items() if k not in list(args.keys())})

    input_args.update(args)


    if elements_vec is None:
        elements_vec=range(len(exp.target.element))


    if data_points_range is None:
        n_data_points=len(exp.data)
        data_points_range=range(n_data_points)


    lambda_max_90CL=np.array([2.3026,3.8897,5.3223,6.6808,7.9936,9.2747,10.5321,11.7709])

    if mchi_vec is None:
        if mchi_scale=='log':
            mchi_vec=np.logspace(np.log10(mchi_min),np.log10(mchi_max),n_points)
        else:
            mchi_vec=np.linspace(mchi_min,mchi_max,n_points)
    else:
        mchi_vec=np.array(mchi_vec)
            
    sigma_lim_vec=np.array([])
    for mchi in mchi_vec:

        energy,rate=wimp_dd_rate_function(exp,hamiltonian,vmin,delta_eta,mchi,j_chi=j_chi,delta=delta,rho_loc=rho_loc,velocity_dependence=velocity_dependence,response_functions=response_functions, elements_vec=elements_vec,**input_args)

        rate_th_vec=rate         

        sigma_lim=1.e10
        for n_bin in data_points_range:
            try:
                try:
                    count_upper_bound=exp.data[n_bin][2]+1.28*exp.data[n_bin][3]
                except:
                    n_counts=int(exp.data[n_bin][2])
                    if n_counts<8:
                        count_upper_bound=lambda_max_90CL[n_counts]
                    else:
                        count_upper_bound=n_counts+1.28*np.sqrt(n_counts)

                rate_th=rate_th_vec[n_bin]

                if rate_th>0:
                    sigma_lim=min(sigma_lim,count_upper_bound/rate_th)

            except:
                pass

        # after trying all lines in the data file with at least three entries and ignorining those with two
        # checks if Optimal Interval method applies.
        if  'Optimal Interval' in exp.data.info:
            rate_th1=rate_th_vec[data_points_range]
            rate_th=np.choose(rate_th1>0,[0.,rate_th1])
            mu=np.sum(rate_th)
            fc=np.append(0.,np.cumsum(rate_th)/mu)
            cl=0.9
            iflag=0
            mub=0.
            if_min=1
            upper_limit=yellin.upperlim(cl,if_min,fc,mub,fc,iflag)
            if upper_limit>0:
                sigma_lim=min(sigma_lim,upper_limit/mu)


        # tries also background subtraction, if exp.binned_background is defined
        try:
            if  'background subtraction' in exp.binned_background.__doc__:
                bck=np.array([exp.binned_background()[n][2] for n in data_points_range])
                x=np.array([exp.data[n][2] for n in data_points_range])
                sigma_x=np.array([])

                for n in data_points_range:
                    if len(exp.data[n])==3:
                        sigma_x=np.append(sigma_x,np.sqrt(exp.data[n][2]))
                    else:
                        sigma_x=np.append(sigma_x,exp.data[n][3])

                #s=maxwellian_rate(exp,1.,mchi,delta,r,v0=v0,u_esc=u_esc)[0]
                s1=rate_th_vec[data_points_range]

                s=np.choose(s1>0,[0.,s1])

                if  'physical_minimum=False' in exp.binned_background.__doc__:
                    upper_limit=cross_section_exclusion_bck_subtraction(s,bck,x,sigma_x,physical_minimum=False)
                else:
                    upper_limit=cross_section_exclusion_bck_subtraction(s,bck,x,sigma_x,physical_minimum=True)

                sigma_lim=min(sigma_lim,upper_limit)

        except:
            #print(exp.name+' background subtraction failed!')
            pass

        sigma_lim_vec=np.append(sigma_lim_vec,sigma_lim)

    nn=np.where(sigma_lim_vec<1.e10)
    return (mchi_vec[nn],cross_section_rescaling*sigma_lim_vec[nn])



def binned_background(exp,verbose=False):
    path=os.getcwd()+'/'+EXPERIMENTS_PATH+'/'+exp.name
    if os.path.isfile(path+'/binned_background.tab'):                                                                                                                           
                                                                                                                                                                                
        if os.path.isfile(path+'/background.tab') or os.path.isfile(path+'/background.py'):
            
            exp.background=get_function_dir(path,'background',verbose=verbose)
            
            time_binned_background=os.stat(path+'/binned_background.tab').st_mtime
            
                                                                                                                                                                                
            try:                                                                                                                                                                
                time1=os.stat(path+'/background.tab').st_mtime
                
            except:                                                                                                                                                             
                time1=0.
                                                                                                                                                                                
            try:                                                                                                                                                                
                time2=os.stat(path+'/background.py').st_mtime                                                                                                                   
            except:                                                                                                                                                             
                time2=0.                                                                                                                                                        
                                                                                                                                                                                
            time_background=max(time1,time2)                                                                                                                                    
            time_data=os.stat(path+'/data.tab').st_mtime                                                                                                                        
                                                                                                                                                                                
            if time_binned_background>time_background and time_binned_background>time_data:                                                                                     
                output,info=get_data(path+'/binned_background',verbose=verbose)
                if len(output)==len(exp.data):                                                                                                                                
                    return (output,info)
                else:                                                                                                                                                           
                    print('binned_background not loaded because with different dimension of data.tab.')                                                                         
                    print('Delete binned_background.tab file to recalculate it.')                                                                                               
            else:                                                                                                                                                               
                if max([len(exp.data[n]) for n in range(len(exp.data))])>1:                                                                                                 
                    output=()                                                                                                                                         
                    output_table=np.array([])                                                                                                                                   
                    for n in range(len(exp.data)):                                                                                                                            
                        if len(exp.data[n])>1:                                                                                                                                
                            output_bin=integrate.quadrature(exp.background,exp.data[n][0],exp.data[n][1],rtol=1.e-03, maxiter=100, vec_func=False)[0]                       
                            output+=(np.array([exp.data[n][0],exp.data[n][1],output_bin]),)                                                                                                                 
                            output_table=np.append(output_table,[exp.data[n][0],exp.data[n][1],output_bin]).reshape(-1,3)                                                   
                                                                                                                                                                                
                    output_string=exp.background.__doc__+' Binned from background file'                                                                                         
                    with open(path+'/binned_background.tab','wd') as f:                                                                                                         
                        np.savetxt(path+'/binned_background.tab',output_table,delimiter="  ",fmt="%s", header=output_string)                                                    
                    return (output,exp.background.__doc__+'Binned from data.tab')                                                                                                                                              
                else:                                                                                                                                                           
                    print('background cannot be binned, data.tab does not contain energy intervals')                                                                            
                    return                                                                                                                                                     
                                                                                                                                                                                
        else:
            output,info=get_data(path+'/binned_background',verbose=verbose)
            
            
            if len(exp.data)==len(output):                                                                                                                                    
                                                                                                                                                                                
                return (output,info)                                                                                                                                                  
            else:                                                                                                                                                               
                print('No binned background loaded: data.tab and binned_background.tab have a different number of bins')                                                        
                return                                                                                                                                                         
                                                                                                                                                                                
    else:                                                                                                                                                                       
        if os.path.isfile(path+'/background.tab') or os.path.isfile(path+'/background.py'):                                                                                     
            exp.background=get_function_dir(path,'background',verbose=verbose)                                                                                                                  
            if max([len(exp.data[n]) for n in range(len(exp.data))])>1:                                                                                                     
                                                                                                                                                                                
                output=()                                                                                   
                output_table=np.array([])                                                                                                                                       
                for n in range(len(exp.data)):                                                                                                                                
                    if len(exp.data[n])>1:                                                                                                                                    
                        output_bin=integrate.quadrature(exp.background,exp.data[n][0],exp.data[n][1],rtol=1.e-03, maxiter=100, vec_func=False)[0]                           
                        output+=(np.array([exp.data[n][0],exp.data[n][1],output_bin]),)                                                                                                                       
                        output_table=np.append(output_table,[exp.data[n][0],exp.data[n][1],output_bin]).reshape(-1,3)                                                       
                                                                                                                                                                                
                output_string=exp.background.__doc__+' Binned from background file'                                                                                             
                with open(path+'/binned_background.tab','wd') as f:                                                                                                             
                    np.savetxt(path+'/binned_background.tab',output_table,delimiter="  ",fmt="%s", header=output_string)                                                        
                                                                                                                                                                                
                return (output,exp.background.__doc__+'Binned from data.tab')                                                                                                                                                  
                                                                                                                                                                                
            else:                                                                                                                                                               
                print('background cannot be binned, data.tab does not contain energy intervals')                                                                                
                                                                                                                                                                                
                return                                                                                                                                                         

            


        
        
def cross_section_exclusion_bck_subtraction(s,bck,x,sigma,physical_minimum=True):

    '''Calculates the upper bound on the cross section if x and sigma are                                
    vectors containing the measured count rates and errors and s0 and                                    
    b0 the normalized expected wimp rates and background. The                                            
    dimension of the vectors is equal to the number of experimental                                      
    bins.                                                                                                
                                                                                                         
    '''                                                                                                  

    cross_section_exclusion_bck_subtraction.rho_lim=1.
    n=1.28 # 90% C.L. is 1.28 sigma                                                                      
                                                                                                         
    matrix=np.array([np.sum(s**2/sigma**2),np.sum(bck*s/sigma**2),                                       
                     np.sum(bck*s/sigma**2),np.sum(bck**2/sigma**2)]).reshape(2,2)                       
                                                                                                         
    vector=np.array([np.sum(x*s/sigma**2),np.sum(x*bck/sigma**2)])                                       
                                                                                                         
    # finds the minimum (if it exists) of the chi square as a function                                                  
    # of the cross section and of the normalization of the signal
    try:
        sigma_min,rho_min=np.linalg.solve(matrix, vector)                                                    
    except:
        return 1e10

        
    if physical_minimum:
        #both rho_min and sigma_min must be positive                                                         
                                                                                                         
        if sigma_min<0 and rho_min<0:                                                                        
            sigma_min=0.                                                                                     
            rho_min=0.                                                                                       
                                                                                                         
        elif sigma_min<0 and rho_min>0:                                                                      
            sigma_min=0.                                                                                     
            rho_min=np.sum(x*bck/sigma**2)/np.sum(bck**2/sigma**2)                                           
        elif sigma_min>0 and rho_min<0:                                                                      
            sigma_min=np.sum(x*s/sigma**2)/np.sum(s**2/sigma**2)                                             
            rho_min=0.                                                                                       
                                                                                                         
    chi2_min=np.sum((sigma_min*s+rho_min*bck-x)**2/sigma**2)                                             
                                                                                                         
    a=np.sum(s**2/sigma**2)                                                                              
    b=np.sum(bck**2/sigma**2)                                                                            
    c=2*np.sum(bck*s/sigma**2)                                                                           
    d=-2.*np.sum(s*x/sigma**2)                                                                           
    e=-2.*np.sum(bck*x/sigma**2)                                                                         
    f=np.sum(x**2/sigma**2)-chi2_min-n**2                                                                
                                                                                                         
    a_big=c**2-4*a*b                                                                                     
    b_big=2.*(c*e-2*b*d)                                                                                 
    c_big=e**2-4.*b*f                                                                                    
                                                                                                         
    delta=b_big**2-4.*a_big*c_big                                                                        
                                                                                                         
    sigma_lim=max((-b_big+np.sqrt(delta))/(2.*a_big),(-b_big-np.sqrt(delta))/(2.*a_big))                 

                                                                                                         
    a_big=c**2-4*a*b                                                                                     
    b_big=2.*(c*d-2*a*e)                                                                                 
    c_big=d**2-4.*a*f                                                                                    
                                                                                                         
    delta_rho=b_big**2-4.*a_big*c_big                                                                    
                                                                                                         
    rho_1=(-b_big+np.sqrt(delta_rho))/(2.*a_big)                                                         
    rho_2=(-b_big-np.sqrt(delta_rho))/(2.*a_big)                                                         
                                                                                                         
                                                                                                         
                                                                                                         
    aa=b                                                                                                 
    bb=c*sigma_lim+e                                                                                     
    cc=a*sigma_lim**2+d*sigma_lim+f                                                                      
                                                                                                         
    delta_lim=bb**2-4.*aa*cc                                                                             
                                                                                                         
    rho_lim=-bb/(2.*aa)
    
    if physical_minimum:                                                                                                     
        if rho_lim<0:                                                                                        
            sigma_lim=max((-d+np.sqrt(d**2-4.*a*f))/(2.*a),(-d-np.sqrt(d**2-4.*a*f))/(2.*a))                 
            rho_lim=0.
            
    #print(str(sigma_lim))
    cross_section_exclusion_bck_subtraction.rho_lim=rho_lim    
    return sigma_lim                                                                                     
            


def read_coupling_from_function_name(c):
    c=c[2:]                                                                                        
    c_out=None                                                                                     
    for response in nuclear_response_list:                                                      
        if response in c:                                                                          
            if c.count('_')<3:                                                                     
                c="'"+response+"'"+c.replace(response,'').replace('_',',')                         
                c_out=eval(c)                                                                      
            else:                                                                                  
                __c1__="'"+response+"'"+c[:find_nth(c,'_',3)].replace('Sigma','').replace('_',',')     
                __c2__=c[find_nth(c,'_',3)+1:].split('_')                                              
                c_out=eval(__c1__)+tuple(__c2__)                                                           
                                                                                                   
                                                                                                   
                                                                                                   
    if c_out is None:                                                                              
        for n_spin in n_coeff_vec.values():                                                     
            for n in n_spin:                                                                       
                if str(n+1)==c:                                                                    
                    c_out=int(c)-1                                                                 
                                                                                                   
    if c_out is None:                                                                              
        print('Wilson coefficient name was not recognized')                                        
    else:                                                                                          
        return c_out                                                                               


def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start
    
        


def clean_formula(formula):
    symbols,nn=read_formula(formula)
    nn_new=np.array([],dtype='int')
    symbols_new=np.array([])
    for s in np.unique(symbols):
        symbols_new=np.append(symbols_new,s)
        nn_new=np.append(nn_new,np.sum(nn[np.where(s==symbols)]))

    new_formula=''

    for s,n in zip(symbols_new,nn_new):
        if n>1:
            new_formula+=s+str(n)
        else:
            new_formula+=s

    return new_formula



delta_vearth=14.9
vp=np.array([9.,12.,7.])

def v_earth_sun(ecliptic_longitude,verbose=False,v_rot_gal=[0.,220.,0.],v_sun_rot=[9.,12.,7.]):
    ue=29.79

    beta_x=np.radians(-5.5303)
    beta_y=np.radians(59.575)
    beta_z=np.radians(29.812)
    lambda_x=np.radians(266.141)
    lambda_y=np.radians(-13.3485)
    lambda_z=np.radians(179.3212)

    vec_cos_beta=np.array([np.cos(beta_x),np.cos(beta_y),np.cos(beta_z)])
    vec_cos_lambda=np.array([np.cos(lambda_x),np.cos(lambda_y),np.cos(lambda_z)])
    vec_sin_lambda=np.array([np.sin(lambda_x),np.sin(lambda_y),np.sin(lambda_z)])
    if verbose:
        v_sun_gal=np.array(v_rot_gal)+np.array(v_sun_rot)
        A=np.sum(v_sun_gal*vec_cos_beta*vec_cos_lambda)
        B=-np.sum(v_sun_gal*vec_cos_beta*vec_sin_lambda)
        lambda_0=np.arctan(A/B)
        v_earth_sun.lambda_0=lambda_0
        print(70*'=')
        print('modulation maximum phase='+str(np.degrees(lambda_0)*365/360)+' from vernal equinox, i.e. day no.'+str(80.+np.degrees(lambda_0)*365/360)+' of the year')
        print(70*'=')

    return np.array([ue*np.cos(beta_x)*np.sin(ecliptic_longitude-lambda_x),ue*np.cos(beta_y)*np.sin(ecliptic_longitude-lambda_y),ue*np.cos(beta_z)*np.sin(ecliptic_longitude-lambda_z)])


def streamed_halo_function(velocity_distribution_gal=None,v_rot_gal=np.array([0.,220.,0.]), v_sun_rot=np.array([9.,12.,7.]), v_esc_gal=550,n_vmin_bin=1000,yearly_modulation=False, vmin=None, delta_eta=True,full_year_sampling=False, year=None, day_of_the_year=None, outputfile=None,recalculate=False,modulation_phase=152.5,**args):

    '''
    Calculates the halo function defined as the integral from vmin to infinity of f(v)/v, with v the WIMP speed and 
    f(v) the WIMP speed distribution, both in the Earth rest frame. 

    Expressing the halo function as eta(v,t)=eta0(v)+eta1(v)*cos[2pi/T*(t-t0)] with T=1 year and defining:
    
    eta0_i=eta0(v_i)
    eta1_i=eta1(v_i)

    eta0(v)=sum_i delta_eta0_i*theta(v_i-v)
    eta1(v)=sum_i delta_eta1_i*theta(v_i-v)

    (with theta representing a Heaveside theta function) the routine calculates: 

                                delta_eta=True     delta_eta=False
    yearly_modulation=False      delta_eta0_i           eta0_i
    yearly_modulation=True       delta_eta1_i           eta0_i
 
    in s/km, with eta0_i=eta0(v_i) and delta_eta_i=eta_i-eta_i-1 
    on a sample of velocity values v_i in km/s for i=1,...,n_vmin_bin (eta_0=eta(vmin=0)) contained in the array vmin. 
    
    eta0 and eta1 are estimated differently depending on the values of velocity_distribution_gal, full_year_sampling,
    modulation_phase and day_of_the_year:

    if full_year_sampling=True:

          eta0 = 1/T*sum_i eta(t_i), i=1,...,365 (time average over the 365 days of the year) 
          eta1 = 1/T*sum_i eta(t_i) cos[2pi/T*(t_i-modulation_phase)] (cosine transform over the 365 days of the year)

    if full_year_sampling=False:                                                               
                                                   eta0                             eta1
  
    velocity_distribution_gal=None             eta(day_of_the_year)            d eta/d v_earth_gal*delta_v_earth (t=day_of_the_year)
    velocity_distribution_gal=!None            eta(day_of_the_year)            [eta(modulation_phase)-eta(modulation_phase-T/2)]/2    
    
    with v_earth_gal the norm of the time-dependent velocity of the Earth in the Galactic_rest_frame and  delta_v_earth=14.8 km/s


    By default the routine calculates v_i, delta_eta0_i and v_i,delta_eta1_iwhich are directly 
    used as inputs from the signal routines wimp_dd_rate and diff_rate. In this case the output values v_i are the upper boundaries
    of the intervals v_i-v_{i-1} 
    --------------------------------------------------------------------------------------
    Input:

    velocity_distribution_gal: WIMP velocity distribution f(u,**args) in the Galactic rest frame (default value: None). 
    If passed the halo functions are calculated through a numerical integration.
    The first argument of f must be an ndarray u=array([ux,uy,uz]) that represents the WIMP velocity in 
    the Galactic rest frame, where ux points toward the galactic center, uy in the direction of the motion of the rotational curve and uz 
    toward the galactic North pole. **args can contain any additional parameter.       
    If velocity_distribution_gal=None the halo functions are calculated using analytical expressions for a standard maxwellian
    truncated at v_esc_gal.
    

    v_rot_gal: galactic rotational velocity at Sun's position in km/s (default value: [0.,220.,0.]).

    v_sun_rot: peculiar velocity of the Sun with respect to the rotation curve in km/s ((default value: [9.,12.,7.]).

    v_esc_gal(default value: 550): escape velocity in km/s.

    n_vmin_bin(default value: 50): number of values of output vmin array.   

    yearly_modulation: (default value: False)

    vmin: (default value:None)  list or array of velocities v_i in km/s that overrides the output vmin array.  
    If vmin=None it is set to n_vmin_bin linearly spaced values from 0 to v_max  with v_max the maximal WIMP speed in the Earth 
    reference frame, v_max=v_esc_gal+|v_earth_gal|, with v_earth_gal the velocity of the Earth in the galactic frame.
    
    delta_eta: default value: True
    
    full_year_sampling: (default value: False). 

    day_of_the_year: (default value: modulation_phase-T/4

    outputfile: (default value: None).  
    If velocity_distribution_gal=!None the halo functions are saved in an output file 
    with name outputfile. The output file is saved in the folder Halo_functions of the user's working directory. 
    If missing, the folder Halo_functions is automatically created.
    If outputfile=None the name of the outputfile is set to:

    output_file=distribution_name+'_no_full_year_sampling_eta_halo_func_num.npy' if full_year_sampling=False
    output_file=distribution_name+'_eta_halo_func_num.npy' if full_year_sampling=True full_year_sampling=True

    with distribution_name the name of the halo function passed with the velocity_distribution_gal argument
    (specifically, if velocity_distribution_gal=f -> distribution_name=f.__name__).

    The content of the output file depends on full_year_sampling: 

          if full_year_sampling=False: four columns containing vmin, eta(day_of_the_year), eta(modulation_phase), eta(modulation_phase-T/2)
          if full_year_sampling=True: 366 columns containing vmin and eta(t_i) with i=1,...,365    

    modulation_phase: (default value: 152.5 (2nd of June))

    If outputfile is passed the content of the corresponding file is used to calculate
    the output, irrespective on the value of velocity_distributon_gal. In this case the variable 
    full_year_sampling must be set according to the content of outputfile.


    **args: additional arguments of the numerical velocity distribution passed with velocity_distribution_gal.
    ---------------------------------------------------------------------------------------
    Output:

    Two arrays contaning vmin in km/s and one of the four arrays containing eta0, eta1, delta_eta0, delta_eta1
    The arrays vmin, delta_eta0 are needed by the routines diff_rate and wimp_dd_rate to calculate unmodulated signals
    The arrays vmin, delta_eta1 are needed by the routines diff_rate and wimp_dd_rate to calculate modulated signals
    ---------------------------------------------------------------------------------------

    '''    
    from timeit import default_timer as timer

    lambda_0=(modulation_phase-80)*2.*np.pi/365
    if day_of_the_year is None:
        power_expansion_lambda=lambda_0-np.pi/2
    else:
        power_expansion_lambda=(day_of_the_year-80)*2.*np.pi/365
    v_sun_gal=v_rot_gal+v_sun_rot
    v0=np.sqrt(np.dot(v_rot_gal,v_rot_gal))

    input_params={}
    input_params['v_rot_gal']=v_rot_gal
    input_params['v_sun_rot']=v_sun_rot
    input_params['v_esc_gal']=v_esc_gal
    input_params['full_year_sampling']=full_year_sampling         
    input_params.update(args)

    params=parameters(input_params)
    
    if velocity_distribution_gal is None and outputfile is None:
        
        if full_year_sampling:

            if year is None:
                year=np.arange(0.,2.*np.pi,2*np.pi/365.)
            else:
                year=2*np.pi/365*year
            delta_vec=np.array([])
            vmin_vec=np.array([])

            for lambda0 in year:

                vmin_out,delta_out=streamed_halo_function_maxwellian(v_rot_gal=v_rot_gal, v_sun_rot=v_sun_rot, v_esc_gal=v_esc_gal, n_vmin_bin=n_vmin_bin, yearly_modulation=False, vmin=vmin, v_earth_sun=v_earth_sun(lambda0), delta_eta=delta_eta, **args)
                
                delta_vec=np.append(delta_vec,delta_out)
                vmin_vec=np.append(vmin_vec,vmin_out)

            delta_year_vec=delta_vec.reshape(year.shape[0],-1)
            vmin_year_vec=vmin_vec.reshape(year.shape[0],-1)

            if yearly_modulation:
                delta_eta=np.sum((delta_year_vec.T*np.cos(year-lambda_0)).T,0)*2/year.shape[0]
            else:
                delta_eta=np.sum(delta_year_vec,0)*1/year.shape[0]
               
            vmin_ave=np.sum(vmin_year_vec,0)*1/year.shape[0]

            return vmin_ave, delta_eta 

        else:
            
            if day_of_the_year is None:
                v_earth_sun_value=[0,0,0]
            else:
                v_earth_sun_value=v_earth_sun(power_expansion_lambda)
            
            return streamed_halo_function_maxwellian(v_rot_gal=v_rot_gal, v_sun_rot=v_sun_rot, v_esc_gal=v_esc_gal, n_vmin_bin=n_vmin_bin, yearly_modulation=yearly_modulation, vmin=vmin, v_earth_sun=v_earth_sun_value,delta_eta=delta_eta, **args)

    # numerical calculation
    else:
        vmin_start=0.

        v_esc_earth=v_esc_gal+np.sqrt(np.dot(v_sun_gal,v_sun_gal))

        #path='Halo_functions'
        path=HALO_FUNCTIONS_PATH
        if not os.path.isdir(path):
            os.makedirs(os.path.join(path))
            #p = subprocess.call("mkdir -p "+path, stdout=subprocess.PIPE, shell=True)

        if vmin is None:
            vmin=np.linspace(vmin_start,v_esc_earth,n_vmin_bin)
        ## in all cases adds v=0 at the beginning. uses it if delta_eta=True and ignores it if delta_eta=False
        vmin=np.append(0,vmin)

        # june-december
        if not full_year_sampling:

            year=np.array([power_expansion_lambda,lambda_0,lambda_0+np.pi])

            if outputfile is None:
                inputfile=velocity_distribution_gal.__name__+'_no_full_year_sampling_eta_halo_func_num.npy'
            else:
                inputfile=outputfile

            if os.path.isfile(path+'/'+inputfile) and not recalculate:
                print('loading '+inputfile)
                print('type: print(np.load(\''+path+'/'+inputfile+'\',allow_pickle=True)[-1]) to get input parameters')
                print(70*'=')

                with open(path+'/'+inputfile, 'rb') as file:
                    vmin, eta_num_vec,params=pickle.load(file)

            else:
                velocity_distribution_gal_spherical=convert_to_spherical(velocity_distribution_gal,**args)

                print('calculating normalization...')
                norm=tplquad(lambda u,cos_theta,phi: u**2*velocity_distribution_gal_spherical(u,cos_theta,phi), 0, 2.*np.pi, -1, 1, 0., v_esc_gal, epsabs=1.49e-7, epsrel=1.49e-7)[0]
                print('normalization='+str(norm))                
                eta_num_vec=np.array([])
                n_tot=vmin.shape[0]*year.shape[0]

                for v_lab in vmin:
                    print(70*'=')
                    eta0_num=np.array([])
                    
                    for lambda0 in year:
                        v_earth=v_earth_sun(lambda0)
                        
                        v0_tilde=v_sun_gal+v_earth
                        v0_tilde_norm=np.sqrt(np.sum(v0_tilde**2))

                        if v_lab<v0_tilde_norm:
                            if v_lab<v0_tilde_norm-v_esc_gal:
                                v_lab=v0_tilde_norm-v_esc_gal

                        vmax=vmax_func(v0_tilde_norm=v0_tilde_norm,u_esc=v_esc_gal)
                        earth_velocity_distribution_spherical=boost1(velocity_distribution_gal,v0_tilde,**args)
                        s1_num=timer()

                        if v_lab<v_esc_gal-v0_tilde_norm:
                            cos_theta_max=1
                        elif v_lab>v_esc_gal-v0_tilde_norm and v_lab<v_esc_gal+v0_tilde_norm:
                            cos_theta_max=(v_esc_gal**2-v_lab**2-v0_tilde_norm**2)/(2.*v_lab*v0_tilde_norm)
                        else:
                            cos_theta_max=-1.
                            
                        print(str(n_tot)+' points remaining')
                            
                        integral = tplquad(lambda v,cos_theta, phi: v**2*earth_velocity_distribution_spherical(v,cos_theta,phi)/(v*norm),0.,2.*np.pi,-1., cos_theta_max, v_lab, vmax, epsabs=1.49e-7, epsrel=1.49e-7)
                        eta0_num=np.append(eta0_num,integral[0])
                        s2_num=timer()
                        n_tot-=1

                        print('vmin='+str(v_lab)+' and lambda0='+str(lambda0)+' completed in '+str(s2_num-s1_num)+' sec with value '+str(integral[0])+' and error '+str(integral[1]))
                    eta_num_vec=np.append(eta_num_vec,eta0_num).reshape(-1,year.shape[0])

                with open(path+'/'+inputfile, 'wb') as file:
                    pickle.dump((vmin,eta_num_vec,params), file)
                

            if yearly_modulation:
                eta_avg=(eta_num_vec[:,1]-eta_num_vec[:,2])/2.                
            else:
                eta_avg=eta_num_vec[:,0]                


            # does not return vmin[0]=0 added the beginning
            # returns all delta_eta and eta[1:]
            if delta_eta:
                delta_eta=np.diff(eta_avg[::-1])[::-1]
                return vmin[1:],delta_eta

            else:
                return vmin[1:], eta_avg[1:]
            
        # numerical calculation, full year sampling
        else:
            year=np.arange(0.,2.*np.pi,2*np.pi/365.)

            if outputfile is None:
                inputfile=velocity_distribution_gal.__name__+'_eta_halo_func_num.npy'
            else:
                inputfile=outputfile

            if os.path.isfile(path+'/'+inputfile) and not recalculate:
                print('loading '+inputfile)
                print('type: print(np.load(\''+path+'/'+inputfile+'\',allow_pickle=True)[-1]) to get input parameters')
                print(70*'=')

                with open(path+'/'+inputfile, 'rb') as file:
                    vmin, eta_num_vec,params=pickle.load(file)

            else:
                velocity_distribution_gal_spherical=convert_to_spherical(velocity_distribution_gal,**args)

                norm=tplquad(lambda u,cos_theta,phi: u**2*velocity_distribution_gal_spherical(u,cos_theta,phi), 0, 2.*np.pi, -1, 1, 0., v_esc_gal, epsabs=1.49e-7, epsrel=1.49e-7)[0]
                eta_num_vec=np.array([])
                n_tot=vmin.shape[0]*year.shape[0]

                for v_lab in vmin:
                    print(70*'=')
                    eta0_num=np.array([])
                    for lambda0 in year:
                        v_earth=v_earth_sun(lambda0)
                        #np.array([ue*np.cos(beta_x)*np.sin(lambda0-lambda_x),ue*np.cos(beta_y)*np.sin(lambda0-lambda_y),ue*np.cos(beta_z)*np.sin(lambda0-lambda_z)])
                        v0_tilde=v_sun_gal+v_earth
                        v0_tilde_norm=np.sqrt(np.sum(v0_tilde**2))

                        if v_lab<v0_tilde_norm:
                            if v_lab<v0_tilde_norm-v_esc_gal:
                                v_lab=v0_tilde_norm-v_esc_gal

                        vmax=vmax_func(v0_tilde_norm=v0_tilde_norm,u_esc=v_esc_gal)
                        earth_velocity_distribution_spherical=boost1(velocity_distribution_gal,v0_tilde,**args)
                        s1_num=timer()

                        if v_lab<v_esc_gal-v0_tilde_norm:
                            cos_theta_max=1
                        elif v_lab>v_esc_gal-v0_tilde_norm and v_lab<v_esc_gal+v0_tilde_norm:
                            cos_theta_max=(v_esc_gal**2-v_lab**2-v0_tilde_norm**2)/(2.*v_lab*v0_tilde_norm)
                        else:
                            cos_theta_max=-1.

                        integral = tplquad(lambda v,cos_theta, phi: v**2*earth_velocity_distribution_spherical(v,cos_theta,phi)/(v*norm),0.,2.*np.pi,-1., cos_theta_max, v_lab, vmax, epsabs=1.49e-7, epsrel=1.49e-7)
                        eta0_num=np.append(eta0_num,integral[0])
                        s2_num=timer()
                        n_tot-=1
                        print(str(n_tot)+' points remaining')
                        print('vmin='+str(v_lab)+' and lambda0='+str(lambda0)+' completed in '+str(s2_num-s1_num)+' sec with error '+str(integral[1]))
                    eta_num_vec=np.append(eta_num_vec,eta0_num).reshape(-1,year.shape[0])
                    
                with open(path+'/'+inputfile, 'wb') as file:
                    pickle.dump((vmin,eta_num_vec,params), file)
                print('run ended at:'+time.strftime("%H:%M:%S"))

            if yearly_modulation:
                #eta_avg=(np.sum(eta_num_vec[:-1]*np.cos(year-lambda_0),1)+np.sum(eta_num_vec[1:]*np.cos(year-lambda_0),1))/2*2./year.shape[0]
                eta_avg=2*np.sum(eta_num_vec*np.cos(year-lambda_0),1)/year.shape[0]
            else:
                #eta_avg=(np.sum(eta_num_vec[:-1],1)+np.sum(eta_num_vec[1:],1))/2*1./year.shape[0]
                eta_avg=np.sum(eta_num_vec,1)/year.shape[0]

            if delta_eta:
                delta_eta=np.diff(eta_avg[::-1])[::-1]
                return vmin[1:], delta_eta
            else:
                return vmin[1:], eta_avg[1:]




def vmax_func(v0_tilde_norm,u_esc):
    def vmax_spherical(phi,cos_theta):
        a=1.
        b=v0_tilde_norm*cos_theta
        c=v0_tilde_norm**2-u_esc**2
        return np.sqrt(b**2-a*c)-b
    return vmax_spherical


def boost(distribution,v0_tilde_norm,**args):
    def boosted_distribution(v,cos_theta,phi):
        sin_theta=np.sqrt(1.-cos_theta**2)
        vx=v*sin_theta*np.cos(phi)
        vy=v*cos_theta
        vz=v*sin_theta*np.sin(phi)
        return distribution(np.array([vx,vy,vz])+np.array([0,v0_tilde_norm,0]),**args)
    return boosted_distribution


def boost1(distribution,v0_tilde,**args):
    def boosted_distribution(v,cos_theta,phi):
        sin_theta=np.sqrt(1.-cos_theta**2)
        vx=v*sin_theta*np.cos(phi)
        vy=v*cos_theta
        vz=v*sin_theta*np.sin(phi)
        return distribution(np.array([vx,vy,vz])+v0_tilde,**args)
    return boosted_distribution



def convert_to_spherical(distribution,**args):
    def spherical(v,cos_theta,phi):
        sin_theta=np.sqrt(1.-cos_theta**2)
        vx=v*sin_theta*np.cos(phi)
        vy=v*sin_theta*np.sin(phi)
        vz=v*cos_theta
        return distribution([vx,vy,vz],**args)
    return spherical


def streamed_halo_function_maxwellian(v_rot_gal=[0.,220.,0.], v_sun_rot=[9.,12.,7.], v_esc_gal=550, n_vmin_bin=1000,yearly_modulation=False,vrms=None,vmin=None,v_earth_sun=[0.,0.,0.],delta_eta=True):

    
    v_rot_gal=np.array(v_rot_gal)
    v_sun_rot=np.array(v_sun_rot)    
    
    v_sun_gal=v_rot_gal+v_sun_rot

    v_earth_gal=v_sun_gal

    
    if vrms is None:
        v0=np.sqrt(np.dot(v_rot_gal,v_rot_gal))
        vrms=np.sqrt(3./2.)*v0

    if vmin is None:
        vmin_start=0.
        v_esc_earth=v_esc_gal+np.sqrt(np.dot(v_earth_gal,v_earth_gal))
        vmin=np.linspace(vmin_start,v_esc_earth,n_vmin_bin)

    if delta_eta:
        #removes one value when doing average; another value calculating delta_eta's
        vmin=np.append([0,0],vmin)
    else:
        #removes one value when doing average
        vmin=np.append([0],vmin)
        
            
    vmin_upper=vmin[1:]
    vmin_lower=vmin[:-1]

    vmin=vmin_upper
    
    if yearly_modulation:
        eta=eta1_ave_bin(vmin_lower,vmin_upper,v_esc_gal,vrms,v_rot_gal,v_sun_rot,v_earth_sun)
    else:
        eta=eta0_ave_bin(vmin_lower,vmin_upper,v_esc_gal,vrms,v_rot_gal,v_sun_rot,v_earth_sun)

    if delta_eta:
        #delta_eta=np.append(np.diff(eta[::-1])[::-1],eta[-1])
        delta_eta=np.diff(eta[::-1])[::-1]
        return vmin[1:],delta_eta
    else:
        return vmin,eta


def streamed_halo_function_maxwellian2(v_rot_gal=[0.,220.,0.], v_sun_rot=[9.,12.,7.], v_esc_gal=550, n_vmin_bin=50,yearly_modulation=False,vrms=None,vmin=None,v_earth_sun=[0.,0.,0.],delta_eta=True):

    v_rot_gal=np.array(v_rot_gal)
    v_sun_rot=np.array(v_sun_rot)    
    
    v_sun_gal=v_rot_gal+v_sun_rot

    v_earth_gal=v_earth_sun+v_sun_gal

    
    if vrms is None:
        v0=np.sqrt(np.dot(v_rot_gal,v_rot_gal))
        vrms=np.sqrt(3./2.)*v0

    if vmin is None:
        vmin_start=0.
        v_esc_earth=v_esc_gal+np.sqrt(np.dot(v_earth_gal,v_earth_gal))
        vmin=np.linspace(vmin_start,v_esc_earth,n_vmin_bin)
    else:
        vmin=np.array(vmin)
    
    if yearly_modulation:

        eta=eta1_halo_function_maxwellian(vmin,u_esc_gal=v_esc_gal,v_rot_gal=v_rot_gal,v_sun_rot=v_sun_rot,v_earth_sun=v_earth_sun,vrms=vrms)
                
    else:
        eta=eta0_halo_function_maxwellian(vmin,u_esc_gal=v_esc_gal,v_rot_gal=v_rot_gal,v_sun_rot=v_sun_rot,v_earth_sun=v_earth_sun,vrms=vrms)
        
    if delta_eta:

        if yearly_modulation:
            eta_v0=eta1_halo_function_maxwellian(0,u_esc_gal=v_esc_gal,v_rot_gal=v_rot_gal,v_sun_rot=v_sun_rot,v_earth_sun=v_earth_sun,vrms=vrms)
        else:
            eta_v0=eta0_halo_function_maxwellian(0,u_esc_gal=v_esc_gal,v_rot_gal=v_rot_gal,v_sun_rot=v_sun_rot,v_earth_sun=v_earth_sun,vrms=vrms)
        eta=np.append(eta_v0,eta)
        
        delta_eta=np.diff(eta[::-1])[::-1]
        return vmin,delta_eta
    else:
        return vmin,eta



    
def eta0_halo_function_maxwellian(v,u_esc_gal=550.,vrms=None,v_rot_gal=[0.,220.,0.],v_sun_rot=[9.,12.,7.],v_earth_sun=[0.,0.,0]):

    
    v_rot_gal=np.array(v_rot_gal)
    v_sun_rot=np.array(v_sun_rot)
    v_earth_sun=np.array(v_earth_sun)
    
    v_sun_gal=v_rot_gal+v_sun_rot

    if vrms is None:
        v0=np.sqrt(np.dot(v_rot_gal,v_rot_gal))
        vrms=np.sqrt(3./2.)*v0

    vearth_gal=np.sqrt(np.dot(v_sun_gal+v_earth_sun,v_sun_gal+v_earth_sun))


    eta=np.sqrt(3./2.)*vearth_gal/vrms
    z=np.sqrt(3./2.)*u_esc_gal/vrms
    N=1./(sp.erf(z)-2./np.sqrt(np.pi)*z*np.exp(-z**2))
    x=np.sqrt(3./2.)*v/vrms
    constant=N/eta*np.sqrt(3./(2.*np.pi))*1/vrms


    condition1=x<=z-eta
    condition2=x>=z+eta 

    output1=constant*(np.sqrt(np.pi)/2.*(sp.erf(x+eta)-sp.erf(x-eta))-2*eta*np.exp(-z**2))
    output2=constant*(np.sqrt(np.pi)/2*(sp.erf(z)-sp.erf(x-eta))-(z+eta-x)*np.exp(-z**2))
    output3=0.

    return np.choose(condition1,[np.choose(condition2,[output2,output3]),output1])    


def diff_eta0_halo_function_maxwellian(v,u_esc_gal=550.,vrms=None,v_rot_gal=[0.,220.,0.],v_sun_rot=[9.,12.,7.],v_earth_sun=[0.,0.,0]):
    v_rot_gal=np.array(v_rot_gal)
    v_sun_rot=np.array(v_sun_rot)
    v_earth_sun=np.array(v_earth_sun)
    
    v_sun_gal=v_rot_gal+v_sun_rot

    if vrms is None:
        v0=np.sqrt(np.dot(v_rot_gal,v_rot_gal))
        vrms=np.sqrt(3./2.)*v0

    vearth_gal=np.sqrt(np.dot(v_sun_gal+v_earth_sun,v_sun_gal+v_earth_sun))

    eta=np.sqrt(3./2.)*vearth_gal/vrms
    delta_eta=np.sqrt(3./2.)*delta_vearth/vrms
    z=np.sqrt(3./2.)*u_esc_gal/vrms
    N=1./(sp.erf(z)-2./np.sqrt(np.pi)*z*np.exp(-z**2))
    x=np.sqrt(3./2.)*v/vrms
    constant=N*np.sqrt(3./(2.*np.pi))*1/vrms

    condition1=x<=z-eta
    condition2=x>=z+eta


    output1=constant*(np.exp(-(x+eta)**2)+np.exp(-(x-eta)**2)-2.*np.exp(-z**2))

    output2=constant*(np.exp(-(x-eta)**2)-np.exp(-z**2))

    output3=0.

    return np.choose(condition1,[np.choose(condition2,[output2,output3]),output1])    


def eta1_halo_function_maxwellian(v,u_esc_gal=550.,vrms=None,v_rot_gal=[0.,220.,0.],v_sun_rot=[9.,12.,7.],v_earth_sun=[0.,0.,0]):

    v_rot_gal=np.array(v_rot_gal)
    v_sun_rot=np.array(v_sun_rot)
    v_earth_sun=np.array(v_earth_sun)
    
    v_sun_gal=v_rot_gal+v_sun_rot

    if vrms is None:
        v0=np.sqrt(np.dot(v_rot_gal,v_rot_gal))
        vrms=np.sqrt(3./2.)*v0

    vearth_gal=np.sqrt(np.dot(v_sun_gal+v_earth_sun,v_sun_gal+v_earth_sun))

    eta=np.sqrt(3./2.)*vearth_gal/vrms
    delta_eta=np.sqrt(3./2.)*delta_vearth/vrms

    
    return delta_eta/eta*(\
    diff_eta0_halo_function_maxwellian(v,u_esc_gal=u_esc_gal,vrms=vrms,v_rot_gal=v_rot_gal,v_sun_rot=v_sun_rot,v_earth_sun=v_earth_sun)-\
    eta0_halo_function_maxwellian(v,u_esc_gal=u_esc_gal,vrms=vrms,v_rot_gal=v_rot_gal,v_sun_rot=v_sun_rot,v_earth_sun=v_earth_sun)\
    )





    
def eta0_ave_bin(v1,v2,v_esc_gal=550.,vrms=None,v_rot_gal=[0.,220.,0.],v_sun_rot=[9.,12.,7.],v_earth_sun=[0.,0.,0]):
    v2=np.choose(v2==v1,[v2,v1+1e-6])
    return (eta0_halo_function_maxwellian_int(v2,v_esc_gal=v_esc_gal,vrms=vrms,v_rot_gal=v_rot_gal,v_sun_rot=v_sun_rot,v_earth_sun=v_earth_sun)-\
            eta0_halo_function_maxwellian_int(v1,v_esc_gal=v_esc_gal,vrms=vrms,v_rot_gal=v_rot_gal,v_sun_rot=v_sun_rot,v_earth_sun=v_earth_sun))/\
            (v2-v1)


def eta1_ave_bin(v1,v2,v_esc_gal=550.,vrms=None,v_rot_gal=[0.,220.,0.],v_sun_rot=[9.,12.,7.],v_earth_sun=[0.,0.,0]):

    v2=np.choose(v2==v1,[v2,v1+1e-6])
    return (eta1_halo_function_maxwellian_int(v2,v_esc_gal=v_esc_gal,vrms=vrms,v_rot_gal=v_rot_gal,v_sun_rot=v_sun_rot,v_earth_sun=v_earth_sun)-\
            eta1_halo_function_maxwellian_int(v1,v_esc_gal=v_esc_gal,vrms=vrms,v_rot_gal=v_rot_gal,v_sun_rot=v_sun_rot,v_earth_sun=v_earth_sun))/\
            (v2-v1)

    
    
def eta0_halo_function_maxwellian_int(v,v_esc_gal=550.,vrms=None,v_rot_gal=[0.,220.,0.],v_sun_rot=[9.,12.,7.],v_earth_sun=[0.,0.,0]):

    v_rot_gal=np.array(v_rot_gal)
    v_sun_rot=np.array(v_sun_rot)
    v_earth_sun=np.array(v_earth_sun)
    
    v_sun_gal=v_rot_gal+v_sun_rot

    if vrms is None:
        v0=np.sqrt(np.dot(v_rot_gal,v_rot_gal))
        vrms=np.sqrt(3./2.)*v0

    vearth_gal=np.sqrt(np.dot(v_sun_gal+v_earth_sun,v_sun_gal+v_earth_sun))

    eta=np.sqrt(3./2.)*vearth_gal/vrms
    z=np.sqrt(3./2.)*v_esc_gal/vrms
    N=1./(sp.erf(z)-2./np.sqrt(np.pi)*z*np.exp(-z**2))
    x=np.sqrt(3./2.)*v/vrms
    constant=N/eta/np.sqrt(np.pi)

    if z>eta:
        condition1=x<=z-eta
        condition2=x>=z+eta
    
        output1=constant*(np.sqrt(np.pi)/2.*((x+eta)*sp.erf(x+eta)-(x-eta)*sp.erf(x-eta))+1./2.*(np.exp(-(x+eta)**2)-np.exp(-(x-eta)**2))-2.*eta*x*np.exp(-z**2))
    
        output2=constant*(np.sqrt(np.pi)/2.*((x+eta)*sp.erf(z)-(x-eta)*sp.erf(x-eta))+1./2.*np.exp(-z**2)*(eta**2-2*eta*x-2*eta*z+x**2-2.*x*z+1.+z**2)-1./2.*np.exp(-(x-eta)**2))
        output3=constant*(np.sqrt(np.pi)*eta*sp.erf(z)-2.*eta*z*np.exp(-z**2))

    elif z<eta:
        condition1=x<=eta-z
        condition2=x>=eta+z

        #constant=N/eta/np.sqrt(np.pi)
        output1=x/eta

        output2=N/eta*1./np.sqrt(np.pi)*(np.sqrt(np.pi)/2.*((eta+x)*sp.erf(z)-(eta-x)*sp.erf(eta-x))+np.exp(-z**2)/2.*((eta-x)**2+z*(z-2*x-2*eta)+1)-1./2*np.exp(-(eta-x)**2))

        output3=1.

    return np.choose(condition1,[np.choose(condition2,[output2,output3]),output1])


def diff_eta0_halo_function_maxwellian_int(v,v_esc_gal=550.,vrms=None,v_rot_gal=[0.,220.,0],v_sun_rot=[9.,12.,7.],v_earth_sun=[0.,0.,0.]):

    v_rot_gal=np.array(v_rot_gal)
    v_sun_rot=np.array(v_sun_rot)
    v_earth_sun=np.array(v_earth_sun)
    
    v_sun_gal=v_rot_gal+v_sun_rot
    
    if vrms is None:
        v0=np.sqrt(np.dot(v_rot_gal,v_rot_gal))        
        vrms=np.sqrt(3./2.)*v0

    v_earth_gal=np.sqrt(np.dot(v_sun_gal+v_earth_sun,v_sun_gal+v_earth_sun))

        
    eta=np.sqrt(3./2.)*v_earth_gal/vrms
    delta_eta=np.sqrt(3./2.)*delta_vearth/vrms
    z=np.sqrt(3./2.)*v_esc_gal/vrms
    N=1./(sp.erf(z)-2./np.sqrt(np.pi)*z*np.exp(-z**2))
    x=np.sqrt(3./2.)*v/vrms
    constant=N/np.sqrt(np.pi)

    if z>eta:
        condition1=x<=z-eta
        condition2=x>=z+eta 

        output1=constant*(np.sqrt(np.pi)/2.*(sp.erf(x+eta)+sp.erf(x-eta))-2.*x*np.exp(-z**2))
    
        output2=constant*(np.sqrt(np.pi)/2.*(sp.erf(z)+sp.erf(x-eta))+(eta-x-z)*np.exp(-z**2))
    
        output3=constant*(np.sqrt(np.pi)*sp.erf(z)-2.*z*np.exp(-z**2))

    elif z<eta:
        condition1=x<=eta-z
        condition2=x>=eta+z

        output1=0.

        output2=((3./(2*np.pi*vrms**2))**(1/2.))/N/(4*np.sqrt(np.pi))*np.exp(z**2)*(np.exp(-(eta-z)**2)-np.exp(-z))*(x-eta+z)

        output3=((3./(2*np.pi*vrms**2))**(1/2.))/N/(4*np.sqrt(np.pi))*np.exp(z**2)*(np.exp(-(eta-z)**2)-np.exp(-z))*(2*z)

    return np.choose(condition1,[np.choose(condition2,[output2,output3]),output1])


def eta1_halo_function_maxwellian_int(v,v_esc_gal=550.,vrms=None,v_rot_gal=[0.,220.,0.],v_sun_rot=[9.,12.,7.],v_earth_sun=None):

    v_rot_gal=np.array(v_rot_gal)
    v_sun_rot=np.array(v_sun_rot)
    v_earth_sun=np.array(v_earth_sun)
    
    v_sun_gal=v_rot_gal+v_sun_rot

    if vrms is None:
        v0=np.sqrt(np.dot(v_rot_gal,v_rot_gal))        
        vrms=np.sqrt(3./2.)*v0

    v_earth_gal=np.sqrt(np.dot(v_sun_gal+v_earth_sun,v_sun_gal+v_earth_sun))

    eta=np.sqrt(3./2.)*v_earth_gal/vrms
    delta_eta=np.sqrt(3./2.)*delta_vearth/vrms


    return delta_eta/eta*(\
    diff_eta0_halo_function_maxwellian_int(v,v_esc_gal=v_esc_gal,vrms=vrms,v_rot_gal=v_rot_gal,v_sun_rot=v_sun_rot,v_earth_sun=v_earth_sun)-\
    eta0_halo_function_maxwellian_int(v,v_esc_gal=v_esc_gal,vrms=vrms,v_rot_gal=v_rot_gal,v_sun_rot=v_sun_rot,v_earth_sun=v_earth_sun)\
    )



class parameters(object):
    def __init__(self, input):
        self.dict=input
    def __str__(self):
        string=''
        for key,value in self.dict.items():
            string+=key+'='+str(value)+'\n'
        return string




def haxton_to_all_spins_sign(c):
    output=1.

    if c==3 or c==5 or c==6 or c==10 or c==11 or c==12 or c==13 or c==14 or c==15 or c==20:
        output=-1

    return output


def update_exp_info(exp,hamiltonian,j_chi=0.5):
    '''for experiment exp updates the info contained in the set of
    response functions files of hamiltonian and j_chi with the same
    content of the input files in the experiment directory
    '''
    load_response_functions(exp,hamiltonian,j_chi,verbose=False)
    outputdir=EXPERIMENTS_PATH+'/'+exp.name+'/Response_functions/spin_'+print_spin(j_chi)

    for (r_old,(__c1__,__c2__)) in zip(exp.response_functions[hamiltonian,j_chi],hamiltonian.coeff_squared_list):
     r_c1_c2=()
     for r in r_old[:-1]:
             r_c1_c2+=r,
     r_c1_c2+=get_info(exp),
     outputfile=outputdir+'/'+'c_'+print_coupling(__c1__)+'_c_'+print_coupling(__c2__)+'.npy'

     if os.path.isfile(outputfile):
         with open(outputfile, 'wb') as file:
             pickle.dump(r_c1_c2, file)
         print(outputfile+' info updated')
     else:
         
         print('could_not find '+outputfile)




Al=element('Al')
Ar=element('Ar')
C=element('C')
Ca=element('Ca')
F=element('F')
Fe=element('Fe')
Ge=element('Ge')
H=element('H')
He=element('He')
I=element('I')
Mg=element('Mg')
N=element('N')
Na=element('Na')
Ne=element('Ne')
Ni=element('Ni')
O=element('O')
S=element('S')
Si=element('Si')
W=element('W')
Xe=element('Xe')


element_list=[Al,Ar,C,Ca,F,Fe,Ge,H,He,I,Mg,N,Na,Ne,Ni,O,S,Si,W,Xe]

element_obj_names_string='WD.Al WD.Ar WD.C WD.Ca WD.F WD.Fe WD.Ge WD.H WD.He WD.I WD.Mg WD.N WD.Na WD.Ne WD.Ni WD.O WD.S WD.Si WD.W WD.Xe'

def list_elements():
    print('Available element objects in WimPyDD:')
    print('import WimPyDD as WD')
    print(element_obj_names_string)
    print('Use print() to get info on each element. For instance:')
    print('Type print(WD.Ge)')
    print(Ge)

def c_tau_SI(sigma_p,mchi,cn_over_cp=1):
    hbarc2=0.389e-27 #(hbar*c)^2 in GeV^2 * cm^2
    mn=0.931
    mu=mchi*mn/(mchi+mn)
    return np.sqrt(np.pi*sigma_p/hbarc2)/mu*np.array([1+cn_over_cp,1-cn_over_cp])

    
SI=eft_hamiltonian('Spin-independent',{1: c_tau_SI})
    

def c_tau_SD(sigma_p,mchi,cn_over_cp=1):
    hbarc2=0.389e-27 #(hbar*c)^2 in GeV^2 * cm^2
    mn=0.931
    mu=mchi*mn/(mchi+mn)
    return np.sqrt(16/3*np.pi*sigma_p/hbarc2)/mu*np.array([1+cn_over_cp,1-cn_over_cp])

    
SD=eft_hamiltonian('Spin-dependent',{4: c_tau_SD})

maxwellian_halo_function=streamed_halo_function()
maxwellian_halo_function_yearly_modulation=streamed_halo_function(yearly_modulation=True)


XENON_1T_2018=experiment('XENON_1T_2018')
PICO60_2019=experiment('PICO60_2019')
DAMA_LIBRA_2019=experiment('DAMA_LIBRA_2019')


def update_response_functions_time_stamps():

    exp_dir_list=os.listdir('WimPyDD/Experiments')
    
    for exp_dir in exp_dir_list:

        if os.path.isdir('WimPyDD/Experiments/'+exp_dir+'/Response_functions/'):
        
            spin_list=[s for s in os.listdir('WimPyDD/Experiments/'+exp_dir+'/Response_functions/') if 'spin' in s]

            for spin in spin_list:
            
                npy_list=[s for s in os.listdir('WimPyDD/Experiments/'+exp_dir+'/Response_functions/'+spin+'/') if '.npy' in s]
                if len(npy_list)>0:
                    for npy_i in range(len(npy_list)):
                        win_path('WimPyDD/Experiments/'+exp_dir+'/Response_functions/'+spin+'/'+npy_list[npy_i]).touch()
                    #p = subprocess.call("touch "+'WimPyDD/Experiments/'+exp_dir+'/Response_functions/'+spin+'/*.npy', stdout=subprocess.PIPE, shell=True)



def get_mapping(hamiltonian,pn=False):
    '''
    Returns a dictionary containing the mapping between the couplings of an effective Hamiltonian and the indices of a vector.
    Used by wimp_dd_matrix.
    ----------------------------------------------
    Input
    
    hamiltonian: Hamiltonian object instantiated with the eft_hamiltonian class
    pn: if True the couplings are in the proton-neutron base, if False in the isospin base (default: False)
    ----------------------------------------------
    Example:

    If:
    >>>hamiltonian.wilson_coefficients.keys()
    >>>[1,3]

    >>>m=WD.get_mapping(hamiltonian)
    >>>m
    >>>{(1, 0): 0, (1, 1): 1, (3, 0): 2, (3, 1): 3}

    index   coupling
    ----------------
    0        c_1^0
    1        c_1^1
    2        c_3^0
    3        c_3^1
    ----------------

    if pn=True:
    >>>m=WD.get_mapping(hamiltonian, pn=True)
    >>>{(1, 'p'): 0, (1, 'n'): 1, (3, 'p'): 2, (3, 'n'): 3}

    index   coupling
    ----------------
    0        c_1^p
    1        c_1^n
    2        c_3^p
    3        c_3^n
    ----------------
    N.B. The output matrix M of wimp_dd_matrix is always in the isospin base. The mapping in the proton-neutron base can be used after rotating M to the matrix M_pn with:

    U=WD.rotation_from_isospin_to_pn(hamiltonian)
    M_pn=np.dot(U,np.dot(M,U))
    '''

    mapping={}
    n=0
    for c in hamiltonian.couplings:
        mapping[c,0]=n
        n+=1
        mapping[c,1]=n
        n+=1
    if pn:
        mapping={((c,'p') if tau==0 else (c,'n')):v for (c,tau),v in mapping.items()}

    return mapping

def wimp_dd_matrix(exp,hamiltonian,n_bin,vmin,delta_eta,mchi,delta=0,j_chi=0.5,verbose=False,**args):
    '''                                                                         
    Outputs the matrix M that calculates the direct-detection rate R for a couplings array c:

    R=np.dot(c,np.dot(M),c))

    for a given experiment, hamiltonian, energy bin and velicity distribution.

    A dictionary with the mapping between the array components of c and the couplings of the hamiltonian is provided by the routine get_mapping.

    N.B. The output matrix M of wimp_dd_matrix is always in the isospin base. The M_pn in the proton-neutron base can be obtained using:

    U=WD.rotation_from_isospin_to_pn(hamiltonian)

    M_pn=np.dot(U,np.dot(M,U))
    ----------------------------------------------
    Input
    
    exp: object belonging to experiment class
    hamiltonian: object belonging to eft_hamiltonian class
    n_bin: integer that identifies one of the experimental bins as initialized in data.tab
    vmin:array containing a list of WIMP stream speed velocities in the lab frame in km/s.
    delta_eta: array containing the contribution of each stream to the halo function eta(v) in (km/sec)^-1
               (the routine streamed_halo_function can be used to calculate it)
    mchi: WIMP mass in GeV
    delta: Mass splitting for inelastic scattering in keV (default value=0)
    j_chi: WIMP spin (default value=0.5)
    verbose: Passed to load_response_functions, if called. (default value: False)
    ----------------------------------------------
    Example:
    
    >>>c_1_3=WD.eft_hamiltonian('c_1_3',{1: lambda: [1,1],3: lambda: [1,1]})
    >>>mapping=WD.get_mapping(c_1_3)
    
    >>>mapping[3,0]
    >>>2

    the index of the coupling c_3^0 is 2.

    >>>m=WD.wimp_dd_matrix(experiment,c_1_3,n_bin,vmin,delta_eta,mchi)
    '''

    h=hamiltonian

    dim=2*len(hamiltonian.couplings)
    mapping=get_mapping(hamiltonian)
    matrix=np.zeros(dim*dim).reshape(dim,dim)
    for tau in range(2):
        for tau_prime in range(2):
            wimp_dd_rate(exp, h, vmin, delta_eta, mchi,j_chi=j_chi, delta=delta,tau_range=[tau],tau_prime_range=[tau_prime],verbose=verbose,**args)
            for c1,c2 in hamiltonian.coeff_squared_list:
                m=mapping[c1,tau]
                m_prime=mapping[c2,tau_prime]
                matrix[m,m_prime]=wimp_dd_rate.contributions[n_bin,(c1,c2)]
                if non_symmetric_interference(c1,c2):
                    matrix[m,m_prime]=matrix[m,m_prime]/2.

                matrix[m_prime,m]=matrix[m,m_prime]

    return (matrix+matrix.T)/2


def rotation_from_isospin_to_pn(hamiltonian):
    '''                                                                         
    Returns a matrix rotating isospin base to proton-neutron base.              
    ----------------------------------------------                              
    Input                                                                       
                                                                                
    hamiltonian: object belonging to eft_hamiltonian class                      
    '''
    mapping=get_mapping(hamiltonian)
    mapping_pn={((c,'p') if tau==0 else (c,'n')):v for (c,tau),v in mapping.items()}
    inv_mapping_pn={v:k for k,v in mapping_pn.items()}
    n_dim=len(mapping)
    rotation=np.zeros(n_dim*n_dim).reshape(n_dim,n_dim)
    for c in hamiltonian.couplings:
        n1=mapping[c,0]
        n2=mapping[c,1]
        rotation[n1,n1]=1
        rotation[n1,n2]=1
        rotation[n2,n1]=1
        rotation[n2,n2]=-1
    return rotation
