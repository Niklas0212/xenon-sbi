import WimPyDD as WD
import numpy as np

def func(element,isotope,q):
     #initializes the output to default
     out=WD.default_nuclear_response_functions(element,isotope,
     q)
     # from table 1 of 1208.1094 (one-body currents)
     s00=[0.054731,-0.146897,0.182479,-0.128112,0.0539978,
     -0.0133335,0.00190579,-1.48373e-4, 5.11732e-6,-2.06597e-8]
     s11=[0.048192,-0.148361,0.202347,-0.151853,0.0674284,
     -0.0179342,0.00286368,-2.65795e-4,1.2965e-5,-2.47418e-7]
     s01=[-0.102732,0.297105, -0.387513,0.281816,-0.122388,
     0.0317668,-0.0049233, 4.39836e-4,-2.02852e-5,3.46755e-7]
     hbarc=197.e-3 #in GeV*fm, q is in GeV
     b=2.2853 # in fm, from the caption of Table 1 of 1208.1094
     u=(b*q/hbarc)**2/2
     #overwrites 'Sigma_prime_prime'
     n1=WD.nuclear_current['Sigma_prime_prime']
     out[n1,0,0]=1/3*np.sum(np.array([coeff*u**n for n,coeff in
     enumerate(s00)])*np.exp(-u))
     out[n1,0,1]=1/6*np.sum(np.array([coeff*u**n for n,coeff in
     enumerate(s01)])*np.exp(-u))
     out[n1,1,0]=out[1,0,1]
     out[n1,1,1]=1/3*np.sum(np.array([coeff*u**n for n,coeff in
     enumerate(s11)])*np.exp(-u))
     #overwrites 'Sigma_prime'
     n2=WD.nuclear_current['Sigma_prime']
     out[n2]=2*out[n1]
     return out
    # all other response functions are not modified, default from 1308.6288 used.
    
    



