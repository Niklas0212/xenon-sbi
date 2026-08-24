from numpy import *
def f(element,isotope,q):
    '''Definition in 183w_func_w.py used:\nfor M interaction used Helm factor, for Sigma'' and Sigma' interactions\nused definitions from Appendix C of 1505.01926 with sn=-0.17,sp=0\n((table XII of hep-ph/0406218), form factors for all other interactions are missing,\nso they are set to zero.'''

    #For M interaction used Helm factor, for Sigma'' and Sigma'
    #interactions used definitions from Appendix C of 1505.01926.  For
    #all other interactions Haxton definitions from 1308.6288 are
    #missing, so they are set to zero. For the spin-dependent for
    #factors we take the approximation in appendix C of 1505.01926 with
    #the 183w nuclear spins carried by protins and neutrons taken from
    #table XII of hep-ph/0406218 (ISPSM, Ellis-Floris,
    #Phys.Lett. B263(1991)259-266; Nucle. Phys. B400 (1993)25-36)

    

    
    hbarc=197.e-3 #GeV*fm, q is supposed to be in GeV    
    a_nucleus=element.a[isotope]
    b=sqrt(41.467e0/(45.e0*a_nucleus**(-1./3.)-25.*a_nucleus**(-2./3.)))    
    y=(b*q/(2.e0*hbarc))**2
    
                        
    output=func_w(element.data_mod[isotope],y)    
    hbarc=197.e-3 #GeV*fm, q is supposed to be in GeV 
    a_nucleus=element.a[isotope]
    z_nucleus=element.z
    j_nucleus=element.spin[isotope]
    r=0.92*a_nucleus**(1./3.)+2.68-0.78*sqrt((a_nucleus**(1./3.)-3.8)**2+0.2)
    b=r/sqrt(2.)
    y=(b*q/(2.e0*hbarc))**2

    #Haxton definitions for M, sigma'' and sigma' interactions are overwritten with custom ones.

    # 01)M
    output[0,0,0]=(2.*j_nucleus+1.)/(16.*pi)*a_nucleus**2*helm_form_factor_squared(element,isotope,q)
    output[0,0,1]=(2.*j_nucleus+1.)/(16.*pi)*a_nucleus*(2.*z_nucleus-a_nucleus)*helm_form_factor_squared(element,isotope,q)
    output[0,1,0]=output[0,0,1]
    output[0,1,1]=(2.*j_nucleus+1.)/(16.*pi)*(2.*z_nucleus-a_nucleus)**2*helm_form_factor_squared(element,isotope,q)    
    
    if j_nucleus==0:
        return output
    
    sn=-0.17
    sp=0.


    s=array([(sp+sn)/2.,(sp-sn)/2.])
    s_tau_tau_prime=kron(s,s).reshape(2,2)

    r=0.92*a_nucleus**(1./3.)+2.68-0.78*sqrt((a_nucleus**(1./3.)-3.8)**2+0.2)
    z=q*r/hbarc
    # 02) Sigma''
    output[1,0,0]=4./(3*pi)*((2*j_nucleus+1.)*(j_nucleus+1.))/j_nucleus*s_tau_tau_prime[0,0]*exp(-z**2/4.)
    output[1,0,1]=4./(3*pi)*((2*j_nucleus+1.)*(j_nucleus+1.))/j_nucleus*s_tau_tau_prime[0,1]*exp(-z**2/4.)
    output[1,1,0]=output[1,0,1]
    output[1,1,1]=4./(3*pi)*((2*j_nucleus+1.)*(j_nucleus+1.))/j_nucleus*s_tau_tau_prime[1,1]*exp(-z**2/4.)    


    # 03) Sigma'
    output[2,0,0]=output[1,0,0]*2.
    output[2,0,1]=output[1,0,1]*2.
    output[2,1,0]=output[1,1,0]*2.
    output[2,1,1]=output[1,1,1]*2.    

    return output
    


def helm_form_factor_squared(element,isotope,q):
    '''Helm form factor (Helm R 1956 Phys. Rev. 104 1466), see introduction of hep-ph/0608035 for r1 parameterization.'''
    if q==0:
        return 1
    
    a=0.52 #in fm
    s=0.9 #in fm
    c=1.23*element.a[isotope]**(1./3.)-0.6 # in fm
    r1=sqrt(c**2+7./3.*pi**2*a**2-5*s**2)
    hbarc=197.e-3 #GeV*fm, q is supposed to be in GeV
    x=q*r1/hbarc
    y=q*s/hbarc
    j1=sin(x)/x**2-cos(x)/x
    return (3.*j1/x)**2*exp(-y**2) 


def func_w(a,y):
    f=0.
    for i in range(a.shape[len(a.shape)-1]):
        f+=a[...,i]*y**i
        
    return f*exp(-2*y)



