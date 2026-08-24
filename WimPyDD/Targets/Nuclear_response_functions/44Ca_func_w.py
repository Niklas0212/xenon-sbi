from numpy import *
def func_w_custom(element,isotope,q):
    '''Definition in 44ca_func_w.py used:\nfor M interaction used Helm factor, no the spin-dependent form factors\n(even-even isotope), form factors for all other interactions are missing,\nso they are set to zero.'''


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

    #Haxton definitions for M, sigma'' and sigma' interactions are overwritten with cutom ones.

    # 01)M
    output[0,0,0]=(2.*j_nucleus+1.)/(16.*pi)*a_nucleus**2*helm_form_factor_squared(element,isotope,q)
    output[0,0,1]=(2.*j_nucleus+1.)/(16.*pi)*a_nucleus*(2.*z_nucleus-a_nucleus)*helm_form_factor_squared(element,isotope,q)
    output[0,1,0]=output[0,0,1]
    output[0,1,1]=(2.*j_nucleus+1.)/(16.*pi)*(2.*z_nucleus-a_nucleus)**2*helm_form_factor_squared(element,isotope,q)    
    

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


