import numpy as np
def f(e_prime,e_ee):
    '''refer to 1002.1028 for details.'''
    a_res=0.0091
    b_res=0.448
    c_res=0.0

    sigma=a_res*e_ee+b_res*np.sqrt(e_ee)+c_res

    res=0.
    if e_prime>0.1:
        res=1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-(e_prime-e_ee)**2/(2.*sigma**2))
    return res
