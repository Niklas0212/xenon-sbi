import numpy as np

def f(er,eprime,exp,n_element,n_isotope,n_bin):
    if n_bin==0:
        '''Fig. 3 of 1902.04031 upper one'''
        if exp.target.element[n_element].name=='fluorine':
            
            x_f=np.array([3.56, 3.67, 3.69, 4.10, 5.88])
            y_f=np.array([0.0, 0.192, 0.501, 0.800, 1.0])
            return np.interp(er,x_f,y_f)
        
        elif exp.target.element[n_element].name=='carbon':
            
            x_c=np.array([6.44, 6.53, 10.24, 11.56, 13.09])
            y_c=np.array([0.0, 0.17, 0.495, 0.799, 1.0])
            return np.interp(er,x_c,y_c)
        
        else:
            return 1.

    elif n_bin==1:
        '''Fig. 3 of 1902.04031 lower one'''
        if exp.target.element[n_element].name=='fluorine':

            x_f=np.array([3.64, 3.76, 3.78, 4.20, 6.03])
            y_f=np.array([0.0, 0.18, 0.49, 0.797, 1.0])
            return np.interp(er,x_f,y_f)
        
        elif exp.target.element[n_element].name=='carbon':

            x_c=np.array([7.13, 7.19, 10.73, 13.23, 13.87])
            y_c=np.array([0.0, 0.19, 0.501, 0.792, 1.0])
            return np.interp(er,x_c,y_c)
        
        else:
            return 1.
