#En este modulo se encuentran funciones auxiliares
import numpy as np

def GetNewton(f,df,xn,itmax=10000,precision=1e-14):
    '''
    Implementacion de Newton Rhapson
    '''
    error = 1.
    it = 0
    
    while error >= precision and it < itmax:
        
        try:
            
            xn1 = xn - f(xn)/df(xn)
            
            error = np.abs(f(xn)/df(xn))
            
        except ZeroDivisionError:
            print('Zero Division')
            
        xn = xn1
        it += 1
        
    if it == itmax:
        print("itmax reached")
        return False
    else:
        return xn
    
