import numpy as np

def DerivadaCentral(f,x,h):
    d = 0.
    
    if h != 0:
        d = (f(x+h) - f(x-h))/(2*h)
        
    return d

def DerivadaDerecha(f,x,h):
    
    d = 0.
    
    if h != 0:
        d = (f(x+h) - f(x))/h
        
    return d

def Lagrange(x,X,i):
    
    L = 1
    
    for j in range(X.shape[0]):
        if i != j:
            L *= (x - X[j])/(X[i]-X[j])
            
    return L

def Interpolate(x,X,Y):
    
    Poly = 0
    
    for i in range(X.shape[0]):
        Poly += Lagrange(x,X,i)*Y[i]
        
    return Poly