from funciones import GetNewton
import numpy as np 
import sympy as sym

x = sym.Symbol('x',real=True)

def GetLaguerre(n,x):
    
    if n==0:
        return 1
    
    elif n==1:
        return 1-x
    
    else:
        poly = ((2*(n-1)+1-x)*GetLaguerre(n-1,x)-(n-1)*GetLaguerre(n-2,x))/n
        return poly
    
    
def GetAllRootsGlag(n):
    
    tolerancia = 14
    xn = np.linspace(0,n+(n-1)*np.sqrt(n),50)
    
    poly = GetLaguerre(n,x)
    df_poly = sym.diff(poly,x,1)
    
    poly = sym.lambdify(x,poly,'numpy')
    df_poly = sym.lambdify(x,df_poly,'numpy')
    
    Roots = np.array([])
    
    for i in xn:
        
        root = GetNewton(poly,df_poly,i)
        #print(type(root))
        if  type(root)!=bool:
            croot = np.round(root, tolerancia)
            if croot not in Roots:
                Roots = np.append(Roots, croot)
                
    Roots.sort()
    
    return Roots

def GetWeightsGLag(n):
    Roots = GetAllRootsGlag(n)
    #n+1-esimo polinomio de laguerre
    poly = GetLaguerre(n+1,x)
    poly = sym.lambdify(x,poly,'numpy')
    return Roots/(((n+1)**2)*(poly(Roots))**2)


def GetHermite(n,x):
    if n==0:
        return 1
    elif n==1:
        return 2*x
    else:
        return 2*x*GetHermite(n-1,x)-2*(n-1)*GetHermite(n-2,x)
    
def GetAllRootsGHer(n):
    tolerancia = 14
    xn = np.linspace(-np.sqrt(4*n+1),np.sqrt(4*n+1),50)
    
    poly = GetHermite(n,x)
    df_poly = sym.diff(poly,x,1)
    
    poly = sym.lambdify(x,poly,'numpy')
    df_poly = sym.lambdify(x,df_poly,'numpy')
    
    Roots = np.array([])
    
    for i in xn:
        
        root = GetNewton(poly,df_poly,i)
        #print(type(root))
        if  type(root)!=bool:
            croot = np.round(root, tolerancia)
            if croot not in Roots:
                Roots = np.append(Roots, croot)
                
    Roots.sort()
    
    return Roots

def GetWeightsGHer(n):
    Roots = GetAllRootsGHer(n)
    #n-1 -esimo polinomio de hermite
    poly = GetHermite(n-1,x)
    poly = sym.lambdify(x,poly,'numpy')
    numerador = 2**(n-1)*np.math.factorial(n)*np.sqrt(np.pi)
    denominador = n**2*(poly(Roots))**2
    return numerador/denominador
    