from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

class Integrator:
    
    def __init__(self,x,f):
        self.f = f
        self.x = x
        self.h = self.x[1] - self.x[0]
        self.y = f(self.x)
        
        self.Integral = 0.
        
    def GetIntegral(self):
        '''Calcula la integral con el metodo del trapecio'''
        
        self.Integral += 0.5*(self.y[0]+self.y[-1])
        
        #self.Integral += np.sum( self.y[1:-1] )
        
        for i in range(1,self.y.shape[0]-1):
            self.Integral += self.y[i]
        
        self.Integral *= self.h
        
        return self.Integral

        
class Simpson(Integrator):
    
    def __init__(self,x,f):
        Integrator.__init__(self,x,f)
        
    def Simple(self):
        '''Calcula la integral con el metodo de Simpson 1/3 (simple)'''
        midpoint = 0.5*(self.x[0]+self.x[-1])
        
        self.Integral = 0
        f_midpoint = self.f(midpoint)
        self.Integral +=self.y[0]+ f_midpoint +self.y[-1]
        
        return self.Integral*self.h/3
    
    def GetSimpleError(self):
        '''Calcula el error asociado a Simpson 1/3 (simple)'''
        d = self.GetDerivative()
        #print(f"antes de quitar nans {d}")
        d = d[~np.isnan(d)]
        #print(f"despues de quitar nans {d}")
        
        max_ = np.max(np.abs(d))
        
        self.error = self.h**5*max_/90
        
        return self.error
        
        
    def GetIntegral(self):
        '''Calcula la integral con el metodo de Simpson Compuesto'''
        
        self.Integral = 0.
        
        self.Integral += self.y[0] + self.y[-1]
        
        for i in range( len(self.y[1:-1]) ):
            
            if i%2 == 0:
                self.Integral += 4*self.y[i+1]
            else:
                self.Integral += 2*self.y[i+1]
          
        return self.Integral*self.h/3
    
    def GetDerivative(self):
        d = self.f(self.x + 2*self.h) - 4*self.f(self.x + self.h) + 6*self.f(self.x) - 4*self.f(self.x - self.h) + self.f(self.x - 2*self.h)
        d /= self.h**4
        
        return d
    
    def GetError(self):
        '''Calcula el error asociado a Simpson Compuesto'''
        
        d = self.GetDerivative()
        #print(f"antes de quitar nans {d}")
        d = d[~np.isnan(d)]
        #print(f"despues de quitar nans {d}")
        
        max_ = np.max(np.abs(d))
        
        self.error = (self.x[-1]-self.x[0])*self.h**4*max_/180
        
        return self.error