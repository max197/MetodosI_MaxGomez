import numpy as np

class Robot:
    
    def __init__(self, dt, Layers, Id=0,overfitPenalty = 0.05):
        
        self.Id = Id
        self.dt = dt
        
        
        #Desplazamiento
        self.r = np.random.uniform([0.,0.])
        
        #Vector velocidad
        theta = 0
        self.v = np.array([1.*np.cos(theta),1.*np.sin(theta)])

        # Capacidad o aptitud del individuo
        self.Fitness = np.inf
        self.overfitPenalty = overfitPenalty
        
        #Steps controla que tanto avanza el robot en la region de interes.
        #Durante el movimiento el robot aumenta el valor de esta variable en una unidad si
        #la partcula se encuentra entre -1 y 1
        self.Steps = 0

        # Brain
        self.Layers = Layers
        
    def GetR(self):
        return self.r
    
    def Evolution(self):
        self.r += self.v*self.dt 
        #Si -1+epsilon<= x <=1-epsilon incrementamos el numero de pasos en 1
        epsilon = 0.5
        if self.r[0]<=1-epsilon and self.r[0]>=-1+epsilon:
            self.Steps += 1

        if self.r[0]>1 or self.r[0]<-1:
            self.Steps= 0
    # Cada generación regresamos el robot al origin
    # Y volvemos a estimar su fitness
    def Reset(self):
        self.Steps = 0.
        self.r = np.array([0.,0.])
        self.Fitness = np.inf    
        
    # Aca debes definir que es mejorar en tu proceso evolutivo
    def SetFitness(self):
        if self.Steps<= 0:
            self.Fitness = np.inf
        else:
            self.Fitness = 1/self.Steps
        
    def BrainActivation(self,x,threshold=0.7): 
        # Forward pass - la infomación fluye por el modelo hacia adelante
        for i in range(len(self.Layers)):   
            if i == 0:
                output = self.Layers[i].Activation(x)
            else:
                output = self.Layers[i].Activation(output)
        
        self.Activation = np.round(output,4)

    
        # Cambiamos el vector velocidad
        if self.Activation[0] > threshold:
            self.v = -self.v
            
            # Deberias penalizar de alguna forma, dado que mucha activación es desgastante!
            #penalizacion
            self.Steps = self.overfitPenalty*self.Steps
            

    
        return self.Activation
    
    # Aca mutamos (cambiar de parametros) para poder "aprender"
    def Mutate(self):
        for i in range(len(self.Layers)):
            self.Layers[i].Mutate()
    
    # Devolvemos la red neuronal ya entrenada
    def GetBrain(self):
        return self.Layers