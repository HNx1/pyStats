# We implement a number of optimizers in a linear regression setting
# This LinearOptimiser class can easily be extended to general regression or classification settings 
# just override the grad method (as it contains an analytic gradient which doesn't generalise), with a new cost function if needed (for example in a classification setting)
# model: ys = xs @ theta + eps
# Nesterov/momentum often needs a lower learning rate than the default 1e-3
# AMSgrad and ADAM need higher learning rates in such a simple setting to have comparable convergence 
import numpy as np


class LinearOptimiser():
    def __init__(self,xs,ys):
        xs = [[1]+x for x in xs]  if type(xs[0])==list else [[1,x]for x in xs]  
        self.xs=np.array(xs)
        self.ys=np.array(ys)
        self.theta=np.zeros(len(xs[0]))
        self.phi=np.zeros(len(xs[0]))
        self.psi=np.zeros(len(xs[0]))
        self.lossi=[self.RSS()]
        self.mus=[0.3]
        self.betas=[0.9,0.999]
        self.adamEps=10e-8


    def reset(self):
        # reset parameters
        self.theta=np.zeros(len(self.xs[0]))
        self.lossi=[self.RSS()]
        self.mus=[0.1]
        self.phi=np.zeros(len(self.xs[0]))
        self.psi=np.zeros(len(self.xs[0]))
    
    def RSS(self):
        # gets residual sum of squares
        eps = self.ys - self.xs @ self.theta
        #Equivalent to (y-x*theta)T * (y-x*theta)
        return np.dot(eps,eps)

    def minRSS(self):
        # Return analytical minimum of RSS if it exists
        self.analytic()
        return self.RSS()


    def analytic(self):
        self.reset()
        # Finds analytic solution of linear regression, use try except block as matrix inversion may fail
        try:
            self.theta = np.linalg.inv(np.transpose(self.xs)@self.xs) @ np.transpose(self.xs) @ self.ys
        except:
            print("Cannot invert X_T * X")
    
    def grad(self,dropout,drop):
        # Returns gradients of objective function (RSS) with respect to parameters (theta): 2*(X.T * X * theta - X.T * Y). Supports dropout
        grads=2*np.transpose(self.xs) @ (self.xs @ self.theta - self.ys)
        if dropout:
            grads *= np.random.binomial(size=grads.shape, n=1, p=1-drop)
        return grads
    
    def optimize(self,func,iters=100,lr=1e-3,reset=True,dropout=False,drop=0.2):
        if reset:
            self.reset()
        # Finds an optimizer solution for theta
        for t in range(iters):
            func(lr,dropout,drop,t+1)
            self.lossi.append(self.RSS())

    def classic(self,lr,dropout,drop,t):
        self.theta += -lr*self.grad(dropout,drop)
    
    def momentum(self,lr,dropout,drop,t):
        self.phi=-lr*self.grad(dropout,drop) +self.phi*self.mus[0]
        self.theta += self.phi
    
    def nesterov(self,lr,dropout,drop,t):
        # Nesterov Accelerated Gradient (standard not Bengio/Sutskever)
        phi=self.theta-lr*self.grad(dropout,drop)
        self.theta = phi+ self.mus[0]*(phi-self.phi)
        self.phi=phi
        return
    
    def adam(self,lr,dropout,drop,t):
        g=self.grad(dropout,drop)
        self.phi=self.betas[0]*self.phi+(1-self.betas[0])*g
        self.psi=self.betas[1]*self.psi+(1-self.betas[1])*g**2       
        phi=self.phi/(1-self.betas[0]**t)
        psi=self.psi/(1-self.betas[1]**t)
        self.theta += -lr*phi/(np.sqrt(psi)+self.adamEps)
    
    def amsGrad(self,lr,dropout,drop,t):
        g=self.grad(dropout,drop)
        self.phi=self.betas[0]*self.phi+(1-self.betas[0])*g
        psi=self.betas[1]*self.psi+(1-self.betas[1])*g**2 
        self.psi=np.maximum(psi,self.psi)
        self.theta += -lr*self.phi/(np.sqrt(self.psi)+self.adamEps)

    def rmsProp(self,lr,dropout,drop,t):
        # Store E[g**2] in self.psi
        g=self.grad(dropout,drop)
        self.psi=self.betas[0]*self.psi+(1-self.betas[0])*g**2
        self.theta += -lr*g/np.sqrt(self.psi)
