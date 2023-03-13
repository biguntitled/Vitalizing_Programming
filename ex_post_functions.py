import numpy as np
import scipy.optimize as optimize
from scipy import interpolate
from numba import njit, int64, double
import math


def solve_backwards(beta,W,T):
    # 2. Initialize
    Vstar_bi = np.nan+np.zeros([W+1,T])
    Cstar_bi = np.nan + np.zeros([W+1,T])
    Cstar_bi[:,T-1] = np.arange(W+1) 
    Vstar_bi[:,T-1] = np.sqrt(Cstar_bi[:,T-1])
    # 3. solve
    
    # Loop over periods
    for t in range(T-2, -1, -1):  #from period T-2, until period 0, backwards  
        
        #loop over states
        for w in range(W+1):
            c = np.arange(w+1)
            w_c = w - c
            V_next = Vstar_bi[w_c,t+1]
            V_guess = np.sqrt(c)+beta*V_next
            Vstar_bi[w,t] = np.amax(V_guess)
            Cstar_bi[w,t] = np.argmax(V_guess)

    return Cstar_bi, Vstar_bi

def solve_VFI(par):
    Cstar = np.zeros([par.W+1])
    
    # Parameters for VFI
    max_iter = par.max_iter   # maximum number of iterations
    delta = 1000 #difference between V_next and V_now
    tol = par.tol #convergence tol. level
    it = 0  #iteration counter 
    V_now = np.zeros([par.W+1]) #arbitrary starting values
    
    while (max_iter>= it and tol<delta):
        it = it+1
        V_next = V_now.copy()
        for w in range(par.W+1):
            c = np.arange(w+1)
            w_c = w - c
            V_guess = np.sqrt(c)+par.beta*V_next[w_c]
            V_now[w] = np.amax(V_guess)
            Cstar[w] = np.argmax(V_guess)
        delta = np.amax(np.abs(V_now - V_next))
    
    class sol: pass
    sol.C = Cstar
    sol.V = V_now
    sol.it = it

    return sol


def solve_consumption_grid_search(par):
     # initialize solution class
    class sol: pass
    sol.C = np.zeros(par.num_W)
    sol.V = np.zeros(par.num_W)
    
    # consumption grid as a share of available resources
    grid_C = np.linspace(0.0,1.0,par.num_C) 
    
    # Resource grid
    grid_W = par.grid_W

    # Init for VFI
    delta = 1000 #difference between V_next and V_now
    it = 0  #iteration counter 
    
    while (par.max_iter>= it and par.tol<delta):
        it = it+1
        V_next = sol.V.copy()
        for iw,w in enumerate(grid_W):  # enumerate automaticcaly unpack w
            c = grid_C*w
            w_c = w - c
            V_guess = np.sqrt(c)+par.beta*np.interp(w_c,grid_W,V_next)
            index = np.argmax(V_guess)
            sol.C[iw] = c[index]
            sol.V[iw] = np.amax(V_guess)
        delta = np.amax(np.abs(sol.V - V_next))
    
    return sol

def solve_consumption_uncertainty(par):
     # initialize solution class
    class sol: pass
    sol.V = np.zeros([par.num_W,par.T]) 
    sol.C = np.zeros([par.num_W,par.T])
    sol.grid_W = np.zeros([par.num_W,par.T])
    
    # consumption grid as a share of available resources
    grid_C = np.linspace(0.0,1.0,par.num_C)
    
    # Loop over periods
    for t in range(par.T-1, -1, -1):  #from period T-1, until period 0, backwards 
        W_max = max(par.eps)*t+par.W
        grid_W = np.linspace(0,W_max,par.num_W) 
        sol.grid_W[:,t] = grid_W
    
        for iw,w in enumerate(grid_W):
            c = grid_C*w
            w_c = w - c
            EV_next = 0
        
            if t<par.T-1:
                for s in range(par.K):
                    # weight on the shock 
                    weight = par.pi[s]
                    # epsilon shock
                    eps = par.eps[s]
                    # expected value
                    EV_next +=weight*np.interp(w_c+eps,sol.grid_W[:,t+1],sol.V[:,t+1])
            V_guess = np.sqrt(c)+par.beta*EV_next
            index = np.argmax(V_guess)
            sol.C[iw,t] = c[index]
            sol.V[iw,t] = np.amax(V_guess)
        
    return sol



def util(c,par):
    return (c**(1.0-par.rho))/(1.0-par.rho)

def solve_consumption_deaton(par):
     # initialize solution class
    class sol: pass
    sol.V = np.zeros([par.num_W,par.T]) 
    sol.C = np.zeros([par.num_W,par.T])
    sol.grid_W = np.zeros([par.num_W,par.T])
    
    # consumption grid as a share of available resources
    grid_C = np.linspace(0.0,1.0,par.num_C)
    
    # Loop over periods
    for t in range(par.T-1, -1, -1):  #from period T-1, until period 0, backwards 
        W_max = max(par.eps)*t+par.W
        grid_W = np.linspace(0,W_max,par.num_W) 
        sol.grid_W[:,t] = grid_W
    
        for iw,w in enumerate(grid_W):
            c = grid_C*w
            w_c = w - c
            EV_next = 0
        
            if t<par.T-1:
                for s in range(par.num_shocks):
                    # weight on the shock 
                    weight = par.eps_w[s]
                    # epsilon shock
                    eps = par.eps[s]
                    # next period assets
                    w_next = par.R*w_c+eps
                    # expected value
                    EV_next +=weight*np.interp(w_next,sol.grid_W[:,t+1],sol.V[:,t+1])
            V_guess = util(c,par)+par.beta*EV_next
            index = np.argmax(V_guess)
            sol.C[iw,t] = c[index]
            sol.V[iw,t] = np.amax(V_guess)
        
    return sol

def setup():
    # Setup specifications in class. 
    class par: pass
    par.beta = 0.90
    par.rho = 0.5
    par.R = 1.0
    par.sigma = 0.2
    par.mu = 0
    par.W = 10
    
    # Shocks and weights
    par.num_shocks = 5
    x,w = gauss_hermite(par.num_shocks)
    par.eps = np.exp(par.sigma*np.sqrt(2)*x)
    par.eps_w = w/np.sqrt(np.pi)
    
    # Grid
    par.num_W = 200
    par.num_C = 200
    par.grid_W = np.linspace(0,par.W,par.num_W)
    
    # Parameters for VFI
    par.max_iter = 200   # maximum number of iterations
    par.tol = 10e-2 #convergence tol. level 
        
    return par


def solve_deaton_infty(par):
     # initialize solution class
    class sol: pass
    sol.V = np.zeros([par.num_W]) 
    sol.C = np.zeros([par.num_W])
    
    # consumption grid as a share of available resources
    grid_C = np.linspace(0.0,1.0,par.num_C)
    
    # initialize counter
    sol.it = 0
    sol.delta = 2000

    while (par.max_iter>= sol.it and par.tol<sol.delta):
        
        V0 = sol.V.copy()
        interp = interpolate.interp1d(par.grid_W,V0, bounds_error=False, fill_value = "extrapolate")
        
        for iw,w in enumerate(par.grid_W):
            c = grid_C*w
            w_c = w - c
            EV_next = 0
            
            for s in range(par.num_shocks):
                # weight on the shock 
                weight = par.eps_w[s]
                # epsilon shock
                eps = par.eps[s]
                # next period assets
                w_next = par.R*w_c+eps
                # expected value
                EV_next +=weight*interp(w_next)
            V_guess = util(c,par)+par.beta*EV_next
            index = np.argmax(V_guess)
            sol.C[iw] = c[index]
            sol.V[iw] = np.amax(V_guess)
        
        sol.it += 1
        sol.delta = max(abs(sol.V - V0)) 
    
    return sol


def gauss_hermite(n):

    # a. calculations
    i = np.arange(1,n)
    a = np.sqrt(i/2)
    CM = np.diag(a,1) + np.diag(a,-1)
    L,V = np.linalg.eig(CM)
    I = L.argsort()
    V = V[:,I].T

    # b. nodes and weights
    x = L[I]
    w = np.sqrt(math.pi)*V[:,0]**2

    return x,w


def Chebyshev(fhandle,points,m,n):
    
    # This is the Chebyshev Interpolation (Regression algorithm)      
    #  in approximation of a scalar function, f(x):R->R                
    #    The approach follow Judd (1998, Allgortihm 6.2, p. 223)         
#############################################################################
# INPUT ARGUMENTS:
#             fhandle:               The funtion, that should be approximated
#             interval:              The interval for the approximation of f(x).
#             m:                     number of nodes used to construct the approximation. NOTE: m>=n+1
#             n:                     Degree of approximation-polynomial
# 
# OUTPUT ARGUMENTS:
#             f_approx:              The vector of approximated function values
#             f_actual:              The vector of actual function values
#             points:                The vector of points, for which the function is approximated
##################################################################################


    assert (m>=n+1), 'The specified parameters are not acceptable. Make sure m>n'

    a = points[0]
    b = points[-1]
    number = points.size
    f_approx = np.nan + np.zeros((number))  # Initial vector to store the approximated function values
    f_actual = np.nan + np.zeros((number))  # Initial vector to store the actual function values

    for x in range(number):                   # Loop over the x values
        ai = np.nan +np.zeros((n+1))         # Initial vector to store the Chebyshev coefficients
        f_hat = 0                             # Initial value of the approximating function
        for i in range(n+1):                  # Loop over the degree of the approximation polynomial. 
            nom = 0                           # Initial value for step 4
            denom = 0                         # Initial value for step 4
            for k in range(m):                # Loop over the approximation notes
                
                # Step1: Compute the m Chebyshev interpolation notes in [-1,1]    
                zk = -np.cos(((2*(k+1)-1)/(2*m))*np.pi)

                # Step 2: Adjust the nodes to the [a,b] interval
                xk = (zk+1)*((b-a)/2)+a

                # Step 3: Evaluate f at the approximation notes. Loaded from f.m
                yk = fhandle(xk);  

                # Step 4: Compute Chebyshev coefficients. Tn=cos(i*acos(zk)) is loaded from Tn.m
                nom += yk*Tn(zk,i)
                denom += Tn(zk,i)**2
                if k==m-1:
                    ai[i] = nom/denom
            
            f_hat = f_hat+ai[i]*Tn(2*(points[x]-a)/(b-a)-1,i)  # The Chebyshev approximation of f(x)
            f_temp = fhandle(points[x])                       # Actual function value, f(x)

        f_approx[x] = f_hat
        f_actual[x] = f_temp

    return f_approx, f_actual, points

def Tn(x,n):
    return np.cos(n*np.arccos(x))