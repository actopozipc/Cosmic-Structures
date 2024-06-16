import numpy as np
def calc_acc( X ):
    acc = np.zeros_like(X) # init to zero
    rank = np.arange(len(X)) # create rank indices 0,...,N-1
    Xs = np.fmod(1.0+X,1.0) # periodically wrap X to be inside [0,1)
    Xm = np.mean(Xs) # determine X
    # determine the sorting order:
    sorted_ind = np.argsort(Xs)
    # implement eq. (4.46) with 1 added to rank (i=1,...):
    acc[sorted_ind] = (rank+0.5)/len(X)-(Xs[sorted_ind]-Xm)-0.5
    return acc

def step( X, V, a, da, aend ):
    tbegin = 2-2/np.sqrt(a)
    tend = 2-2/np.sqrt(aend)
    amid = 1/(1-(tbegin+tend)/4)**2
    dt = tend-tbegin
    X = X + 0.5*dt * V
    V = V - dt * 3.0/2.0 * amid * calc_acc( X )
    X = X + 0.5*dt * V
    a = a + da
    return (X,V,a)
def zeldovich_approximation(N,a):
    Q = np.linspace(0,1,N)
    X = Q+a/(2*np.pi)*np.sin(2*np.pi*Q)
    V = 1./(2*np.pi)*np.sin(2*np.pi*Q)
    return (X,V)
def gradient_phi(q, A):
    return A*np.sin(2*np.pi*q)*2*np.pi
def nbody_simulation(a0, amax,N, x_initial=zeldovich_approximation(500,0.5)[0], v_initial=0.5**(3/2)/(2*np.pi)*np.sin(2*np.pi*np.linspace(0,1,500))):

    X =  x_initial# the initial conditions for X
    V = v_initial # the initial conditions for V
    da = 0.01 # choose some appropriate time step
    a = a0 # set a to a0 where ICs are specified
    amax = amax # the end time of the simulation
    istep = 0 # step counter
    while a<amax:
        anext = np.minimum(a+da,amax)
        (X,V,a) = step( X, V, a, da, anext )
        istep += 1
    return (X,V,a)

def zeldovich_solution(q,a,A):
    return q-a*A*np.sin(2*np.pi*q)

import matplotlib.pyplot as plt
if __name__=="__main__":
    number = 500
    a0 = 0.5
    (X,V,a) = nbody_simulation(a0, 0.5,number)
    plt.plot(X,V)
    Q = np.linspace(0,1,number)
    plt.plot(zeldovich_solution(Q,a0,-1/(4*np.pi**2)), -gradient_phi(Q,-1/(4*np.pi**2)))
    plt.show()