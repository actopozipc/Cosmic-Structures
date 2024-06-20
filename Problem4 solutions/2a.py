import numpy as np
import matplotlib.pyplot as plt
def calc_acc( X ):
    acc = np.zeros_like(X) # init to zero
    ank = np.arange(len(X)) # create rank indices 0,...,N-1
    Xs = np.fmod(1.0+X,1.0) # periodically wrap X to be inside [0,1)
    Xm = np.mean(Xs) # determine X
    # determine the sorting order:
    sorted_ind = np.argsort(Xs)
    rank = np.empty_like(sorted_ind)
    rank[sorted_ind] = np.arange(len(X))
    # implement eq. (4.46) with 1 added to rank (i=1,...):
    acc[sorted_ind] = (rank[sorted_ind]+0.5)/len(X)-(Xs[sorted_ind]-Xm)-0.5
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

A = 1/4/np.pi**2

    
def vgradPhi(x):
    return -A*2*np.pi*np.sin(2*np.pi*x)

def step_con( X, V, a, da, aend ):
    X = X + 0.5*da * V
    V = (a/(aend))**(3/2) * V - calc_acc(X)/a * (1-(a/aend)**(3/2))
    X = X + 0.5*da * V
    a = a + da
    return (X,V,a)

def nbody_and_zeldovich(max_a, N=1000):
    Q = np.linspace(0,1,N)
    a0 = 0.1
    X = Q - a0*vgradPhi(Q) # the initial conditions for X
    V = -vgradPhi(Q)#*a0**(3/2)# the initial conditions for V
    da = 0.001 # choose some appropriate time step
    a = a0 # set a to a0 where ICs are specified
    amax = max_a # the end time of the simulation
    istep = 0 # step counter

    Xz = Q - amax * vgradPhi(Q)
    Vz = -vgradPhi(Q) * amax

    while a<amax:
        anext = np.minimum(a+da,amax)
        (X,V,a) = step_con( X, V, a, da, anext )
        istep += 1
    return (X,V, amax, Xz, Vz)
if __name__ == "__main__":
    fig, axs = plt.subplots(2, 2)
    a_amxs = np.linspace(0.1,2,4).reshape(2,2)
    for i in range(2):
        for j in range(2):
            X,V,amax,Xz,Vz = nbody_and_zeldovich(a_amxs[i,j])
            axs[i,j].plot(X,V*amax)
            axs[i,j].plot(Xz,Vz,':')
            axs[i,j].set_xlabel(f"X for a={a_amxs[i,j]}")
            axs[i,j].set_ylabel(f"P for a={a_amxs[i,j]}")
    plt.show()
