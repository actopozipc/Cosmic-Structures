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

def nbody_and_zeldovich(max_a, N=1000, start_a=0.1):
    Q = np.linspace(0, 1, num=N, endpoint=False)

    a0 = start_a
    X = Q - a0*vgradPhi(Q) # the initial conditions for X
    V = -vgradPhi(Q)*a0**(3/2)# the initial conditions for V
    da = 0.0001 # choose some appropriate time step
    a = a0 # set a to a0 where ICs are specified
    amax = max_a # the end time of the simulation
    istep = 0 # step counter

    Xz = Q - amax * vgradPhi(Q)
    Vz = -vgradPhi(Q) * amax

    while a<amax:
        anext = np.minimum(a+da,amax)
        (X,V,a) = step( X, V, a, da, anext )
        istep += 1
    return (X,V, amax, Xz, Vz)
def nbody_and_zeldovich_with_timesteps(timesteps, N=1000, start_a=0.1):
    Q = np.linspace(0, 1, num=N, endpoint=False)
    a0 = start_a
    X = Q - a0*vgradPhi(Q) # the initial conditions for X
    V = -vgradPhi(Q)*a0**(3/2)# the initial conditions for V
    da = 0.0001 # choose some appropriate time step
    a = a0 # set a to a0 where ICs are specified
    istep = 0 # step counter
    amax = 0.5
    Xz = Q - amax * vgradPhi(Q)
    Vz = -vgradPhi(Q) * amax

    while istep<timesteps:
        anext = np.minimum(a+da,amax)
        (X,V,a) = step( X, V, a, da, anext )
        istep += 1
    return (X,V, amax, Xz, Vz)
def L_infinite_norm(x_zeldovich, x_nbody):
    return np.max(np.abs([x_zeldovich[i] - x_nbody[i] for i in range(len(x_nbody))]))
if __name__ == "__main__":
    differences = []
    times = []
    for t in [4,8,16,32,64,128]:
        X,V,amax,Xz,Vz = nbody_and_zeldovich_with_timesteps(t)
        diff = L_infinite_norm(X,Xz)
        differences.append(diff)
        times.append(t)
    plt.loglog(differences,times)
    plt.xlabel("L_norm")
    plt.ylabel("timesteps")
    plt.show()
