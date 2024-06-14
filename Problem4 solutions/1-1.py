import numpy as np
import matplotlib.pyplot as plt
pi = np.pi
def phi(q, A):
    return A*np.cos(2*pi*q)
def gradient_phi(q, A):
    return A*np.sin(2*pi*q)*2*pi
def x(q,D,A):
    return q+D*gradient_phi(q,A) #plus instead of minus because minus is already in the gradient method
def delta(q,D,A):
    return 1/(1-D*A*4*pi**2 * np.cos(2*pi*q)) -1
def x_planewave(q,D):
    return q + (D* np.sin(2*pi*q))/(2*pi)
def v_planewave(q,D):
    return (np.sin(2*pi*q))/(2*pi)
def delta_planewave(q,D):
    pass
if __name__=="__main__":
    a = 10
    q = np.linspace(0,1,100)
    D = 0
    
    X = x(q,D,A=a)
    del_d = -gradient_phi(q,a)
    fig, axs = plt.subplots(1, 3)
    plt.figure(1)
    for i in range(3):
        
        X = x(q,D,A=a)
        D = D + 1/(4*a*np.pi**2 )
        del_d = -gradient_phi(q,a)
        axs[i].plot(X, del_d)
        axs[i].set_xlabel("X")
        axs[i].set_ylabel("del_D X")
    plt.show()
    plt.figure(2)
    D=0
    fig, axs = plt.subplots(1, 3)
    for i in range(3):
        
        X = x_planewave(q,D)
        d_x = v_planewave(q,D)
        D = D + 1/(4*a*np.pi**2 )
        axs[i].plot(X, d_x)
        axs[i].set_xlabel("X")
        axs[i].set_ylabel("V")
    plt.show()