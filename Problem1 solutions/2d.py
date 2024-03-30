import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
'''
Calculates H^2 for given parameters

'''
def Hubble_a(a):
    #This might be a little bit controversial in numerics, but I have a software dev background:
    #normally, companies hire own people to write unit tests
    #In science/physics, one mostly writes unit tests for ones own code (or at least thats how I was taught at prof Janis lecture)
    #Writing my own unit tests provides a huge danger: Missing tests for the edge cases I forgot to think of when I wrote the function
    #Since python only has type hinting and recommends duck typing, I want to type check everything before returning in normal functions 
    # -> seems to be good versus errors with units or other edge cases that should be normally covered by unit tests
    #(I cant do that in anonymous functions, but on the other side, are anonymous functions even a good code pattern?)
    if not isinstance(a, float):
        raise Exception("a is not float")
    #Also some practice to miss out on unit testing: Always declare variables only at local scopes
    #(This might be a good practice anyway)
    omega_r = 8*10**-5;
    omega_m=0.3
    omega_lambda=0.7
    omega_k = 1-omega_r-omega_m-omega_lambda;
    H_squared = np.sqrt((omega_r*a**-4 + omega_m*a**-3 + omega_lambda))
    return H_squared;

def t_of_a(a):
    res = np.zeros_like(a)
    for i,ai in enumerate(a):
        t,err = quad(lambda ap : 1.0/(ap*Hubble_a(ap)),0,ai)

        res[i] = t
    return res
'''
Sets labels and titel for a given axis

'''
def set_plot_labels(axs_index1, axs_index2, x_label="t", y_label="a(t)", title=""):
    if not isinstance(axs_index1, int): #I should maybe type check the other arguments too but ig I wont use this ever again
        raise Exception("index is not int")
    ax = axs[axs_index1, axs_index2]
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()

if __name__ == "__main__":
    fig, axs = plt.subplots(2,2)
    #Numerical solution of the equation
    H0 = 1/(13.97*10**9)
    a = np.logspace(-8,1,100)
    set_plot_labels(0,0, "t*H0", "a(t)","Numerical solution of the equation" )
    axs[0,0].loglog(t_of_a(a)*H0,a)
    #Asymptotic solutions
    a = np.arange(0,20,0.01)
    #Since every solution contains some sqrt(constant)*t, I will set sqrt(constant)=1
    names = ["radiation only", "mass only", "lambda only", "only curvature"]
    for name_index, f in  enumerate([lambda t: np.sqrt(2*t), lambda t: (3/2 * t)**2/3, lambda t: np.exp(t), lambda t: t]):
        axs[1,0].plot(a,f(a), label=names[name_index])
        axs[0,1].loglog(a,f(a), label=names[name_index])
        axs[1,1].loglog(H0*a,f(a), label=names[name_index])
    set_plot_labels(1,0, title= "Asymptotic solutions")
    set_plot_labels(0,1, title="Asymptotic solutions on loglog")
    set_plot_labels(1, 1, x_label="H0*t", title="Asymptotic solutions on loglog with H0*t")
    axs[1,0].set_xlim(-2,20)
    axs[1,0].set_ylim(0,20)
    axs[0,1].set_xlim(0,2)
    axs[0,1].set_ylim(0,10)
    plt.show()