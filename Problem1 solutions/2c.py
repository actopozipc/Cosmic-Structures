import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":
    a = lambda omega_m, eta: (omega_m/(2*(1-omega_m)))*((np.cosh(np.emath.sqrt(1-omega_m)*eta)-1)) #emath needed for omega_m=1.3 >:(
    t = lambda omega_m, eta: (omega_m/(2*(1-omega_m)))*((np.sinh(np.emath.sqrt(1-omega_m)*eta)/(np.emath.sqrt(1-omega_m))-eta))
    a_values = []
    t_values = []
    #note how omega_m isnt defined in the next line? It does not matter to Mr. Python, since this function will only be called 
    #in a local scope where a variable with the name omega_m is LOCALLY defined
    # :) :) :) :) :) :) :) :) completely normal
    #anyway, I looooove functional programming! 
    #adds the real values of a function f(omega_m, t) for t=np.arange(0,100,0.1) to a list l
    append_to_list = lambda l,f: list(map(lambda t_value: l.extend(map(lambda a_value: np.real(f(a_value, t_value)), [omega_m])), np.arange(0,50,0.1))) 
    #Given two lists l1 and l2, it finds l2[i] if l1[i] is almost zero
    big_crunch = lambda l1,l2: [l2[i] for i in range(len(l1)) if l1[i]<0.001]
    for omega_m in [0.3,0.999,1.3]:
        append_to_list(a_values, a);
        append_to_list(t_values,t);
        plt.plot(t_values, a_values, label=f"{omega_m}")
        print(big_crunch(a_values,t_values))
        a_values.clear()
        t_values.clear()
    plt.xlim(0,50)
    plt.ylim(0,50)
    plt.xlabel("t(tau)")
    plt.ylabel("a(tau)")
    plt.legend()
    plt.show()