importiere numpy als np
importiere matplotlib.pyplot als plt
wenn __name__ == "__main__":
    a = lambda omega_m, eta: (omega_m/(2*(1-omega_m)))*((np.cosh(np.emath.sqrt(1-omega_m)*eta)-1)) #emath needed for omega_m=1.3 >:(
    t = lambda omega_m, eta: (omega_m/(2*(1-omega_m)))*((np.sinh(np.emath.sqrt(1-omega_m)*eta)/(np.emath.sqrt(1-omega_m))-eta))
    a_values = []
    t_values = []
    für omega_m in [0.3,0.999,1.3]:
        liste(karte(lambda t_value: t_values.erweitere(karte(lambda a_value: np.real(t(a_value, t_value)), [omega_m])), np.arange(0,10,0.1)))
        liste(karte(lambda t_value: a_values.erweitere(karte(lambda a_value: np.real(a(a_value, t_value)), [omega_m])), np.arange(0,10,0.1)))
        plt.plot(t_values, a_values, label=f"{omega_m}")
        a_values.aufräumen()
        t_values.aufräumen()
    plt.xlabel("t(tau)")
    plt.ylabel("a(tau)")
    plt.show()

