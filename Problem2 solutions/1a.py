import numpy as np
import matplotlib.pyplot as plt
def powerspectrum_from_the_book(alpha,zeile,spalte):
    N = 128
    L = 1.0 # the 'physical' box size, sets units of the fundamental mode
    kmin = 2.0*np.pi/L # the fundamental mode of the box
    kmax = kmin * N/2 # the Nyquist mode of each linear dimension
    k1d = np.fft.fftfreq(N,d=1.0/kmin/N) # get FFT mode vector
    kv = np.meshgrid(k1d,k1d,k1d) # get k vector, k = (kx,ky,kz)
    norm = 1.0/(2*np.pi)**(-1.5)
    kmod = np.sqrt(kv[0]**2 + kv[1]**2 + kv[2]**2) # modulus of k = (kx,ky,kz)
    fk = np.random.normal(size=(N,N,N)) + 1j * np.random.normal(size=(N,N,N))
    fk = fk * kmod ** -(alpha/2) # multiply with amplitude
    fk[0,0,0] = 0.0 # zero f(k=0), to enforce zero mean
    f = np.fft.ifftn(fk) # inverse transform
    f = norm * np.real( f ) # take real part and normalise
    ax[zeile,spalte].imshow(f[:,:,32],extent=[0,L,0,L])

if __name__ == "__main__":
    fig, ax = plt.subplots(2,3)
    for alpha in np.arange(6):
        #row and column to plot on for 2,3
        zeile = (alpha -1) // 3
        spalte = (alpha -1) % 3
        powerspectrum_from_the_book((alpha)*1.0,zeile,spalte)
        ax[zeile,spalte].set_title(f"alpha={alpha}")
    plt.show()