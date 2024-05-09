import numpy as np
def powerspectrum_from_the_book(alpha):

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
    return f, fk, kmod, kv

def gradient():
    f, fk, kmod, kv = powerspectrum_from_the_book(3.0);
    # create an empty N*N*N*3 field (4D!) to hold the gradient field
    fD1 = np.zeros([N,N,N,3],dtype=complex)
    # loop over dimensions
    for i in range(3):
        # store component i in gradient field
        fD1[...,i] = fk * 1j * kv[i] / kmod
        fD1[0,0,0,i] = 0.0
    # inverse transform along axes 0,1,2 (but not 3) of the 4d array
    D1 = np.real(np.fft.ifftn(fD1,axes=[0,1,2]))
    return D1

'''
Maybe I am stupid D: I can see how it is straightforward to change the gradient code to get the hessian, but isnt there a way to use the
gradient in order to get the hessian directly?
H_ij = k_i/gradient * k_j/gradient I believe, but this is stupid with the given fD1 because of the dimension
'''
def hessian():
    f, fk, kmod, kv = powerspectrum_from_the_book(3.0);
    # create an empty N*N*N*3 field (4D!) to hold the gradient field
    fH1 = np.zeros([N,N,N,3,3],dtype=complex)
    # loop over dimensions
    for i in range(3):
        for j in range(3):
            # store component i in gradient field
            fH1[...,i,j] = fk  * kv[i] * kv[j]/ kmod**2 #where does the 1j go?
            fH1[0,0,0,i,j] = 0.0
    # inverse transform along axes 0,1,2 (but not 3) of the 4d array
    H1 = np.real(np.fft.ifftn(fH1,axes=[0,1,2]))
    return H1
if __name__ == "__main__":
    N = 128
    print(hessian())