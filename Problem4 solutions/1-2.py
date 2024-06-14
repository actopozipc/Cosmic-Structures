def powerspectrum_from_the_book(alpha):
    if not isinstance(alpha, float):
        raise Exception("alpha is not float")
    L = 1.0 # the 'physical' box size, sets units of the fundamental mode
    kmin = 2.0*np.pi/L # the fundamental mode of the box
    kmax = kmin * N/2 # the Nyquist mode of each linear dimension
    k1d = np.fft.fftfreq(N,d=1.0/kmin/N) # get FFT mode vector
    kv = np.meshgrid(k1d,k1d,k1d) # get k vector, k = (kx,ky,kz)
    fk = np.random.normal(size=(N,N,N)) + 1j * np.random.normal(size=(N,N,N))
    norm = 1.0/(2*np.pi)**(-1.5)
    kmod = np.sqrt(kv[0]**2 + kv[1]**2 + kv[2]**2) # modulus of k = (kx,ky,kz)
    fk = fk * kmod ** -(alpha/2) # multiply with amplitude
    fk[0,0,0] = 0.0 # zero f(k=0), to enforce zero mean
    f = np.fft.ifftn(fk) # inverse transform
    f = norm * np.real( f ) # take real part and normalise
    return f, fk, kmod, kv

def gradient(a):
    f, fk, kmod, kv = powerspectrum_from_the_book(a);
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
def hessian(alpha):
    f, fk, kmod, kv = powerspectrum_from_the_book(alpha);
    # create an empty N*N*N*3 field (4D!) to hold the gradient field
    fH1 = np.zeros([N,N,N,3,3],dtype=complex)
    # loop over dimensions
    for i in range(3):
        for j in range(3):
            # store component i in gradient field
            fH1[...,i,j] = fk  * kv[i] * kv[j]/ kmod**2 #where does the 1j go?
            fH1[0,0,0,i,j] = 0.0
    H1 = np.real(np.fft.ifftn(fH1,axes=[0,1,2]))
    return H1
def signature(alpha):
    if not isinstance(alpha, float):
        raise Exception("alpha is not float")
    #get hessian for a powerspectrum
    H = hessian(alpha) #dimension N,N,N,3,3
    eigen = np.sort(np.linalg.eigvals(H), axis=3) #axis=3 keeps the dimension smooth, thank you stackoverflow
    #signature has to have dimension N,N,N,1
    signature = np.zeros((N,N,N))
    #python loops are slow as fuck, is there no faster way? I am so bad 
    for dim1 in range(len(eigen)):
        for dim2 in range(len(eigen[dim1])):
            for dim3 in range(len(eigen[dim2])):
                #whoever came up with this insane notation of array[boolean expression]....there is a special place in hell
                signature[dim1,dim2,dim3] = len(eigen[dim1,dim2,dim3][eigen[dim1,dim2,dim3]>0.0]) #number of eigenvalues >0.0 for a point x
    return signature,eigen
def lagrange_map(D):
    test = np.linspace(0,1,N)
    q = np.meshgrid(test,test,test)
    print(q)
    g = gradient(alpha)
    positions = np.zeros([N,N,N,3])
    for i in range(3):
        positions[...,i] = q[i] + D*g[...,i]
    return positions
import matplotlib.pyplot as plt
import numpy as np
def delta_d(D, eigen):
    return 1/((1+D*eigen[...,0]) + (1+D*eigen[...,1]) + (1+D*eigen[...,2])) -1
def plot_particles():
    a = [3.0,5.0,7.0]
    N = 32
    D = 10
    fig =plt.figure(figsize=(10,10),constrained_layout=True)
    array_pos = np.arange(1,10).reshape((3,3))
    for i in range(3):
        alpha = a[i]
        sig, eigen = signature(alpha)
        delta = delta_d(D,eigen)
        deltas = [0,3e3, 9e3, 3e6, 6e6, 3e9, 6e9, 3e12, 6e12, 3e15]
        for j in range(3):
            positions = lagrange_map(deltas[array_pos[i,j]])
            ax = fig.add_subplot(3,3,array_pos[i,j],projection='3d')
            ax.set_xlabel(f"alpha={alpha}")
            ax.set_ylabel(f"D={np.average(delta.flatten())}")
            ax.scatter(positions[...,0], positions[...,1], positions[...,2],c=delta)
    plt.show()
if __name__ == "__main__":
    #plot_particles()
    alpha = 3.0
    N = 64
    L = 1.0
    D = [10,3e3, 4e5]
    f, fk, kmod, kv = powerspectrum_from_the_book(alpha); #I love global variables, such a good coding practice (not)
    sig, eigen = signature(alpha)
    fig, axs = plt.subplots(1,3)
    for d in range(len(D)):
        delta_average = delta_d(D[d],eigen)
        axs[d].hist(delta_average.flatten())
    plt.show()
    plt.figure(2)
    plt.plot()
    