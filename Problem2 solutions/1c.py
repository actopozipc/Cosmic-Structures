'''
Object oriented programming, forgive me for copy-pasting more than twice! D: 
But I dont know how to import a module that starts with a number and I am too lazy to rename the file 1b.py
'''
import numpy as np
import matplotlib.pyplot as plt
def powerspectrum_from_the_book(alpha):
    if not isinstance(alpha, float):
        raise Exception("alpha is not float")
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


def hessian(alpha):
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

def signature(N, alpha):
    if not isinstance(N, int):
        raise Exception("N is not integer")
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
    return signature

if __name__ == "__main__":
    alpha = 3.0
    N = 64
    L = 1.0
    f, fk, kmod, kv = powerspectrum_from_the_book(alpha); #I love global variables, such a good coding practice (not)
    s = signature(N,alpha)
    print(s[s>3]) #sanity check :) this should never ever contain any values!
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(f[:,:,N-1],extent=[0,L,0,L]) #N throws exception
    axs[1].imshow(s[:,:,N-1],extent=[0,L,0,L])
    plt.show()
