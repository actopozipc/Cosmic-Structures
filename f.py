import numpy as np
uv = lambda A,n,alpha,R: A/(2*np.pi**2*(3+alpha+2*n)*R**(3+alpha+2*n))
def cholesky(nsamples):
    sigma2 = uv(1,2,1,1)
    sigma2 = 1
    sigma_t = np.sqrt(sigma2**2/15)
    # Upper triangular matrix from Cholesky decomposition
    # of covariance matrix Σs
    Ls = sigma2/np.sqrt(5) * np.array([[1, 0, 0], \
    [1/3, np.sqrt(8)/3, 0], \
    [1/3, 1/np.sqrt(18), np.sqrt(5/6)]])

    # draw t=(H12,H13,H23) from uncorrelated Gaussian
    t = np.random.normal(0,sigma_t,size=(nsamples,3))

    # draw s=(H11,H22,H33) from correlated Gaussian using Cholesky sampling
    s = np.random.normal(0,1,size=(nsamples,3)) @ Ls.T

    # assemble Hessian matrix from components
    H = np.zeros((nsamples,3,3))
    H[:,0,0] = s[:,0]
    H[:,1,1] = s[:,1]
    H[:,2,2] = s[:,2]
    H[:,0,1] = t[:,0]; H[:,1,0] = t[:,0]
    H[:,0,2] = t[:,1]; H[:,2,0] = t[:,1]
    H[:,2,1] = t[:,2]; H[:,1,2] = t[:,2]
    # get absolute value of the determinant
    h = np.abs(H)
    sigma1 = uv(1,1,1,1)
    sigma1 = 1
    # compute the expectation value
    exp_ncrit = np.sqrt(27)/((2*np.pi)**1.5*sigma1**3) * np.mean( h )
    Cs_empirical = np.cov( s, rowvar=False )

    sigma_s = Ls @ Ls.T
    print("Ls @ Ls.T:")
    print(sigma_s)
    print("np.cov:")
    print(Cs_empirical)

cholesky(300000)