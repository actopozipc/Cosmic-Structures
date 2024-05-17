import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

uv = lambda A ,n , alpha , R : A /(2* np . pi **2*(3+ alpha +2* n ) * R **(3+ alpha +2* n ) )
nsamples = 100000
#sigma0 = 1
#sigma1 = 1
#sigma2 = 1
sigma0 = uv (1 ,0 ,1 ,1)
sigma1 = uv (1 ,1 ,1 ,1)
sigma2 = uv (1 ,2 ,1 ,1)
sigma_t = np.sqrt(sigma2**2/15)

#Covariance matrix for H_ii and f
MEAN = np.zeros(4)
SIGMA = (sigma2**2)/15 * np.array([[3, 1, 1, -5], \
[1, 3, 1, -5], \
[1, 1, 3, -5], \
[-5, -5, -5, 15*(sigma0/sigma2)**2]])
# draw t=(H12,H13,H23) from uncorrelated Gaussian
t = np.random.normal(0,sigma_t,size=(nsamples,3))

# draw s=(H11,H22,H33,f) from correlated Gaussian using Cholesky sampling
s = np.random.multivariate_normal(MEAN, SIGMA, size=(nsamples), check_valid='warn', tol=1e-8)

# assemble Hessian matrix from components
H = np.zeros((nsamples,3,3))
f = np.zeros(nsamples)
H[:,0,0] = s[:,0]
H[:,1,1] = s[:,1]
H[:,2,2] = s[:,2]
f = s[:,3]
H[:,0,1] = t[:,0]; H[:,1,0] = t[:,0]
H[:,0,2] = t[:,1]; H[:,2,0] = t[:,1]
H[:,2,1] = t[:,2]; H[:,1,2] = t[:,2]

# get absolute value of the determinant and its eigenvalues
h = np.abs(np.linalg.det(H))
eigvals = np.linalg.eigvalsh(H)

#set values to 0 where the hessian isn't negative definite
for i in range(len(h)):
    if len(eigvals[i][eigvals[i]<0.0]) != 3:
        h[i] = 0

#set values to 0 for increasing gamma
y = 50
g = np.zeros(y)
for i,gamma in enumerate(np.linspace(-3,5,num=y)):
    for k in range(len(f)):
        if f[k] < gamma*sigma0:
            h[k] = 0
    g[i] = np.sqrt(27)/((2*np.pi)**1.5*sigma1**3) * np.mean( h )

#plt.figure(figsize=(8, 6), dpi=256)
plt.plot(np.linspace(-3,5,num=y), g)
plt.xlabel(r'$\gamma$')
plt.ylabel('Density of maxima')
plt.show()