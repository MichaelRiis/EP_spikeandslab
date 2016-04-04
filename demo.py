import numpy as np 
import pylab as plt
import time

import spikeandslab_ep as EP
reload(EP)


# Number of unknowns and number of observations
D, N = 1000, 500

# Hyperparameters
p0 = 0.01
sigma2 = 2.
zeta0, tau0 = 0, 1.

# Generate problem instance from prior
np.random.seed(0)
A = np.random.normal(0, 1, size = (N, D))
s0 = np.random.binomial(1, p0, size = D)
x0 = s0*np.random.normal(zeta0, tau0, size = D)
y = np.dot(A, x0) + np.sqrt(sigma2)*np.random.normal(0, 1, size = N)

# Fit spike and slab model using EP
SS = EP.SpikeandslabEP(verbose = True, learn_sigma2 = True, learn_p0 = True)
SS.fit(A, y)

# Evaluate
NMSE = np.mean((SS.m - x0)**2)/np.mean(x0**2)

# Plot
plt.figure(figsize = (20, 20))
plt.subplot(2,2,1)
plt.plot(x0, color = 'g', label = 'Groundtruth')
plt.plot(SS.m, color = 'r', label = 'Estimate')
plt.title('Estimated solution with NMSE: %4.3f' % NMSE)
plt.grid(True)
plt.legend()

plt.subplot(2,2,2)
plt.plot(SS.LLs)
plt.title('Marginal likelihood approximation')
plt.xlabel('EM Iterations')
plt.grid(True)

plt.subplot(2,2,3)
plt.plot(SS.sigma2s, label = 'Estimate')
plt.axhline(sigma2, color = 'g', linestyle = '--', label = 'Groundtruth')
plt.title('Updates for noise variance')
plt.xlabel('EM Iterations')
plt.grid(True)
plt.ylim((-0.1, 5))
plt.legend()

plt.subplot(2,2,4)
plt.plot(SS.p0s, label = 'Estimate')
plt.axhline(p0, color = 'g', linestyle = '--', label = 'Groundtruth')
plt.title('Updates for prior probability p0')
plt.xlabel('EM Iterations')
plt.grid(True)
plt.ylim((-0.1, 1.1))
plt.legend()

plt.show()