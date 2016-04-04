""" 
    The class SpikeandslabEP implements Expectation Propagation (EP) for linear Gaussian model with spike and slab prior, where the likelihood is given by

        p(y|w) = N(y|Aw, sigma2*I),

    and the prior on each w_i is given by

        p(w_i) = (1-p0)*delta(w_i) + p0*normal(w_i|zeta0, tau0).

    The EP algorithm is used to approximate the exact posterior distribution, i.e. p(w|y), using a deterministic approximation Q(w) = Normal(w|m, Sigma). EM updates rules are implemented for sigma2 and p0.

    Michael Riis Andersen (michael.riis@gmail.com)
    04/04/2016
"""

import numpy as np

# Auxiliary functions
def Phi_multivariate(eta, theta):
    """ Computes log partition function for multivariate gaussian with natural parameters eta & theta"""
    L = np.linalg.cholesky(theta)
    e = np.linalg.solve(L, eta)
    logdet = 2*np.sum(np.log(np.diag(L)))
    return 0.5*np.dot(e, e) - 0.5*logdet

Phi_univariate = lambda eta, theta: 0.5*eta**2/theta - 0.5*np.log(theta)
log_npdf = lambda x,m,v: -0.5*np.log(2*np.pi*v) - 0.5*(x-m)**2/v
npdf = lambda x,m,v: 1./np.sqrt(2*np.pi*v)*np.exp(-0.5*(x-m)**2/v)


class SpikeandslabEP(object):
    """ Expectation prorogation for linear Gaussian model with spike and slab priors for regression weights """ 

    def __init__(self, max_itt = 1000, max_em_itt = 100, alpha = 0.8, learn_sigma2 = False, learn_p0 = False, verbose = False):

        # Convergence stuff
        self.tol = 1e-4
        self.max_itt, self.max_em_itt = max_itt, max_em_itt
        self.alpha = alpha

        # Initial hyperparameters
        self.zeta, self.tau = 0., 1.
        self.p0 = 0.25
        self.sigma2 = 1.

        # Learn hyperparameters
        self.learn_sigma2 = learn_sigma2
        self.learn_p0 = learn_p0

        # Misc
        self.verbose = verbose
        self.compute_full_covariance = False

    def compute_spike_and_slab_moments(self, gamma_bar, lambda_bar):
        r = np.log(1-self.p0) + log_npdf(0, gamma_bar/lambda_bar, 1./lambda_bar) - np.log(self.p0) - log_npdf(0, gamma_bar/lambda_bar, 1./lambda_bar + self.tau)
        p_active = 1./(np.exp(r)+1)
        c = lambda_bar + 1./self.tau
        d = gamma_bar/c
        mx = p_active*d
        vx = p_active*(d**2 + 1./c) - mx**2
        return mx, vx
    
    # TODO: re-use computation of global approximation to speed up marginal likelihood computation
    def compute_EP_marginal_likelihood(self, m, sigma, gamma, lambda_, J, h):

        lambda_bar, gamma_bar = 1./sigma - lambda_, m/sigma - gamma

        logZtilde = np.sum(Phi_univariate(gamma_bar + gamma, lambda_bar + lambda_)) - np.sum(Phi_univariate(gamma_bar, lambda_bar))
        logZ = np.sum(np.log((1 - self.p0)*npdf(0, gamma_bar/lambda_bar, 1./lambda_bar) + self.p0*npdf(0, gamma_bar/lambda_bar, 1./lambda_bar + self.tau)))
        
        return 0.5*self.D*np.log(2*np.pi) - 0.5*self.N*np.log(2*np.pi) - 0.5*self.N*np.log(self.sigma2) - 0.5*self.yty/self.sigma2 + Phi_multivariate(h + gamma, J + np.diag(lambda_)) + logZ - logZtilde 

    def update_global(self, gamma, lambda_, J, h, diagonal_only = True):

        if(self.N > self.D):
            P = J + np.diag(lambda_)
            m, Sigma = np.linalg.solve(P, h + gamma), np.linalg.inv(P)

        else: # Use Woodburys identity for D > N
            W = self.X/lambda_ # X*diag(1./lambda_)
            R  = np.linalg.solve(self.sigma2*np.identity(self.N) + np.dot(self.X, W.T), W) # Inverse term in woodburys
            
            # Only compute diagonal of posterior covariance?
            if(diagonal_only):
                m = (h + gamma)/lambda_ - np.dot(W.T, np.dot(R, h + gamma))
                Sigma = 1./lambda_ - np.sum(W*R, 0)
            else:
                WtR = np.dot(W.T, R) 
                m, Sigma = (h + gamma)/lambda_  - np.dot(WtR, h + gamma), np.diag(1./lambda_) - WtR

        return m, Sigma

    def fit(self, X, y):

        # Store data and precompute
        self.X, self.y = X, y
        self.XtX, self.Xty, self.yty  = np.dot(X.T, X), np.dot(X.T, y), np.dot(y, y)

        # Extract shape
        self.N, self.D = X.shape

        # Initialize global approximation Q
        m, sigma = np.zeros(self.D), np.ones(self.D)
        m_old = m
        
        # Initialize site approximations
        gamma, lambda_ = np.zeros(self.D), 1e-6*np.ones(self.D)

        self.sigma2s, self.p0s, self.LLs = [], [], []

        # Outer EM iterations for learning noise variance
        for em_itt in range(self.max_em_itt):

            # Store current values of hyperparameters
            self.sigma2s.append(self.sigma2)
            self.p0s.append(self.p0)

            # Pre-compute
            J, h = self.XtX/self.sigma2, self.Xty/self.sigma2
                
            # EP iterations
            for itt in range(self.max_itt):     

                # Compute cavity
                gamma_bar, lambda_bar = m/sigma - gamma, 1./sigma- lambda_
                
                # Compute matching moments
                mx, vx = self.compute_spike_and_slab_moments(gamma_bar, lambda_bar)
            
                # Compute update
                new_lambda = (1-self.alpha)*lambda_ + self.alpha*(1./vx - lambda_bar)
                mask = new_lambda > 0

                # Only update variables with positive site precision
                gamma[mask], lambda_[mask] = (1-self.alpha)*gamma[mask]  + self.alpha*(mx[mask]/vx[mask] - gamma_bar[mask]), new_lambda[mask]

                # Update global approximation
                m, sigma = self.update_global(gamma, lambda_, J, h)
                              
                # Check for EP convergence  
                ep_diff = np.linalg.norm(m-m_old)/np.linalg.norm(m)
                if ep_diff < self.tol:
                    break

                # Store current m for convergence test
                m_old = m

            # Compute marginal likelihood approximation
            L = self.compute_EP_marginal_likelihood(m, sigma, gamma, lambda_, J, h)
            self.LLs.append(L)

            # Optimize w.r.t. noise variance if desired
            sigma2_old = self.sigma2
            if(self.learn_sigma2):
                self.sigma2 = (1-self.alpha)*self.sigma2 + self.alpha*np.mean((y - np.dot(X, m))**2  + np.dot(X**2, sigma))  # Updates from Vila et al: 

            if(self.learn_p0):
                # Compute cavity
                gamma_bar, lambda_bar = m/sigma - gamma, 1./sigma - lambda_

                # Compute pseudo posterior support probabilities
                r = np.log(1-self.p0) + log_npdf(0, gamma_bar/lambda_bar, 1./lambda_bar) - np.log(self.p0) - log_npdf(0, gamma_bar/lambda_bar, 1./lambda_bar + self.tau)
                p_active = 1./(np.exp(r)+1)
                self.p0 = (1-self.alpha)*self.p0 + self.alpha*np.mean(p_active)

            # Check for EM convergence
            sigma2_diff = np.abs(self.sigma2 - sigma2_old) 

            if(self.verbose):
                print 'EM itt %d: %4.3f with diff %6.5f/%6.5f and sigma2 = %4.3f and p0 = %6.5f (EP itt %d)' % (em_itt, L, sigma2_diff, ep_diff, self.sigma2, self.p0, itt)

            if(ep_diff < self.tol and sigma2_diff < self.tol):
                break      
        
        if(self.verbose):
            print('Done in %d EM itt' % em_itt)


        # Compute full covariance?
        if(self.compute_full_covariance):
            m, sigma = self.update_global(gamma, lambda_, J, h, diagonal_only = False)
        else:
            Sigma = np.diag(sigma)  

        # Store
        self.m, self.Sigma, self.L = m, Sigma, L
        self.gamma, self.lambda_ = gamma, lambda_

         # Return
        return m, Sigma, L