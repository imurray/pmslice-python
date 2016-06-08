### First a demo of conventional MCMC for comparison:
################################################################################

from __future__ import print_function

import numpy as np
from simple_slice import slice_sweep

def log_f(theta):
    """log of unnormalized target probability density function"""
    log_var = theta[0]
    var = np.exp(log_var)
    xx = theta[1:]
    return -0.5*log_var**2 - (0.5/var)*np.dot(xx, xx) - 0.5*xx.size*log_var

print('Running conventional MCMC...')
D = 5 # dimensionality of demo
theta = np.random.randn(D) # initial condition
Lp = log_f(theta)
S = int(1e4)
samples = np.zeros((S, D))
for ss in range(S):
    if not ((ss+1) % 10):
        print('Iteration %d / %d' % (ss+1, S), end='\r')
    theta, Lp = slice_sweep(theta, log_f, Lp=Lp) # theta updated here
    samples[ss,:] = theta
print('Iteration %d / %d' % (ss+1, S))


### Now a pseudo-marginal version:
################################################################################

import pmslice

# In the pseudo-marginal setup we only have an unbiased random estimate of the
# probability. As a demo, here is some function where exp(log_fhat(theta)) is an
# unbiased estimate of exp(log_f(theta)) above. The pmslice module needs the
# estimator to expose its sources of randomness as keyword arguments.
def log_fhat(theta, rand=np.random.rand, randn=np.random.randn):
    K = np.ceil(10*rand())
    return log_f(theta) + np.sum(randn(K)) - 0.5*K

# Then most of the code is the same as before. Three new lines are marked below.
# There are no tuning parameters, and the only problem-specific detail is
# knowing which random number generators (rand and/or randn) need replacing.

print('Running PM-Slice MCMC...')
D = 5 # dimensionality of demo
theta = np.random.randn(D) # initial condition
# NEW: the next two lines set up pseudo-marginal slice sampling:
pm_rand = {'rand':pmslice.RandClass(), 'randn':pmslice.RandnClass()}
log_clamped_fn = pmslice.clamp_rand(log_fhat, pm_rand) # used instead of log_f
Lp = log_clamped_fn(theta)
S = int(1e4)
pm_samples = np.zeros((S, D))
for ss in range(S):
    if not ((ss+1) % 10):
        print('Iteration %d / %d' % (ss+1, S), end='\r')
    theta, Lp = slice_sweep(theta, log_clamped_fn, Lp=Lp) # theta updated here
    Lp = pmslice.update_rand(log_fhat, pm_rand, Lp, theta) # NEW: update pm_rand
    pm_samples[ss,:] = theta
print('Iteration %d / %d' % (ss+1, S))

# To sanity check:
# Both samples[:,0] and pm_samples[:,0] should marginally come from N(0,1).

# You probably noticed that the pseudo-marginal demo is a lot slower than the
# conventional MCMC on! That's mainly because in this toy demo log_fhat is more
# expensive than the true function log_f. In real applications, computing the
# true function is expensive and the whole point of using an estimator is that
# it's cheaper. If update_rand were the bottle-neck it could be run every 10
# iterations instead of after every update.

