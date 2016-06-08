"""
A demo implementation of pseudo-marginal slice-sampling

Pseudo-marginal Markov chain Monte Carlo (MCMC) methods sample from a
distribution given only an unbiased estimator of the target probability
density function. This module allows you to perform pseudo-marginal slice
sampling, with very little modification to the code you would write if you
could evaluate the target probability function exactly. The main work for a
user is to expose the random number generators used by their estimator as
keyword arguments, so they can be replaced.

In the pseudo-marginal MCMC setup, let "fhat" be the unbiased estimator of
a distribution over variables theta. It is assumed that the log of this
estimator can be evaluated with a user-provided function of the form:

    log_fhat(theta, rand=np.random.rand, randn=np.random.randn)

That is, the user writes code using standard random number generators,
rand() and/or randn(), such that exp(log_fhat(...)) is an unbiased estimate
of the probability of theta. All random number generators used are exposed as
keyword arguments so they can be replaced.

Pseudo-marginal slice sampling replaces the random number generators with
objects that form part of the Markov chain. These objects are put in a
dictionary, using the names from the estimator's keyword arguments:

    pm_rand = {'rand':pmslice.RandClass(), 'randn': pmslice.RandnClass()}

The objects can be updated in the Markov chain by calling:

    pmslice.update_rand(log_fhat, pm_rand, Lp=None, theta)

In between these updates, you use your conventional MCMC code to update your
variables of interest. That code should be given a function

    log_fhat_clamped = pmslice.clamp_rand(log_fhat, pm_rand)

in place of a function that could provide the true log of the unnormalized
target distribution. Call pmslice.update_rand again after each update, or
each several updates, of your Markov chain.

See the separate demo.py for a full example.

For a more detailed explanation of the method, see the paper:

    Pseudo-Marginal Slice Sampling,
    Iain Murray and Matthew M. Graham,
    JMLR: W&CP, 51:911-919, 2016.
    http://homepages.inf.ed.ac.uk/imurray2/pub/16pmss/

Being generic code in pure python, this module carries some time overhead.
In some applications, more computations could be cached. When function
evaluations are cheap, python-level book-keeping can dominate and rewriting
in another language would be appropriate. This code was written after the
paper was published. The code that was used to produce the original results
is available separately.


More advanced usage:

The signature of the estimator can be arbitrary:
    log_fhat(*args, randX=..., randY=..., randZ=..., **kwargs)
As long as all random number generators used in the code are exposed as
keyword arguments. If you're using generators other than Uniform[0,1] and
N(0,1), you'll need to either rewrite your code to use these primitives, or
extend this module appropriately.

If you wish to update certain blocks of random number draws separately in
the Markov chain, that is easy to do. Some of the draws could call randX()
and others randY(). Then make the signature of the log-estimator:

    log_fhat(theta, randX=np.random.rand, randY=np.random.rand, ...)

Then pmslice.update_rand can update a dictionary with multiple objects:

    pm_rand = {'randX':pmslice.RandClass(), 'randY':pmslice.RandClass(), ...}

You can use an arbitrary number of generators, and they can be of multiple
types (RandClass/RandnClass or any other pmslice-compatible type you create).

What if you wanted to run Hamiltonian Monte Carlo (HMC), or some other
method that uses gradients, on the main variables theta? That's fine. When
calling pmslice.update_rand, pass a log_fhat function that only returns a
single scalar, a log-unnormalized-probability estimate. Then when calling
pmslice.clamp_rand, pass a different log_fhat function that also returns
gradients, and use the resulting clamped function in HMC.

What if you want to do HMC, or some other MCMC method, jointly on the
random number draws and theta? That's an interesting idea, but outside the
scope of this module.
"""

# Iain Murray, June 2016.
# http://iainmurray.net/


import numpy as _np

def _restart_rands(rand_dict):
    for kk in rand_dict:
        rand_dict[kk].pos = 0

def update_rand(log_fhat, rand_dict, Lp=None, *args, **kwargs):
    """Update the auxiliary objects rand_dict in a pseudo-marginal Markov chain.

    Lp should be the log-unnormalized probability of the joint auxiliary state,
        Lp = clamp_rand(log_fhat, rand_dict)(*args, **kwargs)
    Often just:
        Lp = clamp_rand(log_fhat, rand_dict)(theta)
    Lp has usually been computed in the previous Markov chain update.
    However, you can leave Lp=None, and it will be recomputed for you.

    The log-unnormalized-probability of the final auxiliary state is returned,
    so it can be provided to the function for the next MCMC update.
    """
    if Lp is None:
        _restart_rands(rand_dict)
        Lp = log_fhat(*args, **dict(kwargs, **rand_dict))
    for kk in rand_dict:
        rr = rand_dict[kk]
        Lp_threshold = Lp + _np.log(_np.random.rand())
        while True:
            rr.slice_propose()
            _restart_rands(rand_dict)
            Lp_prop = log_fhat(*args, **dict(kwargs, **rand_dict))
            if Lp_prop >= Lp_threshold:
                break
            rr.slice_shrink()
        rr.accept()
        Lp = Lp_prop
    return Lp

def clamp_rand(log_fhat, rand_dict):
    """Return a deterministic function from a random estimator for MCMC updates

    This function helps run a pseudo-marginal Markov chain on variables theta
    with a log-unbiased-estimator function with signature:

        log_fhat(theta, rand=np.random.rand, randn=np.random.randn)

    or more generally:

        log_fhat(*args, randX=..., randY=..., randZ=..., **kwargs)

    where all of the random number generators used by the estimator have been
    exposed as keyword arguments.

    The pseudo-marginal Markov chain will use a dictionary of special objects
    that replace the random number generators. For the first example:
        
        pm_rand = {'rand':pmslice.RandClass(), 'randn':pmslice.RandnClass()}

    This function then creates a version of the estimator that will use the same
    random number draws between updates to pmslice.update_rand.

        log_fhat_clamped = pmslice.clamp_rand(log_fhat, pm_rand)

    This clamped estimator can be used in any conventional Markov chain code
    (expecting a log-unnormalized-probability) to update the variables of
    interest theta. The clamped function takes the same arguments as log_fhat,
    except the random number generators should not be specified.
    """
    def clamped_log_fhat(*args, **kwargs):
        _restart_rands(rand_dict)
        return log_fhat(*args, **dict(kwargs, **rand_dict))
    return clamped_log_fhat

class RandClass(object):
    """
    Objects of this class are used in pseudo-marginal slice sampling to replace
    the np.random.rand() function in the code for an unbiased estimator. See the
    documentation for the rest of the pmslice module.
    """
    # u_prop is an array of values to be emitted, which is maintained to be the
    # same length as uu, previous values, and nu_prop, Gaussian values used in
    # proposal mechanism.
    _rand = _np.random.rand
    def __init__(self):
        self.u_prop = _np.zeros(0)
        self.pos = 0; # the number emitted so far / position next u_prop
        self.accept()
    def accept(self):
        # Copy current proposal to "old values" uu, and set step to zero, so
        # same values will still be emitted.
        self.uu = self.u_prop[:self.pos]
        self.u_prop = self.uu.copy()
        self.step = 0
        # Set up new search direction and bracket
        self.nu = _np.random.randn(self.pos)
        self.mx = _np.random.rand()
        self.mn = self.mx - 1.0
        self.pos = 0
    def _combine(self, uu, nu):
        """Returns reflected_around_inside_hypercube(uu + nu*self.step)"""
        target = _np.abs(uu + nu*self.step)
        ipart = _np.floor(target)
        fpart = target - ipart
        is_odd = (ipart % 2) > 0.9
        fpart[is_odd] = 1 - fpart[is_odd]
        return fpart
    def slice_shrink(self):
        if self.step > 0:
            self.mx = self.step
        else:
            self.mn = self.step
        # In intended uses, algorithms only collapse to point if there's a bug:
        assert(self.mx != self.mn)
    def slice_propose(self):
        self.step = self.mn + (self.mx - self.mn)*_np.random.rand()
        self.u_prop = self._combine(self.uu, self.nu)
        self.pos = 0
    def __call__(self, *args):
        args = _np.array(args, dtype='int64')
        num_needed = _np.prod(args)
        uu_left = self.uu.size - self.pos
        if num_needed > uu_left:
            # Double resevoir, or at least enough to cater current request:
            num_extend = max(num_needed, self.uu.size)
            new_uu = self._rand(num_extend)
            new_nu = _np.random.randn(num_extend)
            new_u_prop = self._combine(new_uu, new_nu)
            self.uu = _np.hstack((self.uu, new_uu))
            self.nu = _np.hstack((self.nu, new_nu))
            self.u_prop = _np.hstack((self.u_prop, new_u_prop))
        # The copy is to prevent user changing random numbers for future calls
        ans = self.u_prop[self.pos:self.pos+num_needed].reshape(args).copy()
        self.pos += num_needed
        return ans

class RandnClass(RandClass):
    """
    Objects of this class are used in pseudo-marginal slice sampling to replace
    the np.random.randn() function in the code for an unbiased estimator. See
    the documentation for the rest of the pmslice module.
    """
    _rand = _np.random.randn
    def _combine(self, uu, nu):
        # Initial interval in RandClass is of width 1, so here get initial
        # interval of width 2\pi, the whole ellipse of combinations.
        beta = 2*_np.pi*self.step
        return uu*_np.cos(beta) + nu*_np.sin(beta)

