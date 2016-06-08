# Pseudo-Marginal Slice Sampling

`pmslice.py` is a python module for running
[pseudo-marginal slice sampling](http://homepages.inf.ed.ac.uk/imurray2/pub/16pmss/)
It is documented in the module itself.

Pseudo-Marginal slice sampling takes a Markov chain Monte Carlo (MCMC)
method that evaluates the log of an unnormalized probability function, and
turns it into a method that only needs the log of an unbiased estimator of
the function.

This small Python module makes it easy to drop pseudo-marginal slice
sampling into whatever (Python) MCMC code you already have. In our
demonstration `demo.py` a pseudo-marginal slice-sampling scheme only
needs three extra lines of code compared to its conventional MCMC version.
None of these lines required deriving anything problem-specific.

`simple_slice.py` is a helper used in the demonstration. The only file you
need to drop into your MCMC codebase (and all that setup.py installs) is
the module `pmslice.py`.

Please refer to [the paper](http://homepages.inf.ed.ac.uk/imurray2/pub/16pmss/)
for more details, and links to other implementations.
