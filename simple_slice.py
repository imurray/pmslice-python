from __future__ import print_function

import numpy as np

def slice_sweep(xx, logdist, widths=1.0, step_out=True, Lp=None):
    """simple axis-aligned implementation of slice sampling for vectors

         xx_next = slice_sample(xx, logdist)
         samples = slice_sample(xx, logdist, N=200, burn=20)

     Inputs:
                xx  D,  initial state (or array with D elements)
           logdist  fn  function: log of unnormalized probability of xx
            widths  D,  or 1x1, step sizes for slice sampling (default 1.0)
          step_out bool set to True (default) if widths sometimes far too small
                Lp  1,  Optional: logdist(xx) if have already evaluated it

     Outputs:
                xx  D,  final state (same shape as at start)
     If Lp was provided as an input, then return tuple with second element:
                Lp  1,  final log-prob, logdist(xx)
    """
    # Iain Murray 2004, 2009, 2010, 2013, 2016
    # Algorithm orginally by Radford Neal, e.g., Annals of Statistic (2003)
    # See also pseudo-code in David MacKay's text book p375

    # startup stuff
    D = xx.size
    widths = np.array(widths)
    if widths.size == 1:
        widths = np.tile(widths, D)
    output_Lp = Lp is not None
    if Lp is None:
        log_Px = logdist(xx)
    else:
        log_Px = Lp
    perm = np.array(range(D))

    # Force xx into vector for ease of use:
    xx_shape = xx.shape
    logdist_vec = lambda x: logdist(np.reshape(x, xx_shape))
    xx = xx.ravel().copy()
    x_l = xx.copy()
    x_r = xx.copy()
    xprime = xx.copy()

    # Random scan through axes
    np.random.shuffle(perm)
    for dd in perm:
        log_uprime = log_Px + np.log(np.random.rand())
        # Create a horizontal interval (x_l, x_r) enclosing xx
        rr = np.random.rand()
        x_l[dd] = xx[dd] - rr*widths[dd]
        x_r[dd] = xx[dd] + (1-rr)*widths[dd]
        if step_out:
            # Typo in early book editions: said compare to u, should be u'
            while logdist_vec(x_l) > log_uprime:
                x_l[dd] = x_l[dd] - widths[dd]
            while logdist_vec(x_r) > log_uprime:
                x_r[dd] = x_r[dd] + widths[dd]

        # Inner loop:
        # Propose xprimes and shrink interval until good one found
        while True:
            xprime[dd] = np.random.rand()*(x_r[dd] - x_l[dd]) + x_l[dd]
            log_Px = logdist_vec(xprime)
            if log_Px > log_uprime:
                break # this is the only way to leave the while loop
            else:
                # Shrink in
                if xprime[dd] > xx[dd]:
                    x_r[dd] = xprime[dd]
                elif xprime[dd] < xx[dd]:
                    x_l[dd] = xprime[dd]
                else:
                    raise Exception('BUG DETECTED: Shrunk to current '
                        + 'position and still not acceptable.')
        xx[dd] = xprime[dd]
        x_l[dd] = xprime[dd]
        x_r[dd] = xprime[dd]

    if output_Lp:
        return xx, log_Px
    else:
        return xx

