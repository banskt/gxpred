#!/usr/bin/env python

import numpy as np
import os
import ctypes
from inference import logmarglik as lml

def prune(newz, prob, oldprobsum, target):
    
    ''' Probability is for the new newz. 
        sum(prob) = 1 - oldprobsum    
    '''
    
    znorm = len(newz[0])
    # print ("Pruning z-states of znorm {:d}".format(znorm))
    sort = np.argsort(prob)[::-1]             # index of decreasing order of prob. prob[sort] should be the sorted array
    cum  = oldprobsum + np.cumsum(prob[sort]) # cumulative sum, ensured that it will reach 1
    nsel = np.where(cum > target)[0]          # find where cum > target

    # assure there is at least one zstate 
    # sometimes posterior values could be so low that they can round off to zero
    # and there would be no state with cum > targ
    if len(nsel) == 0:
        leadk = []
    else:
        zlen = nsel[0] + 1                # when cum > targ for first time (add +1 since indexing starts from 0 / if nsel[0] = 4, then there are 5 states with higher probabilities)
        sel  = sort[:zlen]                # These are our new selection
        sel  = np.sort(sel)               # How about sorting them?
        zlen = len(sel)

        # For debug
        # print(cum)
        # print(prob[sort][:10] / probsum)

        # These are our leading states from which terms with znorm (k+1) will be created
        leadk = [newz[sel[i]] for i in range(zlen)]
        
    return leadk

def create(scaledparams, x, y, features, dist_feature, cmax, nvar, target):

    _path = os.path.dirname(__file__)
    lib = np.ctypeslib.load_library('../lib/zstates.so', _path)
    zcreate = lib.create_zstates
    zcreate.restype  = ctypes.c_int
    zcreate.argtypes = [ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        np.ctypeslib.ndpointer(ctypes.c_int, flags='C_CONTIGUOUS, ALIGNED'),
                        np.ctypeslib.ndpointer(ctypes.c_int, flags='C_CONTIGUOUS, ALIGNED')]

    # Initialize for ||z|| = 0 and 1
    zstates = [[]]
    oldk = zstates
    newk = [[i] for i in range(nvar)]

    # Calculate the posterior probability of the zstates
    posterior   = lml.prob_comps(scaledparams, x, y, features, dist_feature, zstates + newk)
    prob        = np.array(posterior[-len(newk):])
    old_prob    = np.array(posterior[:len(oldk)])
    probsum     = np.sum(prob)
    old_probsum = np.sum(old_prob)

    # Add the ones required
    selk = prune(newk, prob, old_probsum, target)
    if len(selk) > 0:
        zstates += selk

    #print("zstates.py: Working with "+str(len(zstates))+" leading zstates.")

    oldk = selk

    # Iterate over ||z|| = 2 to cmax
    norm = 1
    while norm < cmax:

        # Stop iteration if sum(new posterior) is < 0.02 times sum(old posterior)
        if probsum < (1 - target) * old_probsum:
            break

        norm += 1
        print("Norm is "+str(norm)+" while cmax is "+str(cmax))
        nsel = len(selk)

        # assure there is at least one zstate 
        # sometimes posterior values could be so low that they can round off to zero
        # and there would be no state with cum > targ
        if nsel == 0:
            break;
        else:
            leadk = np.array(selk, dtype=np.int32).reshape(nsel * (norm-1))

            # for first lead create all possible combinations
            # from next lead onwards do not include duplicate combinations.
            #    Note that a duplicate (k+1) entry is possible iff 
            #    two k-mers have at least (k-1) similar elements.
            #    e.g. [2,4,5,8] can be obtained from [2,4,5], [2,4,8], [2,5,8] or [4,5,8]
            # check previous leads to see if any of them has (k-1) elements similar to the current one
            # If so discard the duplicate.
            #
            # ^^^ the above logic has now been moved to a C++ function for speed up. 
            # get the new zstates from a C++ function
            maxnewsize = nsel * (nvar - norm + 1) * norm
            newz       = np.zeros(maxnewsize, dtype=np.int32)
            newstates  = zcreate(nsel, norm-1, nvar, leadk, newz)
            newsize    = newstates * norm
            result     = np.array(newz[:newsize]).reshape((newstates, norm))
            newk       = [sorted(list(result[i])) for i in range(newstates)]

            posterior   = lml.prob_comps(scaledparams, x, y, features, dist_feature, zstates + newk)
            prob        = np.array(posterior[-len(newk):])
            old_prob    = np.array(posterior[:len(oldk)])
            probsum     = np.sum(prob)
            old_probsum = np.sum(old_prob)

            # Add the ones required
            selk = prune(newk, prob, old_probsum, target)
            if len(selk) > 0:
                zstates += selk
                #print(selk)
                
            oldk = selk

    return zstates
