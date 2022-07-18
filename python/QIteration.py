"""
By Linda van der Spaa, 2022
l.f.vanderspaa@tudelft.nl
"""

import numpy as np
from time import time

def qIteration(T, R, discount, epsilon = 1e-3, maxiter = 1000, verbose = False):
  nS = T.shape[0]
  nA = T.shape[1]

  t0 = time()

  Q = np.zeros((nS,nA))
  diffQ = np.inf
  i = 0
  while diffQ > epsilon and i < maxiter:
    diffQ = 0
    V = np.max(Q,1)
    for s in range(nS):
      for a in range(nA):
        oldQ = Q[s,a]
        r = R if len(R.shape)==1 else R[s,a,:]
        Q[s,a] = T[s,a,:].dot(r + discount*V)
        diffQ += abs(Q[s,a] - oldQ)

    if verbose:
      print("Iteration {0}, absolute Q difference: {1}".format(i, diffQ))

    i+=1

  if verbose:
    dt = time() - t0
    print("Computed Q table in {}s".format(dt))

  if diffQ > epsilon:
    print("Warning: Q-Iteration did not converge!")

  return Q