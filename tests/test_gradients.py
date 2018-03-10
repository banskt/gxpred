
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import numpy as np
import time
from utils import model
from inference import log_marginal_likelihood
from inference import logmarglik
from inference import zstates as zs
from utils import hyperparameters


nsample = 300
nsnps = 100
pi = 0.1
mu = 0.0
sigmabg = 0.01
sigma = 0.3
tau = 1 / (0.005 * 0.005)
prior = "gxpred-bslmm"

x, y, csnps, v = model.simulate(nsample, nsnps,
                                pi = pi,
                                mu = mu,
                                sigma = sigma,
                                sigmabg = sigmabg,
                                tau = tau)

nvar = x.shape[0]
nsample = x.shape[1]
params = np.array([pi, mu, sigma, sigmabg, tau])
scaledparams = hyperparameters.scale(params)
cmax = 1

zstates = zs.create(scaledparams, x, y, cmax, nvar, 0.98, prior)

#params = np.array([0.01, 0.1, 0.0003, 1.0, 1 / 0.5 / 0.5])
params = np.array([0.00495222, 0, 0.48975, 0.6, 1 / 0.0001 / 0.0001])

scaledparams = hyperparameters.scale(params)
print(scaledparams)




HPs = []
HPs.append([None, None, None, None, None])
HPs.append([None, None, None,"L1", None])
HPs.append([None, None, "InvG",None, None])
HPs.append([None, None, None,"S2", None])
HPs.append([None, None, "S2","S2", None])
HPs.append([None, None, "InvG","InvG", None])
HPs.append([None, None, "L1","L1", None])


HyPs = []
HyPs.append({"lambda":0.01, "alpha":0.1, "Galpha":0.5, "Gbeta":0.5})
HyPs.append({"lambda":0.05, "alpha":0.01, "Galpha":2, "Gbeta":0.5})
HyPs.append({"lambda":0.1, "alpha":0.001, "Galpha":3, "Gbeta":0.5})


for hyperpriors in HPs:
	for hyperparams in HyPs:
		success, m, der = logmarglik.func_grad(scaledparams, x, y, zstates, prior, hyperpriors, hyperparams)
		print(hyperpriors, hyperparams)
		# Derivative of pi
		delta = 0.00001
		newparams = hyperparameters.scale(params)
		newparams[0] += delta
		newm = logmarglik.func(newparams, x, y, zstates, prior, hyperpriors, hyperparams)
		pi_grad = (newm - m) / delta
		# print ("Derivative of pi by brute force is {:f}".format(pi_grad))
		# print ("Derivative of pi from equation is {:f}".format(der[0]))

		def test_pi_grad():
			assert pi_grad - der[0] < 0.1

		# Derivative of mu
		delta = 0.1
		newparams = hyperparameters.scale(params)
		newparams[1] += delta
		newm = logmarglik.func(newparams, x, y, zstates, prior, hyperpriors, hyperparams)
		mu_grad = (newm - m) / delta
		# print ("Derivative of mu by brute force is {:f}".format(mu_grad))
		# print ("Derivative of mu from equation is {:f}".format(der[1]))

		def test_mu_grad():
			assert mu_grad - der[1] < 0.1

		# Derivative of sigma
		delta = 0.0001
		newparams = hyperparameters.scale(params)
		newparams[2] += delta
		newm = logmarglik.func(newparams, x, y, zstates, prior, hyperpriors, hyperparams)
		sigma_grad = (newm - m) / delta
		# print ("Derivative of sigma by brute force is {:f}".format(sigma_grad))
		# print ("Derivative of sigma from equation is {:f}".format(der[2]))

		def test_sigma_grad():
			assert sigma_grad - der[2] < 0.1

		# Derivative of sigbg
		delta = 0.0001
		newparams = hyperparameters.scale(params)
		newparams[3] += delta
		newm = logmarglik.func(newparams, x, y, zstates, prior, hyperpriors, hyperparams)
		sigbg_grad = (newm - m) / delta
		# print ("Derivative of sigbg by brute force is {:f}".format(sigbg_grad))
		# print ("Derivative of sigbg from equation is {:f}".format(der[3]))

		def test_sigmabg_grad():
			assert sigbg_grad - der[3] < 0.1

		# Derivative of tau
		delta = 0.0001
		newparams = hyperparameters.scale(params)
		newparams[4] += delta
		newm = logmarglik.func(newparams, x, y, zstates, prior, hyperpriors, hyperparams)
		tau_grad = (newm - m) / delta
		# print ("Derivative of tau by brute force is {:f}".format(tau_grad))
		# print ("Derivative of tau from equation is {:f}".format(der[4]))

		def test_tau_grad():
			assert tau_grad - der[4] < 1