#  -*- coding: utf-8 -*-

"""
09/16/2020
@author: Peter Toth, UNR

Explanatory Monte Carlo simulation for imputation project: probit model with a
single imputed occasionally missing RHS variable (x) using one, two, five other
(non-missing) RHS variables (z-s)

Packages needed: numpy, scipy, multiprocessing, random

NOTES:
- we regenerate the missingness indicators every time
"""

import numpy as np
from scipy import optimize as opt
from scipy import linalg
from scipy.stats import norm as normal
import time
import multiprocessing as mp
import random

#############################
#   SIMULATION PARAMETERS   #
#############################

# True values
alpha = 1
#beta = [0.5, -2]
beta = [0.5, -2, 1.2, -0.7, 0.3, 1]#, -1.2]


# Initial values
a0 = 0
#b0 = [0, 0]
b0 = [0, 0, 0, 0, 0, 0]#, 0, 0]

# Replication number, sample sizes
reps = 5000
nlist = [1000, 2000, 4000]

# File names (fname is req'd, stores the aggregate results, resultsFname
# can be set to False)
fname = '5ZsO124.txt'
resultsFname = '5ZsO124'
cores= mp.cpu_count()-2

# Random seed (not implemented), noisiness (should be True only for dev)
seed = 2433523
noise = True

# Fineness of grid; bwidth is locked in with a 1/3 rate
gridno = 400

# direct calculation of conditional expectation or numerically integrated
direct = False


#################
#   Functions   #
#################


def fromZToX(z):
    """ Helper for 'dgp'. Getting the x-values, given the z RHS variables.
     The length of z should be smaller than 10.
    """
    coeffs = np.array([0.6, -0.4, 0.7, 0.1, -0.3, 0.9, 1, -1, 0.5, -0.2]
                      [:z.shape[1]])
    index = np.sum(z * coeffs, axis=1, keepdims=True) \
        + np.random.normal(size=(z.shape[0], 1))
    x = 4 * np.exp(index) / (1 + np.exp(index)) - 2
    return x


def fromZToM(z):
    """ Helper for 'dgp'. Getting the missing indicators, given the z RHS
    variables. The length of z should be smaller than 10.
    """
    coeffs = np.array([0.7, -0.3, 0.75, 0.16, -0.21, 0.56, 0.76, -1.2, 0.54,
                      -0.42][:z.shape[1]])
    probit_index = np.sum(z * coeffs, axis=1, keepdims=True)\
        + np.random.normal(size=(z.shape[0], 1))
    m = ((probit_index < -0.8) | (probit_index > 0.8)).astype('int')
    return m


def dgp(alpha, beta, n, noise=False):
    """ Creating the full information data set (no missing values) assuming a
    probit model. The relationship between x and (the independent) z, also m
    and z is baked in (a weird nonparametric function described by 'fromZToX'
    and 'fromZToM', respectively).

    arguments:
    - coefficients: 'alpha' (float), 'beta' (array of floats with length k)
    - sample size: n

    returns the list of arrays:
    - y: n-by-1, the outcome variable (binary, dtype is integer)
    - x: n-by-1, the scalar, float RHS variable that has missing values later
    - z: n-by-k, the vector of RHS variables without missing values
    - m: n-by-1, vector of missing indicators (=1 when the obs. is missing)
    """
    z = np.hstack((np.ones((n, 1)),
                   4 * np.random.uniform(size=(n, len(beta)-1)) - 2))
    x = fromZToX(z)
    y = (x * alpha + np.sum(z * beta, axis=1, keepdims=True)
         > np.random.normal(size=(n, 1))) \
        .astype('int')
    m = fromZToM(z)

    if noise:
        print('Missingness rate: ', np.mean(m))
        print('Mean of y:', np.mean(y))
    return y, x, z, m


def fullDataMoments(coeffs, y, x, z, weight=None):
    """ Creates the objective function for the estimator that assumes that the
    full dat set is available without missing values.
    Its arguments are the coefficient values 'coeffs' (k+1 numpy array) and
    the data y, x, z in separate 2D arrays. Returns the value of the GMM
    objective function as a float.
    """
    indices = y-normal.cdf(x * coeffs[0] + np.sum(z * coeffs[1:],
                           axis=1, keepdims=True))
    moments = np.matrix(np.mean(np.hstack((x * indices, z * indices)), axis=0))
    if weight is None:
        weight = np.matrix(np.identity(x.shape[1]+z.shape[1]))
    return (moments * weight * moments.transpose())[0, 0]


def fullDataWeights(coeffs, y, x, z):
    """ Estimated optimal weights for the fully-observed moments as a function
    of (true) coefficients and the data.
    """
    indices = y-normal.cdf(x * coeffs[0] + np.sum(z * coeffs[1:],
                           axis=1, keepdims=True))
    moments = np.matrix(np.hstack((x * indices, z * indices)))
    return linalg.inv((moments.transpose() * moments) / y.shape[0])


def gmmFullData(y, x, z, a0, b0, weighting_iteration=1, noise=False):
    """ The infeasible GMM estimator that is applied for the full data set
    pretending the missing x values are there.
    Arguments:
    - y, x, z: variables from the data as (n-by-1, n-by-1, n-by-k+1) 2D arrays
    - a0, b0: the initial values for maximization (1D arrays) - the dimensions
              must agree with the number of columns in x and z
    - noise: boolean, set it to True if want to print the messages of the
             optimizer (automatically suppressed when iteration() is run with
             the MC decorator)
    """
    a0 = np.array(a0, ndmin=1)
    b0 = np.array(b0)
    coeffs0 = np.concatenate((a0, b0))
    weight = None
    for i in range(weighting_iteration):
        optimum = opt.minimize(
                    fullDataMoments, coeffs0, args=(y, x, z, weight),
                    method='Newton-CG', jac=fullDataJacobian)
        coeffs0 = optimum.x
        weight = fullDataWeights(coeffs0, y, x, z)
    optimum = opt.minimize(
                fullDataMoments, coeffs0, args=(y, x, z, weight),
                method='Newton-CG', jac=fullDataJacobian)
    if noise:
        print(optimum.message)
    return optimum.x


def gmmNonMissingData(y, x, z, m, a0, b0, weighting_iteration=1, noise=False):
    """ The feasible GMM estimator that uses the same moments as the infeasible
    estimator gmmFullData, but is only applied for the part of the data that
    is fully observed (non-missing x values, when m=0).
    Arguments:
    - y, x, z: variables from the data as (n-by-1, n-by-1, n-by-k+1) 2D arrays
    - m: missingness indicator, another 2D array (n-by-1)
    - a0, b0: the initial values for maximization (1D arrays) - the dimensions
              must agree with the number of columns in x and z
    - noise: boolean, set it to True if want to print the messages of the
             optimizer (automatically suppressed when iteration() is run with
             the MC decorator)
    """
    a0 = np.array(a0, ndmin=1)
    b0 = np.array(b0)
    y = y[np.squeeze((m == 0))]
    x = x[np.squeeze((m == 0))]
    z = z[np.squeeze((m == 0))]
    return gmmFullData(y, x, z, a0, b0, weighting_iteration, noise)


def fullDataJacobian(coeffs, y, x, z, weight=None):
    residuals1 =  y-normal.cdf(x * coeffs[0] + np.sum(z * coeffs[1:],
                                                      axis=1, keepdims=True))        
    moments = np.matrix(np.mean(np.hstack((x * residuals1,
                                       z * residuals1)), axis=0))  

    residuals1Der = normal.pdf(x * coeffs[0] + np.sum(z * coeffs[1:],
                                                      axis=1, keepdims=True))
    xTilde =np.hstack((x, z))
    momentsDer = np.matrix(np.einsum('ij,ik -> jk',
                                     xTilde * residuals1Der, xTilde)/len(y))
    if weight is None:
        weight = np.matrix(np.identity(moments.shape[1]))
        
    return -np.asarray(2 * (momentsDer * weight * moments.transpose())) \
                     .transpose()[0]


def probXCondlZ(xx, zz, xVals, zs, bwidth):
    """ Takes xx and zz the non-missing part of the sample, grid values from
    xVals and an n-by-k+1 array of (z, m) values (2D array) from the data set,
    and returns the kernel estimates for P[X=x|Z=z] for every x in xVals
    and z in zs as an n-by-xVals.shape[0] array. bwidth (list of 2 floats)
    contains the bandwidths to calculate the Nadaraya-Watson estimator for the
    conditional probabilities.
    """
    if zz.shape[1] > 1:
        zz = np.stack([zz[:, 1:]] * zs.shape[0])
        zis = np.stack([zs[:, 1:-1]] * zz.shape[1])
    elif zz == np.ones(zz.shape):
        return np.mean(xx)
    xVals = np.stack([xVals] * zz.shape[1])
    kernel1 = np.prod(normal.pdf((zz - np.einsum('ijk->jik', zis))
                                 / bwidth[0]), axis=-1)
    kernel2 = normal.pdf((xx - xVals) / bwidth[1])
    results = np.einsum('ij, jk -> ik', kernel1, kernel2) \
        / np.sum(kernel1, axis=1, keepdims=True)
    return results/np.sum(results, axis=1, keepdims=True)


def probXCondlZSpline(xx, zz, xVals, zs, knotno):
    """ Takes xx and zz the non-missing part of the sample, grid values from
    xVals and an n-by-2 array of (z, m) values (2D array) from the data set,
    and returns the natural cubic (b)spline estimates for P[X=x|Z=z] for every
    x in xVals and z in zs as an n-by-xVals.shape[1] array.
    'knotno' contains the number of knots as an estimation parameter, and knots
    are placed in the respective quantiles calculated from zs.
    """
    if zz.shape[1] > 1:
        zz = zz[:, 1:]
        zis = zs[:, 1]
    elif zz == np.ones(zz.shape):
        return np.mean(xx)

    # Getting knots from zis

    # Fitting

    # Predicting
    results = np.ones(zis.shape)
    return results/np.sum(results, axis=1, keepdims=True)


def probXCondlZOracle(z, xVals):
    """ Takes the z-s (n-by-kz 2D array) and grid values from xVals (1D array),
    and returns the true value for P[X=x|Z=z] for every x in xVals and z in z
    as an n-by-xVals.shape[0] array.
    """

    zcoeffs = np.array([0.6, -0.4, 0.7, 0.1, -0.3, 0.9, 1, -1, 0.5, -0.2]
                      [:z.shape[1]])
    z_stack = np.stack([np.sum(z * zcoeffs, axis=1)] * xVals.shape[0], axis=1)
    x_stack = np.stack([xVals] * z.shape[0])
    results = normal.pdf(-np.log(4 / (x_stack + 2) - 1) - z_stack)\
        * 4 / (4 - x_stack**2)
    return results/np.sum(results, axis=1, keepdims=True)


def yCondlOnZ(coeffs, probs, x, z, gridno):
    """ This is a not-so-dumb, but still very basic grid implementation of
    numerical integration for our simulation.
    Arguments:
    - coeffs: the coefficients 1D array at which you would like to evaluate the
              integral (and hence the objective function)
    - probs: conditional probabilities for P[X=xgridval|Z=z] for some xgridvals
             from a grid of X values generated linearly based on gridno
    - x: we need its length technically in the function, but it is completely
         useless here, the data from the full data set for x (2D array)
    - z: variable z from the data (both the missing and non-missing part,
         2D array)
    - gridno: number of grid points on the support of X (fineness of the grid
              for numerical integration)
    """
    # grid: make it a n-by-gridno array
    xGrid = np.stack([np.linspace(start=-2, stop=2, num=gridno)] * len(x))
    expectedYs = normal.cdf(xGrid * coeffs[0] + np.sum(z * coeffs[1:],
                                                       axis=1, keepdims=True))
    return np.sum(probs * expectedYs, axis=1, keepdims=True)


def yCondlOnZDer(coeffs, probs, x, z, gridno):
    """ This is a not-so-dumb, but still very basic grid implementation of
    numerical integration for our simulation.
    Arguments:
    """
    # grid: make it a n-by-gridno array
    xGrid = np.stack([np.linspace(start=-2, stop=2, num=gridno)] * len(x))
    expectedYsDerZ = normal.pdf(xGrid * coeffs[0] + np.sum(z * coeffs[1:],
                                                       axis=1, keepdims=True))
    expectedYsDerX = xGrid * normal.pdf(xGrid * coeffs[0] 
                                        + np.sum(z * coeffs[1:],
                                                 axis=1, keepdims=True))

    return [np.sum(probs * expectedYsDerZ, axis=1, keepdims=True), \
            np.sum(probs * expectedYsDerX, axis=1, keepdims=True)]


def yCondlOnZDirect(coeffs, xx, zz, z, bwidth):
    """ This is a direct NW implementation of the cond'l expectation of g_0.
    Arguments:
    - coeffs: the coefficients 1D array at which you would like to evaluate the
              integral (and hence the objective function)
    - x: we need its length technically in the function, but it is completely
         useless here, the data from the full data set for x (2D array)
    - z: variable z from the data (both the missing and non-missing part,
         2D array)
    """
    phis = normal.cdf(xx * coeffs[0] + np.sum(zz * coeffs[1:],
                                             axis=1, keepdims=True))
    if zz.shape[1] > 1:
        zz = np.stack([zz[:, 1:]] * z.shape[0])
        zis = np.stack([z[:, 1:]] * zz.shape[1])
    elif zz == np.ones(zz.shape):
        return np.mean(xx)

    kernelWeights = np.prod(normal.pdf((zz - np.einsum('ijk->jik', zis))
                                 / bwidth), axis=-1)
    kernelWeights = kernelWeights  \
                    / np.sum(kernelWeights, axis=1, keepdims=True)
    expectedPhis = np.einsum('ij,jk->ik', kernelWeights, phis)
    
    return expectedPhis


def imputeMoments(coeffs, probs, y, x, z, m, gridno, weight=None):
    """The objective function for the imputation estimator that uses the
    analogue of the AD 2017 moments in addition to the feasible moments in
    gmmNonMissingData.
    Its arguments are
    - coeffs: (k+1 numpy array) the coefficient values
    - probs: conditional probabilities for P[X=xgridval|Z=z] for some xgridvals
             from a grid of X values generated linearly based on gridno (array)
    - y, x, z: the data in separate arrays (n-by-1, n-by-1, n-by-k shapes)
    - m: missingness indicator as 2D array
    - gridno: number of grid points on the support of X (fineness of the grid
              for numerical integration)

    Returns the value of the GMM objective function as a float.
    """

    residuals1 = (1 - m) * (y-normal.cdf(x * coeffs[0]
                                         + np.sum(z * coeffs[1:],
                                                  axis=1, keepdims=True)))        
    residuals2 = m * (y - yCondlOnZ(coeffs, probs, x, z, gridno))

    moments = np.matrix(np.mean(np.hstack((x * residuals1,
                                       z * residuals1,
                                       z * residuals2)), axis=0))  
    if weight is None:
        weight = np.matrix(np.identity(moments.shape[1]))
    return (moments * weight * moments.transpose())[0, 0]


def imputeMomentsDirect(coeffs, y, x, z, m, xx, zz, bwidth, weight=None):
    """The objective function for the imputation estimator that uses the
    analogue of the AD 2017 moments in addition to the feasible moments in
    gmmNonMissingData.
    Its arguments are
    - coeffs: (k+1 numpy array) the coefficient values
    - probs: conditional probabilities for P[X=xgridval|Z=z] for some xgridvals
             from a grid of X values generated linearly based on gridno (array)
    - y, x, z: the data in separate arrays (n-by-1, n-by-1, n-by-k shapes)
    - m: missingness indicator as 2D array
    - gridno: number of grid points on the support of X (fineness of the grid
              for numerical integration)

    Returns the value of the GMM objective function as a float.
    """

    residuals1 = (1 - m) * (y-normal.cdf(x * coeffs[0]
                                         + np.sum(z * coeffs[1:],
                                                  axis=1, keepdims=True)))    
    residuals2 = m * (y - yCondlOnZDirect(coeffs, xx, zz, z, bwidth))      

    moments = np.matrix(np.mean(np.hstack((x * residuals1,
                                       z * residuals1,
                                       z * residuals2)), axis=0))  
    if weight is None:
        weight = np.matrix(np.identity(moments.shape[1]))
    return (moments * weight * moments.transpose())[0, 0]


def imputeMomentsWeights(coeffs, probs, y, x, z, m, gridno):
    """ Estimated optimal weights for the fully-observed moments as a function
    of (true) coefficients and the data.
    """
    residuals1 = (1 - m) * (y-normal.cdf(x * coeffs[0]
                                         + np.sum(z * coeffs[1:],
                                                  axis=1, keepdims=True)))
    residuals2 = m * (y - yCondlOnZ(coeffs, probs, x, z, gridno))
    moments = np.matrix(np.hstack((x * residuals1,
                                   z * residuals1,
                                   z * residuals2)))
    return linalg.inv((moments.transpose() * moments) / y.shape[0])


def imputeMomentsWeightsDirect(coeffs, y, x, z, m, xx, zz, bwidth):
    """ Estimated optimal weights for the fully-observed moments as a function
    of (true) coefficients and the data.
    """
    residuals1 = (1 - m) * (y-normal.cdf(x * coeffs[0]
                                         + np.sum(z * coeffs[1:],
                                                  axis=1, keepdims=True)))
    residuals2 = m * (y - yCondlOnZDirect(coeffs, xx, zz, z, bwidth))
    moments = np.matrix(np.hstack((x * residuals1,
                                   z * residuals1,
                                   z * residuals2)))
    return linalg.inv((moments.transpose() * moments) / y.shape[0])


def imputeMomentsJacobian(coeffs, probs, y, x, z, m, gridno, weight):
    residuals1 = (1 - m) * (y-normal.cdf(x * coeffs[0]
                                         + np.sum(z * coeffs[1:],
                                                  axis=1, keepdims=True)))        
    residuals2 = m * (y - yCondlOnZ(coeffs, probs, x, z, gridno))

    moments = np.matrix(np.mean(np.hstack((x * residuals1,
                                       z * residuals1,
                                       z * residuals2)), axis=0))  

    residuals1Der = (1 - m) * (normal.pdf(x * coeffs[0]
                                         + np.sum(z * coeffs[1:],
                                                  axis=1, keepdims=True)))

    xTilde =np.hstack((x,z))
    residuals2DerZ = m * (yCondlOnZDer(coeffs, probs, x, z, gridno)[0])
    residuals2DerX = m * (yCondlOnZDer(coeffs, probs, x, z, gridno)[1])
    momentsDer0 = np.einsum('ij,ik -> jk', xTilde * residuals1Der, xTilde) \
                 /len(y)
    momentsDer1 = np.einsum('ij,ik -> jk', z * residuals2DerZ, z) \
                 /len(y)
    momentsDer2 = np.mean(z * residuals2DerX, axis=0)
    
    momentsDer= np.matrix(np.hstack((momentsDer0,
                                     np.vstack((momentsDer2, momentsDer1)))))  
    if weight is None:
        weight = np.matrix(np.identity(moments.shape[1]))
        
    return -np.asarray(2 * (momentsDer * weight * moments.transpose())) \
                     .transpose()[0]

    
def imputeMomentsHessian(coeffs, probs, y, x, z, m, gridno, weight=None):
    
    return


def gmmImpute(y, x, z, m, a0, b0, gridno, weighting_iteration=1, 
              method='spline', direct=True, noise=False):
    """ The feasible GMM estimator that adds the analogues of the AD 2017
    moments to the moments of the gmmNonMissingData estimator.
    Arguments:
    - y, x, z: variables from the data as (n-by-1, n-by-1, n-by-k+1) 2D arrays
    - m: missingness indicator, another 2D array (n-by-1)
    - a0, b0: the initial values for maximization (1D arrays) - the dimensions
              must agree with the number of columns in x and z
    - gridno: (integer) number of grid points on the support of X (fineness of
              the grid for numerical integration)
    - bwidth: a list-like of two floats, where the first number is the (equal)
              bwidth for the kernels for the Z dimensions, and the second one
              is the bandwidth for the normal kernel for the pdf estimator for
              the X
    - noise: boolean, set it to True if want to print the messages of the
             optimizer (automatically suppressed when iteration() is run with
             the MC decorator)
    """
    # initializing variables
    a0 = np.array(a0, ndmin=1)
    b0 = np.array(b0)
    coeffs0 = np.concatenate((a0, b0))
    n = y.shape[0]
    # nonmissing sample, grid, calculating probs (note this should be cached)
    xx = x[np.squeeze((m == 0))]
    zz = z[np.squeeze((m == 0))]
    xVals = np.linspace(start=-1.999, stop=1.999, num=gridno)
    if method == 'spline':
        knotno = int(gridno / (np.exp(n**1/2) + 1))
        probs = probXCondlZSpline(xx, zz, xVals, np.hstack((z, m)), knotno)
    elif method == 'NW':
        bwidth1 = [2.154 * n**(-1/4), 1.077 * n**(-1/3)]
        if direct:
            bwidth1 = 1.77 * n**(-1/3)
        else:
            probs = probXCondlZ(xx, zz, xVals, np.hstack((z, m)), bwidth1)
    elif method == 'oracle':
        probs = probXCondlZOracle(z, xVals)
    # optimization
    weight = None
    if direct:
        for i in range(weighting_iteration):
            optimum = opt.minimize(
                        imputeMomentsDirect, coeffs0,
                        args=(y, x, z, m, xx, zz, bwidth1, weight),
                        method='BFGS')
            coeffs0 = optimum.x
            weight = imputeMomentsWeightsDirect(coeffs0, y, x, z, m,
                                                xx, zz, bwidth1)
        optimum = opt.minimize(
                imputeMomentsDirect, coeffs0,
                args=(y, x, z, m, xx, zz, bwidth1, weight),
                method='BFGS')        
    else:
        for i in range(weighting_iteration):
            optimum = opt.minimize(
                        imputeMoments, coeffs0,
                        args=(probs, y, x, z, m, gridno, weight),
                        method='Newton-CG',
                        jac=imputeMomentsJacobian)
            coeffs0 = optimum.x
            weight = imputeMomentsWeights(coeffs0, probs, y, x, z, m, gridno)
        optimum = opt.minimize(
                imputeMoments, coeffs0,
                args=(probs, y, x, z, m, gridno, weight),
                method='Newton-CG',
                jac=imputeMomentsJacobian)
    if noise:
        print('\n')
        print(optimum, flush=True)
        print('\n')
    return optimum.x


def yCondlOnMarginalZs(coeffs, probsvector, x, z, gridno):
    """ This is a not-so-dumb, but still very basic grid implementation of
    numerical integration for our simulation. NEEDS WORK TO BRING UP TO v0.2
    """
    # grid: make it a n-by-gridno array
    xGrid = np.tile(np.linspace(start=-2, stop=2, num=gridno), len(x)) \
        .reshape((len(x), gridno))
    expectedYs = normal.cdf(xGrid * coeffs[0] + np.sum(z * coeffs[1:],
                            axis=1, keepdims=True))
    return np.hstack(tuple([np.sum(prob * expectedYs, axis=1, keepdims=True)
                            for prob in probsvector]))


def marginalizedImputeMoments(coeffs, probsvector, y, x, z, m, gridno):
    """ MAY NEED WORK TO BRING IT UP TO v0.2 """
    residuals1 = (1 - m) * (y-normal.cdf(x * coeffs[0]
                                         + np.sum(z * coeffs[1:],
                                                  axis=1, keepdims=True)))
    residuals2 = m * (y - yCondlOnMarginalZs(
                                coeffs, probsvector, x, z, gridno))
    moments = np.matrix(np.mean(np.hstack((x * residuals1,
                                           z * residuals1,
                                           z * residuals2)), axis=0))
    return (moments*moments.transpose())[0, 0]


def gmmMarginalizedImpute(y, x, z, m, a0, b0, gridno, bwidth, noise=False):
    """ NEEDS WORK TO BRING IT UP TO v0.2 """
    a0 = np.array(a0, ndmin=1)
    b0 = np.array(b0)
    coeffs0 = np.concatenate((a0, b0))

    # nonmissing sample, grid, calculating probs (note this should be cached)
    xx = x[np.squeeze((m == 0))]
    zz = z[np.squeeze((m == 0))]
    xVals = np.linspace(start=-2, stop=2, num=gridno)
    probsvector = np.array([probXCondlZ(
                                xx, zz[:, i:i+1], xVals,
                                np.hstack((z[:, i:i+1], m)), bwidth)
                            for i in range(z.shape[1])])

    optimum = opt.minimize(
                marginalizedImputeMoments, coeffs0,
                args=(probsvector, y, x, z, m, gridno), method='BFGS',
                options={'disp': noise})
    if noise:
        print(optimum.x, flush=True)
    return optimum.x


def iteration(n, noise=False):
    """ One iteration of the simulation containing data generation and fitting
    the four estimators (full data set GMM, nonmissing data set GMM,
    AD imputation GMM, marginalized imputation GMM)
    Returns a numpy array.
    """
    # bwidth2 = [0.1, 0.1]
    y, x, z, m = dgp(alpha, beta, n, noise)
    fullDataRes = gmmFullData(y, x, z, a0, b0, 1, noise)
    nonMissingDataRes = gmmNonMissingData(y, x, z, m, a0, b0, 1, noise)
    imputeOracleRes = gmmImpute(y, x, z, m, a0, b0, gridno, 1, 'oracle', direct, noise)
    imputeNWRes = gmmImpute(y, x, z, m, a0, b0, gridno, 1, 'NW', direct, noise)
    # marginalizedImputeRes = gmmMarginalizedImpute(y, x, z, m, a0, b0, gridno,
    #                                              bwidth2, noise)
    # DO THE STUFF: BLOCK MATRIX
    return np.array((fullDataRes, nonMissingDataRes, imputeOracleRes, imputeNWRes))
# , marginalizedImputeRes))
        

def multiIteration(task_queue, done_queue):
        while True:
            args = task_queue.get()
            if args == 'STOP':
                break
            result = iteration(args, noise)
            print(result)
            done_queue.put(result)


def montecarloMulti(oldfuggveny):
    """ Monte Carlo for a function containing one iteration of a
    simulation.

    Simulation parameters needed as global vars:
    - nlist: list(like) object containing sample sizes
    - reps: int, number of replications
    - fname: name of file where the aggregate results (mean, st. dev.)
             are to be printed
    - resultsFname: name of file where the list of estimate values are
                    to be printed

    EVERY TIME YOU RECYCLE, FILL IN THE 'meanlist' and 'stdlist' lines !!!
    You may also want to add names of estimators to print into the file if
    there are too many to compare.

    NOTE: REORGANIZE estimation parameter printing too.
     """

    for n in nlist:
        t0 = time.time()
        
        task_queue = mp.Queue()
        done_queue = mp.Queue()

        for i in range(reps):
            task_queue.put(n)
            
        # creating jobs
        jobs = [mp.Process(target=oldfuggveny,
                           args=(task_queue, done_queue)) \
                for i in range(cores)]    
        
        for i in range(cores):      # starting the jobs
            jobs[i].start()
            time.sleep(1)
        
        for i in range(cores):      # giving the pill
            task_queue.put('STOP') 
        
        # getting the results
        results = np.stack([done_queue.get() for i in range(reps)])
            
        for i in range(cores):      # joining the jobs
            jobs[i].join()    

        # THIS NEEDS TO BE CHANGED (only), add name if you would like to!
        meanlist = np.mean(results, axis=0)
        stdlist = np.std(results, axis=0)

        if resultsFname:
            with open(resultsFname + str(n) + '.npy', 'wb') as f:
                np.save(f, results)
        print("")
        print('n= ', n)
        print('Means: \n', meanlist)
        print('Std. devs.: \n', stdlist)
        print("")
        print("This took", time.time()-t0, 's')
        print("")
        print("")
        with open(fname, 'a') as f:
            f.write('\n\n\nn= '+str(n))
            f.write('\n\nmeans: \n'+str(meanlist))
            f.write('\n\nstandard deviations: \n' + str(stdlist))
            f.write('\n\nThis took ' + str(time.time() - t0) + ' s\n')

    
if __name__== '__main__':
    mp.freeze_support()
    
    ###################
    #    SIMULATION   #
    ###################
    
    with open(fname, 'w') as f:
        f.write('beta= ' + str(beta) + '\nalpha= ' + str(alpha) + '\nseed= '
                + str(seed) + '\nreps= ' + str(reps) + '\n' 
                + 'direct:' + str(direct) + '\n')
    
    random.seed(seed)
    montecarloMulti(multiIteration)
    print('\a')


