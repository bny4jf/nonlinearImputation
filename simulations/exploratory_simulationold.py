#  -*- coding: utf-8 -*-

"""
09/16/2020
@author: Peter Toth, UNR

Explanatory Monte Carlo simulation for imputation project: probit model with a
single imputed occasionally missing RHS variable (x) using one, two, five other
(non-missing) RHS variables (z-s)

Packages needed: numpy, scipy

NOTES:
- we regenerate the missingness indicators every time
"""

import numpy as np
from scipy import optimize as opt
from scipy import linalg
from scipy.stats import norm as normal
import time

arr = np.array([[0.6325] ,
[0.67246094] ,
[0.75765625] ,
[0.680625] ,
[0.67363281] ,
[0.66982422] ,
[0.66025391] ,
[0.63759766] ,
[0.6584375] ,
[0.72] ,
[0.6765625] ,
[0.63095703] ,
[0.696875] ,
[0.71867188] ,
[0.69875] ,
[0.78623047] ,
[0.69109375] ,
[0.6775] ,
[0.65226563] ,
[0.70375] ,
[0.68417969] ,
[0.68117188] ,
[0.73066406] ,
[0.61453125] ,
[0.74746094] ,
[0.64460938] ,
[0.79953125] ,
[0.69125] ,
[0.66445312] ,
[0.68140625] ,
[0.65421875] ,
[0.678125] ,
[0.72960938] ,
[0.71289062] ,
[0.6765625] ,
[0.62148437] ,
[0.73507813] ,
[0.70039062] ,
[0.6646875] ,
[0.63453125] ,
[0.72742188] ,
[0.67177734] ,
[0.70859375] ,
[0.66337891] ,
[0.6996875] ,
[0.65576172] ,
[0.72480469] ,
[0.76117188] ,
[0.67958984] ,
[0.68886719] ,
[0.69515625] ,
[0.60322266] ,
[0.67109375] ,
[0.69697266] ,
[0.67792969] ,
[0.72011719] ,
[0.67324219] ,
[0.655] ,
[0.60929688] ,
[0.78265625] ,
[0.628125] ,
[0.68328125] ,
[0.76929688] ,
[0.65875] ,
[0.66435547] ,
[0.71382813] ,
[0.64890625] ,
[0.62953125] ,
[0.668125] ,
[0.72234375] ,
[0.72978516] ,
[0.69267578] ,
[0.65595703] ,
[0.71046875] ,
[0.65966797] ,
[0.73] ,
[0.66572266] ,
[0.66375] ,
[0.62988281] ,
[0.735625] ,
[0.64289063] ,
[0.68125] ,
[0.74345703] ,
[0.71210937] ,
[0.6934375] ,
[0.79734375] ,
[0.64375] ,
[0.6421875] ,
[0.6984375] ,
[0.66234375] ,
[0.6334375] ,
[0.69085938] ,
[0.64355469] ,
[0.67984375] ,
[0.65839844] ,
[0.70882813] ,
[0.68886719] ,
[0.7103125] ,
[0.69984375] ,
[0.72568359] ,
[0.76738281] ,
[0.67882813] ,
[0.69953125] ,
[0.66621094] ,
[0.67125] ,
[0.69195313] ,
[0.71923828] ,
[0.72125] ,
[0.65882813] ,
[0.60761719] ,
[0.636875] ,
[0.74726562] ,
[0.6746875] ,
[0.69109375] ,
[0.65517578] ,
[0.68398437] ,
[0.75800781] ,
[0.6634375] ,
[0.7140625] ,
[0.7075] ,
[0.68876953] ,
[0.68916016] ,
[0.62804688] ,
[0.690625] ,
[0.653125] ,
[0.61054688] ,
[0.66734375] ,
[0.6596875] ,
[0.671875] ,
[0.665625] ,
[0.6984375] ,
[0.77265625] ,
[0.63730469] ,
[0.653125] ,
[0.64482422] ,
[0.72453125] ,
[0.62015625] ,
[0.71708984] ,
[0.695] ,
[0.70835938] ,
[0.6640625] ,
[0.67296875] ,
[0.68375] ,
[0.70332031] ,
[0.68837891] ,
[0.69007813] ,
[0.68507813] ,
[0.66015625] ,
[0.64345703] ,
[0.66347656] ,
[0.70625] ,
[0.66703125] ,
[0.6475] ,
[0.66054688] ,
[0.68875] ,
[0.6775] ,
[0.6396875] ,
[0.60960938] ,
[0.74460938] ,
[0.73421875] ,
[0.72724609] ,
[0.685625] ,
[0.655] ,
[0.61875] ,
[0.72203125] ,
[0.75449219] ,
[0.64101563] ,
[0.75332031] ,
[0.66015625] ,
[0.71132813] ,
[0.65351562] ,
[0.71609375] ,
[0.77953125] ,
[0.71738281] ,
[0.66609375] ,
[0.63427734] ,
[0.67335938] ,
[0.73115234] ,
[0.69599609] ,
[0.78335938] ,
[0.63144531] ,
[0.82167969] ,
[0.659375] ,
[0.62453125] ,
[0.66582031] ,
[0.70945313] ,
[0.71] ,
[0.7878125] ,
[0.73828125] ,
[0.67539063] ,
[0.71210938] ,
[0.63349609] ,
[0.72453125] ,
[0.61757812] ,
[0.6453125] ,
[0.71201172] ,
[0.69859375] ,
[0.80292969] ,
[0.76210937] ,
[0.70429688] ,
[0.77] ,
[0.72080078] ,
[0.76386719] ,
[0.69742188] ,
[0.724375] ,
[0.59171875] ,
[0.70761719] ,
[0.69960938] ,
[0.70015625] ,
[0.780625] ,
[0.67390625] ,
[0.69306641] ,
[0.73625] ,
[0.66894531] ,
[0.68414063] ,
[0.64875] ,
[0.6321875] ,
[0.62060547] ,
[0.62138672] ,
[0.65703125] ,
[0.6534375] ,
[0.60546875] ,
[0.73476563] ,
[0.7140625] ,
[0.6996875] ,
[0.6446875] ,
[0.69726562] ,
[0.66242188] ,
[0.65722656] ,
[0.66164063] ,
[0.69132813] ,
[0.64912109] ,
[0.679375] ,
[0.68953125] ,
[0.71601562] ,
[0.69484375] ,
[0.73375] ,
[0.69046875] ,
[0.69072266] ,
[0.67132813] ,
[0.5865625] ,
[0.69404297] ,
[0.69625] ,
[0.64033203] ,
[0.6878125] ,
[0.67246094] ,
[0.68476562] ,
[0.68742188] ,
[0.6578125] ,
[0.70109375] ,
[0.71773438] ,
[0.67587891] ,
[0.6596875] ,
[0.63916016] ,
[0.66789063] ,
[0.665] ,
[0.68769531] ,
[0.665625] ,
[0.7390625] ,
[0.67412109] ,
[0.669375] ,
[0.65527344] ,
[0.69875] ,
[0.63460938] ,
[0.67484375] ,
[0.71234375] ,
[0.58964844] ,
[0.73408203] ,
[0.64523438] ,
[0.69863281] ,
[0.660625] ,
[0.65734375] ,
[0.6453125] ,
[0.75371094] ,
[0.671875] ,
[0.65957031] ,
[0.71464844] ,
[0.6890625] ,
[0.75109375] ,
[0.62636719] ,
[0.62039063] ,
[0.70009766] ,
[0.63382813] ,
[0.69394531] ,
[0.64306641] ,
[0.62234375] ,
[0.64789063] ,
[0.77945313] ,
[0.74443359] ,
[0.70136719] ,
[0.743125] ,
[0.66375] ,
[0.698125] ,
[0.67578125] ,
[0.67890625] ,
[0.71083984] ,
[0.72822266] ,
[0.66201172] ,
[0.69351563] ,
[0.7284375] ,
[0.71416016] ,
[0.69677734] ,
[0.64453125] ,
[0.6575] ,
[0.70125] ,
[0.69414062] ,
[0.72246094] ,
[0.68546875] ,
[0.72273438] ,
[0.736875] ,
[0.67601563] ,
[0.64238281] ,
[0.7128125] ,
[0.69085938] ,
[0.65703125] ,
[0.71539063] ,
[0.69589844] ,
[0.69492188] ,
[0.66757812] ,
[0.6909375] ,
[0.68710937] ,
[0.83398437] ,
[0.740625] ,
[0.69625] ,
[0.66279297] ,
[0.79265625] ,
[0.72921875] ,
[0.70458984] ,
[0.69697266] ,
[0.76357422] ,
[0.75015625] ,
[0.66230469] ,
[0.74609375] ,
[0.62007813] ,
[0.65498047] ,
[0.61796875] ,
[0.67128906] ,
[0.64785156] ,
[0.67597656] ,
[0.68710938] ,
[0.65984375] ,
[0.6646875] ,
[0.62822266] ,
[0.66875] ,
[0.76074219] ,
[0.66445313] ,
[0.69046875] ,
[0.70332031] ,
[0.68515625] ,
[0.7015625] ,
[0.67480469] ,
[0.73898438] ,
[0.63023438] ,
[0.72236328] ,
[0.64492187] ,
[0.65859375] ,
[0.76851563] ,
[0.6125] ,
[0.64125] ,
[0.66804688] ,
[0.65605469] ,
[0.72453125] ,
[0.70679688] ,
[0.7] ,
[0.6871875] ,
[0.734375] ,
[0.6521875] ,
[0.62363281] ,
[0.67177734] ,
[0.68703125] ,
[0.72070312] ,
[0.73398437] ,
[0.770625] ,
[0.71765625] ,
[0.70566406] ,
[0.71845703] ,
[0.67492188] ,
[0.67138672] ,
[0.56460938] ,
[0.6740625] ,
[0.61789063] ,
[0.64873047] ,
[0.63070313] ,
[0.76109375] ,
[0.68171875] ,
[0.71435547] ,
[0.73759766] ,
[0.71748047] ,
[0.70453125] ,
[0.63320312] ,
[0.694375] ,
[0.63662109] ,
[0.74111328] ,
[0.71445312] ,
[0.70296875] ,
[0.6628125] ,
[0.68742188] ,
[0.6446875] ,
[0.73320313] ,
[0.63359375] ,
[0.59515625] ,
[0.68691406] ,
[0.71347656] ,
[0.69125] ,
[0.62960938] ,
[0.69828125] ,
[0.63759766] ,
[0.67242188] ,
[0.68925781] ,
[0.72021484] ,
[0.70078125] ,
[0.70921875] ,
[0.68257813] ,
[0.68740234] ,
[0.62875] ,
[0.710625] ,
[0.72382812] ,
[0.699375] ,
[0.67984375] ,
[0.70615234] ,
[0.7125] ,
[0.63691406] ,
[0.71777344] ,
[0.70421875] ,
[0.7215625] ,
[0.75125] ,
[0.70875] ,
[0.66] ,
[0.69429688] ,
[0.62324219] ,
[0.71171875] ,
[0.69580078] ,
[0.66203125] ,
[0.7109375] ,
[0.65265625] ,
[0.79033203] ,
[0.65263672] ,
[0.72363281] ,
[0.66164063] ,
[0.75015625] ,
[0.70058594] ,
[0.73726563] ,
[0.67744141] ,
[0.62171875] ,
[0.65929688] ,
[0.73652344] ,
[0.60703125] ,
[0.65390625] ,
[0.69863281] ,
[0.60757813] ,
[0.73818359] ,
[0.64697266] ,
[0.68847656] ,
[0.64228516] ,
[0.71875] ,
[0.68109375] ,
[0.66796875] ,
[0.63585938] ,
[0.68890625] ,
[0.76386719] ,
[0.6959375] ,
[0.73740234] ,
[0.64015625] ,
[0.69609375] ,
[0.69914063] ,
[0.70195312] ,
[0.6984375] ,
[0.783125] ,
[0.71585938] ,
[0.689375] ,
[0.69101562] ,
[0.72625] ,
[0.61132813] ,
[0.63769531] ,
[0.68857422] ,
[0.62695312] ,
[0.68890625] ,
[0.72109375] ,
[0.62949219] ,
[0.665625] ,
[0.67949219] ,
[0.80734375] ,
[0.69554688] ,
[0.6240625] ,
[0.64140625] ,
[0.69775391] ,
[0.67023438] ,
[0.7434375] ,
[0.75367188] ,
[0.64515625] ,
[0.78154297] ,
[0.65007813] ,
[0.67234375] ,
[0.70820313] ,
[0.71337891] ,
[0.68867187] ,
[0.79550781] ,
[0.68296875] ,
[0.63574219] ,
[0.76630859] ,
[0.71953125] ,
[0.661875] ,
[0.65804688] ,
[0.60890625] ,
[0.68203125] ,
[0.67578125] ,
[0.64150391] ,
[0.61699219] ,
[0.605] ,
[0.69746094] ,
[0.7478125] ,
[0.6228125] ,
[0.73328125] ,
[0.70851563] ,
[0.67314453] ,
[0.74501953] ,
[0.61335938] ,
[0.73691406] ,
[0.72984375] ,
[0.66230469] ,
[0.634375] ,
[0.6584375] ,
[0.71005859] ,
[0.68054688] ,
[0.67177734] ,
[0.63078125] ,
[0.69453125] ,
[0.76054688] ,
[0.62390625] ,
[0.6375] ,
[0.7109375] ,
[0.64742188] ,
[0.76359375] ,
[0.585625] ,
[0.7540625] ,
[0.71875] ,
[0.68695313] ,
[0.79130859] ,
[0.7721875] ,
[0.66865234] ,
[0.66773438] ,
[0.72929687] ,
[0.76699219] ,
[0.59710938] ,
[0.66279297] ,
[0.71044922] ,
[0.63066406] ,
[0.60625] ,
[0.66640625] ,
[0.623125] ,
[0.73828125] ,
[0.70054688] ,
[0.68976563] ,
[0.83226563] ,
[0.68066406] ,
[0.66308594] ,
[0.70375] ,
[0.67734375] ,
[0.75515625] ,
[0.71546875] ,
[0.68140625] ,
[0.73789062] ,
[0.68457031] ,
[0.72539062] ,
[0.69875] ,
[0.60625] ,
[0.66132813] ,
[0.70693359] ,
[0.72] ,
[0.64296875] ,
[0.70917969] ,
[0.7275] ,
[0.69929688] ,
[0.7140625] ,
[0.68390625] ,
[0.6359375] ,
[0.7684375] ,
[0.68171875] ,
[0.69707031] ,
[0.71570313] ,
[0.70007813] ,
[0.67734375] ,
[0.68525391] ,
[0.64345703] ,
[0.67203125] ,
[0.69953125] ,
[0.63769531] ,
[0.66816406] ,
[0.60683594] ,
[0.63882813] ,
[0.625] ,
[0.66767578] ,
[0.72421875] ,
[0.65234375] ,
[0.6109375] ,
[0.72945313] ,
[0.6896875] ,
[0.71710938] ,
[0.76273438] ,
[0.68505859] ,
[0.734375] ,
[0.70445313] ,
[0.7303125] ,
[0.75382813] ,
[0.69726562] ,
[0.75359375] ,
[0.6825] ,
[0.68037109] ,
[0.67828125] ,
[0.70171875] ,
[0.678125] ,
[0.70166016] ,
[0.7346875] ,
[0.78953125] ,
[0.72625] ,
[0.72333984] ,
[0.6196875] ,
[0.7] ,
[0.70126953] ,
[0.66625] ,
[0.7075] ,
[0.64296875] ,
[0.73642578] ,
[0.64326172] ,
[0.6915625] ,
[0.66513672] ,
[0.6915625] ,
[0.70644531] ,
[0.75615234] ,
[0.703125] ,
[0.73109375] ,
[0.66191406]])

print(np.mean(arr))
print(np.std(arr))

import matplotlib.pyplot as plt
cumarr = np.array([np.mean(arr[:i,0]) for i in range(1,len(arr))])
plt.plot(cumarr)

with open('C:/Users/harry/Dropbox/UNR 2020 Fall/Jason 2/simulations/logitFull12v2Res200', 'br') as f:
    arr2 = np.load(f)

cumarr2 = np.array([np.mean(arr2[:i,0]) for i in range(1,len(arr2))])
plt.plot(cumarr2)


#############################
#   SIMULATION PARAMETERS   #
#############################

# True values
alpha = 1
beta = [0.5, -2]

# Initial values
a0 = 0
b0 = [0, 0]

# Replication number, sample sizes
reps = 3
nlist = [100, 200]

# File names (fname is req'd, stores the aggregate results, resultsFname
# can be set to False)
fname = 'OracleVsNWzz.txt'
resultsFname = 'OracleVsNWReszz'

# Random seed (not implemented), noisiness (should be True only for dev)
seed = 2433523
noise = False

# Fineness of grid; bwidth is locked in with a 1/3 rate
gridno = 100


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
                    method='BFGS')
        coeffs0 = optimum.x
        weight = fullDataWeights(coeffs0, y, x, z)
    optimum = opt.minimize(
                fullDataMoments, coeffs0, args=(y, x, z, weight),
                method='BFGS')
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


def probXCondlZ(xx, zz, xVals, zs, bwidth):
    """ Takes xx and zz the non-missing part of the sample, grid values from
    xVals and an n-by-k+1 array of (z, m) values (2D array) from the data set,
    and returns the kernel estimates for P[X=x|Z=z] for every x in xVals
    and z in zs as an n-by-xVals.shape[1] array. bwidth (list of 2 floats)
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
    """ Takes the z-s (1D array) and grid values from xVals (1D array),
    and returns the true value for P[X=x|Z=z] for every x in xVals and z in z
    as an n-by-xVals.shape[1] array.
    """
    if z.shape[1] > 1:
        zs = z[:, 1]
    else:
        zs = z
    z_stack = np.stack([zs] * xVals.shape[0], axis=1)
    x_stack = np.stack([xVals] * z.shape[0])
    results = normal.pdf(-np.log(4 / (x_stack + 2) - 1) - 0.6 + 0.4 * z_stack)\
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


def gmmImpute(y, x, z, m, a0, b0, gridno,
              weighting_iteration=1, method='spline', noise=False):
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
        bwidth1 = [2.154 * n**(-1/3), 1.077 * n**(-1/3)]
        probs = probXCondlZ(xx, zz, xVals, np.hstack((z, m)), bwidth1)
    elif method == 'oracle':
        probs = probXCondlZOracle(z, xVals)
    # optimization
    weight = None
    for i in range(weighting_iteration):
        optimum = opt.minimize(
                    imputeMoments, coeffs0,
                    args=(probs, y, x, z, m, gridno, weight),
                    method='Nelder-Mead')
        coeffs0 = optimum.x
        weight = imputeMomentsWeights(coeffs0, probs, y, x, z, m, gridno)
    optimum = opt.minimize(
            imputeMoments, coeffs0, args=(probs, y, x, z, m, gridno, weight),
            method='BFGS', options={'disp': noise})
    if noise:
        print(optimum.message)
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
        print(optimum.message)
    return optimum.x


def montecarlo(oldfuggveny):
    """ Monte Carlo decorator for a function containing one iteration of a
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
    def iterations(*args):
        for n in nlist:
            t0 = time.time()
            results = np.stack([oldfuggveny(n, *args) for i in range(reps)])

            # THIS NEEDS TO BE CHANGED (only), add name if you would like to!
            meanlist = np.mean(results, axis=0)
            stdlist = np.std(results, axis=0)

            if resultsFname:
                with open(resultsFname+str(n), 'ab') as f:
                    np.save(f, results)
            print("")
            print('n=', n)
            print('Means:', meanlist)
            print('Std. devs.:', stdlist)
            print("")
            print("This took", time.time()-t0, 's')
            print("")
            print("")
            with open(fname, 'a') as f:
                f.write('\n\n\nn= '+str(n))
                f.write('\n\nmeans: \n'+str(meanlist))
                f.write('\n\nstandard deviations: \n' + str(stdlist))
                f.write('\n\nThis took ' + str(time.time() - t0) + ' s\n')

    return iterations


@montecarlo
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
    imputeOracleRes = gmmImpute(y, x, z, m, a0, b0, gridno, 1, 'oracle', noise)
    imputeNWRes = gmmImpute(y, x, z, m, a0, b0, gridno, 1, 'NW', noise)
    # marginalizedImputeRes = gmmMarginalizedImpute(y, x, z, m, a0, b0, gridno,
    #                                              bwidth2, noise)
    # DO THE STUFF: BLOCK MATRIX
    return np.array((fullDataRes, nonMissingDataRes, imputeOracleRes,
                    imputeNWRes))
# , marginalizedImputeRes))


###################
#    SIMULATION   #
###################

with open(fname, 'w') as f:
    f.write('beta= ' + str(beta) + '\nalpha= ' + str(alpha) + '\nseed= '
            + str(seed) + '\nreps= ' + str(reps) + '\n')

iteration()
print('\a')
