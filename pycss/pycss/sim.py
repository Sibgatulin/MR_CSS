import numpy as np
from numba import njit, prange
from pycss import css, residual
from pycss.utils import nbutils


@njit
def get_Amatrix(TE_s, pm):
    """set up A matrix from voxel signal equation
    of dimension #echoes x #species

    :param TE_s: 1d np.array, with echo times in seconds
    :param pm: 2d np.ndarray, parameter matrix of dimension #species x 4
    :returns: A matrix
    :rtype: 2d np.ndarray

    """
    A = np.zeros((len(TE_s), pm.shape[0]), dtype=np.complex128)
    for i in range(A.shape[1]):
        A[:, i] = np.exp((2j * np.pi * pm[i, 2] - pm[i, 3]) * TE_s)
    return A


@njit
def build_signal(TE_s, pm):
    """forward simulate MR signal at echo times TE_s

    :param TE_s: 1d np.array, with echo times in seconds
    :param pm: 2d np.ndarray, parameter matrix of dimension #species x 4
    :returns:
    :rtype: complex 1d np.ndarray

    """
    rho = pm[:, 0] * np.exp(1.j * pm[:, 1])
    A = get_Amatrix(TE_s, pm)
    return np.dot(A, rho.astype(A.dtype))


@njit
def add_noise(signal, SNR):
    """add noise with SNR to signal

    :param signal: complex 1d np.array, sampled MR signal at echo times TE_s
    :param SNR: float
    :returns: noisy signal
    :rtype: complex 1d np.array, sampled MR signal at echo times TE_s w/ noise

    """
    sz = signal.shape
    noise_r = np.random.normal(0, 1/SNR, sz)
    noise_i = np.random.normal(0, 1/SNR, sz)
    return (signal.real + noise_r) + 1j * (signal.imag + noise_i)


@njit
def get_TE_s(nTE, TE1_s, dTE_s, nAcq=1, acq_shift=(0.5)):
    """get 1d numpy array with echo times (TE) in seconds

    :param nTE: int, number of echos
    :param TE1_s: float, first echo time
    :param dTE_s: float, echo spacing in seconds in one acquisition
    :param nAcq: int, number of acquisitions
    :param acq_shift: float in (0, 1), shift in ratios of 1 dTE_s
    :returns: echo times in seconds
    :rtype: numpy.array

    """

    TE_s_interleaf = TE1_s + np.arange(nTE) * dTE_s

    TE_s_array = np.zeros((nTE, nAcq))
    TE_s_array[:, 0] = TE_s_interleaf

    nShifts = len(acq_shift)
    shifts = np.zeros(nAcq - 1)
    for i in range(nShifts):
        shifts[i] = acq_shift[i]

    if nShifts < nAcq - 1:
        for i in range(nShifts, nAcq - 1):
            shifts[i] = acq_shift[-1]

    for i in range(1, nAcq):
        TE_s_array[:, i] = TE_s_array[:, i-1] + shifts[i-1] * dTE_s

    return TE_s_array.ravel()


@njit
def get_dTEeff_s(TE_s):
    return np.diff(TE_s)


@njit
def estimate_bias(TE_s, pm_true, pm_init,
                  Cm, Cp, Cf, Cr, nlparams, shape,
                  tol, itermax):
    sig = css.build_signal(TE_s, pm_true)

    pm0, min_residual = residual.get_global_residual_minimum(TE_s, sig,
                                                             pm_init.copy(),
                                                             Cm, Cp, Cf, Cr,
                                                             nlparams, shape)
    pme, resnorm, iterations = css.varpro(TE_s, sig, pm0,
                                            Cm, Cp, Cf, Cr,
                                            tol, itermax)
    return pm_true - pme


@njit
def assemble_Pm(nlparams, shape, Cp, Cf, Cr, pm_init):
    """create an list of parameter matrices for different parameter values

    :param nlparams: 1d iterable, flattend parameter types and values
    :param shape: 1d iterable, numbers of unique values per type
    :param Cp: 2d np.ndarray, constraints matrix for phases
    :param Cf: 2d np.ndarray, constraints matrix for resonance frequencies
    :param Cr: 2d np.ndarray, constraints matrix for relaxation rates
    :returns: Pm
    :rtype: 3d numpy.ndarray, #params_combinations x #species x #4

    """
    Cm_dummy = np.zeros_like(Cp)
    M, Ntot, Nm, Np, Nf, Nr = css.get_paramsCount(Cm_dummy, Cp, Cf, Cr)

    params_combinations = get_params_combinations(nlparams, shape)
    Pm = np.zeros((params_combinations.shape[0], M, 4))
    for i in range(params_combinations.shape[0]):
        params = params_combinations[i, :]
        Pm[i, :, 1] = pm_init[:, 1] + np.dot(Cp.astype(params.dtype),
                                             params[Nm:Nm+Np])
        Pm[i, :, 2] = pm_init[:, 2] + np.dot(Cf.astype(params.dtype),
                                             params[Nm+Np:Nm+Np+Nf])
        Pm[i, :, 3] = pm_init[:, 3] + np.dot(Cr.astype(params.dtype),
                                             params[Nm+Np+Nf:Nm+Np+Nf+Nr])

    return Pm


@njit
def get_params_combinations(nlparams, shape):
    """given a flattend list (nlparams) of parameter types and unique values
    and the information of how many unique values per parameter type (shape)
    return a list of all possible parameter combinations,
    the same what is returned by numpy meshgrid:

    `params_combinations = np.array(np.meshgrid(*nlparams)).T\
                          .reshape(-1, len(Ntot))`,

    but possible to be njitted.

    :param nlparams: 1d iterable, flattend parameter types and  values
    :param shape: 1d iterable, numbers of unique values per type
    :returns: all possible combinations of parameter types and values
    :rtype: 2d np.ndarray of shape numpy.prod(shape) x len(shape)

    """

    Nparams = len(shape)
    # njittable np.prod
    Ncombinations = 1
    for s in shape:
        Ncombinations *= s

    # get start and end indices of each
    # nonlinear parameter range flattened in nlparams
    cumsum = np.cumsum(shape)
    sep_inds = np.zeros((Nparams, 2), dtype=np.int64)
    sep_inds[:, 1] = cumsum
    sep_inds[1:, 0] = cumsum[:-1]

    # loop over all "sub2ind's" and replace integer index with parameter
    multi_indexes = nbutils.list_multi_indexes(shape)
    params_combinations = np.zeros((Ncombinations, Nparams))
    for r in range(multi_indexes.shape[0]):
        for c in range(multi_indexes.shape[1]):

            prange = nlparams[sep_inds[c, 0]:sep_inds[c, 1]]

            ind = multi_indexes[r, c]

            params_combinations[r, c] = prange[ind]

    return params_combinations
