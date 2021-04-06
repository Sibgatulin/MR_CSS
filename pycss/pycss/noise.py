import numpy as np
from numba import njit, prange
from pycss import css
from pycss.utils import construct_map


@njit
def get_Fisher_matrix(TE_s, pm, Cm, Cp, Cf, Cr):
    J = css.get_Jacobian(TE_s, pm, Cm, Cp, Cf, Cr)
    return np.dot(np.conjugate(J.T), J).real


@njit
def get_CRB_matrix(TE_s, pm, Cm, Cp, Cf, Cr):
    F = get_Fisher_matrix(TE_s, pm, Cm, Cp, Cf, Cr)
    return np.linalg.inv(F)


@njit(parallel=True)
def compute_FIMparams(TE_s, Pm, Cm, Cp, Cf, Cr):

    nTE = len(TE_s)
    nVoxels = Pm.shape[0]
    nParams = Cm.shape[1] + Cp.shape[1] + Cf.shape[1] + Cr.shape[1]

    FIMs = np.zeros((nVoxels, nParams, nParams))
    FIMinvs = np.zeros((nVoxels, nParams, nParams))
    CRLBs = np.zeros((nVoxels, nParams))
    NSAs = np.zeros((nVoxels, nParams))
    Invariants = np.zeros((nVoxels, 3))  # trF, trF^-1, detF

    for i in prange(nVoxels):
        pm = Pm[i, ...].reshape(Pm.shape[1:])

        FIM = get_Fisher_matrix(TE_s, pm, Cm, Cp, Cf, Cr)
        FIMinv = np.linalg.inv(FIM)

        FIMs[i, ...] = FIM
        FIMinvs[i, ...] = FIMinv
        CRLB = np.diag(FIMinv)
        CRLBs[i, ...] = CRLB
        NSAs[i, ...] = nTE / np.diag(FIM) / CRLB
        Invariants[i, ...] = np.array([np.trace(FIM),
                                       np.trace(FIMinv),
                                       np.linalg.det(FIM)])

    return CRLBs, NSAs, Invariants, FIMs, FIMinvs


def mc_css_varpro(Ninr, snr, TE_s, sig, pm0, Cm, Cp, Cf, Cr, tol, itermax):
    Sig = np.tile(sig, [Ninr, 1])
    Sig_noise = css.add_noise(Sig, snr)
    Pm0 = np.tile(pm0, [Ninr, 1, 1])
    Pme, Resnorm, Iterations = css.map_varpro(TE_s, Sig_noise, Pm0,
                                              Cm, Cp, Cf, Cr,
                                              tol, itermax)
    variables = css.extract_variables(Pme, [Cm, Cp, Cf, Cr])
    stats = np.array([[np.mean(v), np.var(v)] for v in variables]).T
    return stats


def compute_FIMmaps(TE_s, Pm, Cm, Cp, Cf, Cr, mask):
    CRLBs, NSAs, Invariants, FIMs, FIMinvs = \
        compute_FIMparams(TE_s, Pm, Cm, Cp, Cf, Cr)
    n = CRLBs.shape[-1]
    return {'CRLBs': np.array([construct_map(CRLBs[:, i], mask)
                                for i in range(n)]),
            'NSAs': np.array([construct_map(NSAs[:, i], mask)
                               for i in range(n)]),
            'trFIM': construct_map(Invariants[:, 0], mask),
            'trInvFIM': construct_map(Invariants[:, 1], mask),
            'detFIM': construct_map(Invariants[:, 2], mask)}
