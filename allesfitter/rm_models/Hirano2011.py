"""
Hirano et al. (2011) Rossiter-McLaughlin model.

This implementation is adapted from EXOZIPPy's exozippy_rossiter.py and
made self-contained for allesfast.

Vectorized rewrite: _compute_m_array and _rm_delta_v now use numpy broadcasting
to eliminate all Python loops, achieving ~10x speedup without external dependencies.
"""

import functools
import numpy as np
from scipy.integrate import simpson as _simpson
from scipy.special import j0 as _scipy_j0
from pytransit import RoadRunnerModel


_TM = RoadRunnerModel("quadratic")


@functools.lru_cache(maxsize=16)
def _gl_nodes_weights(n=64):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _compute_m_array(sigma_array, vsini_kms, u1, u2, zeta_kms, n_gl=64):
    """
    Vectorized M(sigma) computation using numpy broadcasting.

    Replaces the Python loop over sigma with a single matrix operation:
      shape (n_sigma, n_gl) for all intermediate arrays.
    """
    t, w = _gl_nodes_weights(n_gl)           # (n_gl,)
    s = sigma_array                           # (n_sigma,)

    # limb-darkening weight, shape (n_gl,)
    sqrt_1mt2 = np.sqrt(np.maximum(0.0, 1.0 - t * t))
    limb = (1.0 - u1 * (1.0 - sqrt_1mt2) - u2 * (1.0 - sqrt_1mt2) ** 2) / (
        1.0 - u1 / 3.0 - u2 / 6.0
    )

    # macroturbulence exponential, shape (n_sigma, n_gl)
    s2 = (np.pi * zeta_kms) ** 2 * s[:, None] ** 2
    exp_macro = np.exp(-s2 * (1.0 - t[None, :] ** 2)) + np.exp(-s2 * t[None, :] ** 2)

    # Bessel argument and J0, shape (n_sigma, n_gl)
    omega = 2.0 * np.pi * s[:, None] * vsini_kms * t[None, :]
    J0 = _scipy_j0(omega)

    # integrand shape (n_sigma, n_gl), integrate with GL weights
    integrand = limb[None, :] * exp_macro * J0 * t[None, :]
    return integrand @ w                      # (n_sigma,)


def _kepler_eq(m, ecc, thresh=1e-10):
    if ecc < 0.0 or ecc >= 1.0:
        raise ValueError("Eccentricity must satisfy 0 <= ecc < 1.")

    m = np.atleast_1d(np.asarray(m, dtype=float))
    if len(m) == 0:
        return m
    e_anom = m.copy()
    if ecc == 0.0:
        return e_anom

    for _ in range(100):
        f = e_anom - ecc * np.sin(e_anom) - m
        fp = 1.0 - ecc * np.cos(e_anom)
        step = f / fp
        e_anom -= step
        if np.max(np.abs(step)) < thresh:
            break

    return np.mod(e_anom, 2.0 * np.pi)


def _true_anomaly(bjd, tp, period, ecc):
    mean_anom = 2.0 * np.pi * (1.0 + np.mod((bjd - tp) / period, 1.0))
    if ecc <= 0.0:
        return mean_anom
    ecc_anom = _kepler_eq(mean_anom, ecc)
    return 2.0 * np.arctan(np.sqrt((1.0 + ecc) / (1.0 - ecc)) * np.tan(ecc_anom / 2.0))


def _planet_xy(true_anom, ecc, omega, ar, inc, lam):
    r = ar * (1.0 - ecc**2) / (1.0 + ecc * np.cos(true_anom))
    x_old = -r * np.cos(true_anom + omega)
    y_old = -r * np.sin(true_anom + omega) * np.cos(inc)
    cos_lam = np.cos(lam)
    sin_lam = np.sin(lam)
    x = x_old * cos_lam - y_old * sin_lam
    y = x_old * sin_lam + y_old * cos_lam
    z = r * np.sin(true_anom + omega) * np.sin(inc)
    return x, y, z


def _tp_to_t0(tp, period, ecc, omega):
    if (ecc > 0.0) or (omega != 0.5 * np.pi):
        f = 0.5 * np.pi - omega
        e_anom = 2.0 * np.arctan(np.tan(0.5 * f) * np.sqrt((1.0 - ecc) / (1.0 + ecc)))
        return tp + period / (2.0 * np.pi) * (e_anom - ecc * np.sin(e_anom))
    return tp


def _transit_flux(bjd, tp, period, ecc, omega, inc, ar, p, u1, u2):
    t0 = _tp_to_t0(tp, period, ecc, omega)
    _TM.set_data(bjd)
    flux = _TM.evaluate(abs(p), [u1, u2], t0, period, ar, inc, ecc, omega)
    return np.atleast_1d(np.asarray(flux, dtype=float))


def _rm_delta_v(flux, v_sub, cos_thetas, sin_thetas, m_array, sigma_array, zeta_kms, beta_kms, gamma_kms):
    """
    Fully vectorized delta_v computation using numpy broadcasting.

    All time points and sigma values are processed simultaneously:
      shape (n_time, n_sigma) for all intermediate arrays.
    """
    n_time = len(flux)

    # in-transit mask
    f_ = 1.0 - flux                           # (n_time,)
    in_transit = f_ > 0.0

    delta_v = np.zeros(n_time)
    if not np.any(in_transit):
        return delta_v

    # slice to in-transit points only
    f_it  = f_[in_transit]                    # (n_it,)
    vp    = v_sub[in_transit]                 # (n_it,)
    ct    = cos_thetas[in_transit]            # (n_it,)
    st    = sin_thetas[in_transit]            # (n_it,)

    # broadcast shapes: (n_it, n_sigma)
    s  = sigma_array[None, :]                 # (1, n_sigma)
    s2 = s ** 2
    m  = m_array[None, :]                     # (1, n_sigma)

    # broadening envelope, shape (1, n_sigma)
    exp_broad = np.exp(-2.0 * np.pi**2 * beta_kms**2 * s2 - 4.0 * np.pi * gamma_kms * s)
    em = exp_broad * m                        # (1, n_sigma)

    # macroturbulence theta, shape (n_it, n_sigma)
    theta = 0.5 * (
        np.exp(-(np.pi * zeta_kms * ct[:, None]) ** 2 * s2)
        + np.exp(-(np.pi * zeta_kms * st[:, None]) ** 2 * s2)
    )

    # trig terms, shape (n_it, n_sigma)
    two_pi_vp_s = 2.0 * np.pi * vp[:, None] * s
    sin_term = np.sin(two_pi_vp_s)
    cos_term = np.cos(two_pi_vp_s)

    # numerator and denominator integrands
    numer = em * theta * sin_term * s         # (n_it, n_sigma)
    denom = em * (m - f_it[:, None] * theta * cos_term) * s2  # (n_it, n_sigma)

    # integrate over sigma axis using Simpson's rule (same as original)
    num_int = _simpson(numer, x=sigma_array, axis=1)  # (n_it,)
    den_int = _simpson(denom, x=sigma_array, axis=1)  # (n_it,)

    safe = np.abs(den_int) > 1e-30
    delta_v[in_transit] = np.where(safe, f_it / (2.0 * np.pi) * num_int / den_int, 0.0)

    return delta_v


def hirano2011_rm(bjd, tp, period, e, omega, inc, ar, p, u1, u2, vsini, lam, vgamma=1000.0, vzeta=4000.0, vbeta=4000.0):
    """
    Compute RM anomaly in m/s.

    Input units:
    - angles in radians
    - velocities in m/s (vsini, vgamma, vzeta, vbeta)
    - period and times in days

    vbeta : full Gaussian broadening β in m/s (Hirano+2011 Eq. 20)
            β = sqrt(β_thermal² + ξ² + β_IP²)
            where β_thermal = sqrt(2kB·Teff/μ_Fe), ξ = microturbulence,
            β_IP = instrumental profile width = c/(R·2√(2ln2))
            Typical value: ~3–4 km/s for G-type stars observed with R~50000 spectrograph.
    """
    bjd = np.atleast_1d(np.asarray(bjd, dtype=float))

    vsini_kms = vsini / 1e3
    zeta_kms = vzeta / 1e3
    beta_kms = vbeta / 1e3
    gamma_kms = vgamma / 1e3

    true_anom = _true_anomaly(bjd, tp, period, e)
    x, y, z = _planet_xy(true_anom, e, omega, ar, inc, lam)
    v_sub = vsini_kms * x

    flux = _transit_flux(bjd, tp, period, e, omega, inc, ar, p, u1, u2)

    r2 = x**2 + y**2
    cos_thetas = np.sqrt(np.maximum(0.0, 1.0 - r2))
    sin_thetas = np.sqrt(np.maximum(0.0, r2))

    sigma_max = max(5.0 / (vsini_kms + zeta_kms + 0.1), 0.01)
    # For high vsini the Bessel function J0(2πσ·vsini·t) oscillates rapidly,
    # requiring a uniform σ grid dense enough to resolve ~vsini/(vsini+ζ)
    # oscillation cycles.  Linear spacing resolves all cycles equally;
    # 201 points suffice up to vsini ≈ 100 km/s (≲10 cycles in [0, σ_max]).
    n_sigma = max(201, int(40 * vsini_kms * sigma_max) + 1)
    sigma_array = np.linspace(1e-8, sigma_max, n_sigma)
    m_array = _compute_m_array(sigma_array, vsini_kms, u1, u2, zeta_kms)

    delta_v = _rm_delta_v(
        flux, v_sub, cos_thetas, sin_thetas, m_array, sigma_array, zeta_kms, beta_kms, gamma_kms
    )
    delta_v[z < 0] = 0.0

    return -delta_v * 1e3
