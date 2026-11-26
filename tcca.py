"""
Tunable cluster-cluster aggregation (TCCA) for fractal-like aggregates.

Python/Numba translation of the C++ code "aggregate_gen" by N. Moteki
(https://github.com/nmoteki/aggregate_generator, Doi:10.5281/zenodo.3567236),
which implements the tunable CCA of Filippov et al. (2000),
J. Colloid Interface Sci. 229, 261-273.

Algorithm (as in the C++ code)
------------------------------
1. Small initial clusters are built by tunable sequential (particle-cluster)
   aggregation (SA).
2. Clusters are merged pairwise.  The two centroids are placed at the distance
   Gamma that makes the merged cluster obey the fractal law exactly:

       N = kf * (Rg / a)**Df

   with the Filippov convention Rg**2 = a**2 + <|r - r_c|**2>.
3. For each merge, agg1 is translated to a random point on the sphere of
   radius Gamma (outer loop, at most ``iter1max`` trials).  agg2 is rotated
   about a random axis by a random angle (inner loop, at most ``iter2max``
   trials).  A configuration is accepted when no pair overlaps and at least
   one pair is in contact (surface gap below ``tol * mean_rp``).

Extensions beyond the C++ code
------------------------------
* Arbitrary Np.  The C++ code supports only Np = num_sph_SA * 2**levels.
  Filippov et al. (2000) allow clusters of unequal size, so the final size
  need not be a power of 2.  Here a cluster of N monomers is built by
  merging clusters of floor(N/2) and ceil(N/2) monomers, recursively, until
  N <= n_SA.  For Np = n_SA * 2**L
  this is the C++ hierarchy.  The Filippov formula is applied with float
  arithmetic, so it holds for N1 != N2.
* Polydisperse monomers.  Radii follow the same clipped normal distribution
  as ``ptsa``.  Centroids and Rg are mass weighted (m_i ~ r_i**3) and the
  Filippov self term a**2 becomes r_i**2.  The length scale of the fractal
  law of a cluster is its mass-weighted rms monomer radius
  a_c**2 = sum(m_i r_i**2) / sum(m_i), so that Rg_F**2 = a_c**2 + <|x|**2>_m
  mirrors the C++ a**2 + <r**2>.  This keeps the placement scale-consistent:
  a fixed a = mean_rp makes small SA clusters geometrically infeasible when
  their monomers are larger than the mean.  For rel_std_rp = 0 all formulas
  reduce exactly to the C++ ones.
* Failure handling.  The C++ code loops forever or aborts.  Here a step
  that is geometrically impossible (typical at low Df: the required
  centroid distance exceeds the reach of the two clusters) triggers a
  rebuild of that sub-cluster with new radii.  This is exact rejection
  sampling.  An exhausted random search (typical at high Df) is reported
  as Np_final < Np.
* A cell list accelerates the contact test of large clusters.  The accept /
  reject decision of each trial is identical to the brute-force test, so the
  output is bit-identical for the same RNG seed.

Two minor differences from the C++ code: the initial dimer of each SA
cluster has a random orientation (C++: along x; irrelevant after the random
rotations), and the overlap threshold is a surface gap of 1e-6 * mean_rp
(C++: 0), the same safety margin as ``ptsa``.

Units: lengths in the unit of ``mean_rp`` (um in this project).
"""

import numba
import numpy as np


# ---------------------------------------------------------------------------
# Small helpers (Numba)
# ---------------------------------------------------------------------------
@numba.njit
def _sample_radii(Np, mean_rp, rel_std_rp, rng):
    """Monomer radii: normal distribution clipped at +-2 sigma (same as ptsa).

    Returns rp, shape (Np,), in the unit of mean_rp.
    """
    u1 = rng.random(size=Np)
    u2 = rng.random(size=Np)
    snv = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    for i in range(Np):
        if snv[i] < -2.0:
            snv[i] = -2.0
        elif snv[i] > 2.0:
            snv[i] = 2.0
    return mean_rp * rel_std_rp * snv + mean_rp


@numba.njit
def _rotation_matrix(nx, ny, nz, psi):
    """Rotation by angle psi [rad] about unit axis n (Rodrigues; = Eigen AngleAxisd)."""
    c = np.cos(psi)
    s = np.sin(psi)
    t = 1.0 - c
    R = np.empty((3, 3))  # shape: (3, 3)
    R[0, 0] = c + t * nx * nx
    R[0, 1] = t * nx * ny - s * nz
    R[0, 2] = t * nx * nz + s * ny
    R[1, 0] = t * ny * nx + s * nz
    R[1, 1] = c + t * ny * ny
    R[1, 2] = t * ny * nz - s * nx
    R[2, 0] = t * nz * nx - s * ny
    R[2, 1] = t * nz * ny + s * nx
    R[2, 2] = c + t * nz * nz
    return R


@numba.njit
def _random_axis(rng):
    """Random unit vector, drawn as in the C++ code (theta first, then phi)."""
    theta = np.arccos(2.0 * rng.random() - 1.0)
    phi = 2.0 * np.pi * rng.random()
    return (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
            np.cos(theta))


@numba.njit
def _shift_to_mass_centroid(xp, mp, s, e):
    """Translate xp[s:e] so that its mass centroid is at the origin."""
    M = 0.0
    cx = 0.0
    cy = 0.0
    cz = 0.0
    for i in range(s, e):
        M += mp[i]
        cx += mp[i] * xp[i, 0]
        cy += mp[i] * xp[i, 1]
        cz += mp[i] * xp[i, 2]
    cx /= M
    cy /= M
    cz /= M
    for i in range(s, e):
        xp[i, 0] -= cx
        xp[i, 1] -= cy
        xp[i, 2] -= cz


@numba.njit
def _rg2_filippov(xp, rp, mp, s, e):
    """Mass-weighted Rg**2 with self term r_i**2 (Filippov convention).

    Assumes the mass centroid of xp[s:e] is at the origin.
    Monodisperse limit: a**2 + <|x|**2> as in the C++ code.
    """
    M = 0.0
    acc = 0.0
    for i in range(s, e):
        M += mp[i]
        acc += mp[i] * (xp[i, 0] ** 2 + xp[i, 1] ** 2 + xp[i, 2] ** 2
                        + rp[i] ** 2)
    return acc / M


@numba.njit
def _ac2(rp, mp, s, e):
    """Squared length scale a_c**2 = sum(m r**2)/sum(m) of monomers s..e-1."""
    M = 0.0
    acc = 0.0
    for i in range(s, e):
        M += mp[i]
        acc += mp[i] * rp[i] ** 2
    return acc / M


@numba.njit
def _gamma2(M1, M2, Rg1_2, Rg2_2, N, ac2, kf, Df):
    """Squared centroid distance giving the merged cluster Rg**2 = ac2*(N/kf)**(2/Df).

    Exact two-body identity for mass-weighted Rg:
        Rg12**2 = (M1*Rg1**2 + M2*Rg2**2)/(M1+M2) + M1*M2/(M1+M2)**2 * Gamma**2
    Monodisperse limit (M ~ N): Filippov et al. (2000), C++ aggregate_gen_CCA.
    """
    Mt = M1 + M2
    Rg_t2 = ac2 * (N / kf) ** (2.0 / Df)
    return Mt * Mt / (M1 * M2) * Rg_t2 - Mt / M2 * Rg1_2 - Mt / M1 * Rg2_2


# ---------------------------------------------------------------------------
# Tunable sequential aggregation (initial clusters)
# ---------------------------------------------------------------------------
@numba.njit
def _sa_cluster(xp, rp, mp, s, n, kf, Df, min_sep, max_sep, max_search, rng):
    """Build one SA cluster of n monomers in xp[s:s+n].

    Returns 0 on success, 1 if a placement is geometrically impossible
    (detected before the random search), 2 if max_search is exhausted.

    Placement distance of monomer N (C++ aggregate_gen_SA, generalised to
    masses): previous cluster Rg uses the theoretical value
    a_c*((N-1)/kf)**(1/Df), as in Filippov et al. (2000).
    """
    xp[s, 0] = 0.0
    xp[s, 1] = 0.0
    xp[s, 2] = 0.0
    if n == 1:
        return 0

    # dimer in contact (gap = min_sep), random orientation
    phi = 2.0 * np.pi * rng.random()
    u = 2.0 * rng.random() - 1.0
    q = np.sqrt(1.0 - u * u)
    d = rp[s] + rp[s + 1] + min_sep
    xp[s + 1, 0] = d * q * np.cos(phi)
    xp[s + 1, 1] = d * q * np.sin(phi)
    xp[s + 1, 2] = d * u
    _shift_to_mass_centroid(xp, mp, s, s + 2)

    for N in range(3, n + 1):
        inew = s + N - 1
        M1 = 0.0
        for i in range(s, inew):
            M1 += mp[i]
        m = mp[inew]
        Rg1_2 = _ac2(rp, mp, s, inew) * ((N - 1) / kf) ** (2.0 / Df)
        G2 = _gamma2(M1, m, Rg1_2, rp[inew] ** 2, N, _ac2(rp, mp, s, inew + 1),
                     kf, Df)
        if not (G2 > 0.0):
            return 1
        G = np.sqrt(G2)
        rn = rp[inew]
        # contact with sphere j requires | |x_j| - G | < r_j + rn + max_sep
        reachable = False
        for j in range(s, inew):
            dj = np.sqrt(xp[j, 0] ** 2 + xp[j, 1] ** 2 + xp[j, 2] ** 2)
            if abs(dj - G) < rp[j] + rn + max_sep:
                reachable = True
                break
        if not reachable:
            return 1

        found = False
        for _ in range(max_search):
            phi = 2.0 * np.pi * rng.random()
            u = 2.0 * rng.random() - 1.0
            q = np.sqrt(1.0 - u * u)
            x0 = G * q * np.cos(phi)
            x1 = G * q * np.sin(phi)
            x2 = G * u
            overlap = False
            attached = False
            for j in range(s, inew):
                dx = xp[j, 0] - x0
                dy = xp[j, 1] - x1
                dz = xp[j, 2] - x2
                gap = np.sqrt(dx * dx + dy * dy + dz * dz) - (rp[j] + rn)
                if gap < min_sep:
                    overlap = True
                    break
                if gap < max_sep:
                    attached = True
            if attached and not overlap:
                xp[inew, 0] = x0
                xp[inew, 1] = x1
                xp[inew, 2] = x2
                _shift_to_mass_centroid(xp, mp, s, inew + 1)
                found = True
                break
        if not found:
            return 2
    return 0


# ---------------------------------------------------------------------------
# Cluster-cluster merge
# ---------------------------------------------------------------------------
@numba.njit
def _build_cells(xp, rp, s, e, Rbox, h):
    """Cell list of xp[s:e] on a cubic grid [-Rbox, Rbox]**3 with cell size h.

    Returns (nc, cell_start, cell_idx): cell_start shape (nc**3 + 1,),
    cell_idx shape (e - s,) holding global sphere indices sorted by cell.
    """
    nc = int(np.floor(2.0 * Rbox / h)) + 1
    n = e - s
    cid = np.empty(n, dtype=np.int64)
    for i in range(n):
        ix = min(max(int(np.floor((xp[s + i, 0] + Rbox) / h)), 0), nc - 1)
        iy = min(max(int(np.floor((xp[s + i, 1] + Rbox) / h)), 0), nc - 1)
        iz = min(max(int(np.floor((xp[s + i, 2] + Rbox) / h)), 0), nc - 1)
        cid[i] = (ix * nc + iy) * nc + iz
    cell_start = np.zeros(nc * nc * nc + 1, dtype=np.int64)
    for i in range(n):
        cell_start[cid[i] + 1] += 1
    for c in range(nc * nc * nc):
        cell_start[c + 1] += cell_start[c]
    fill = cell_start[:-1].copy()
    cell_idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        cell_idx[fill[cid[i]]] = s + i
        fill[cid[i]] += 1
    return nc, cell_start, cell_idx


@numba.njit
def _cca_merge(xp, rp, mp, s, mid, e, kf, Df, min_sep, max_sep,
               iter1max, iter2max, use_cells, rng):
    """Merge agg1 = xp[s:mid] and agg2 = xp[mid:e] (each centred).

    Translation of the iter1/iter2 search of C++ aggregate_gen_CCA.  The
    result is written to xp[s:e] with its mass centroid at the origin.
    Returns 0 on success, 1 if contact is geometrically impossible
    (Gamma**2 <= 0, or Gamma beyond the sum of the bounding radii; detected
    before the search), 2 if iter1max * iter2max trials are exhausted.
    """
    N1 = mid - s
    N2 = e - mid
    N = N1 + N2

    # random rotation of agg1 around its centroid (as in the C++ code)
    nx, ny, nz = _random_axis(rng)
    psi = 2.0 * np.pi * rng.random()
    R = _rotation_matrix(nx, ny, nz, psi)
    for i in range(s, mid):
        x0 = xp[i, 0]
        x1 = xp[i, 1]
        x2 = xp[i, 2]
        xp[i, 0] = R[0, 0] * x0 + R[0, 1] * x1 + R[0, 2] * x2
        xp[i, 1] = R[1, 0] * x0 + R[1, 1] * x1 + R[1, 2] * x2
        xp[i, 2] = R[2, 0] * x0 + R[2, 1] * x1 + R[2, 2] * x2

    M1 = 0.0
    for i in range(s, mid):
        M1 += mp[i]
    M2 = 0.0
    for i in range(mid, e):
        M2 += mp[i]
    Rg1_2 = _rg2_filippov(xp, rp, mp, s, mid)
    Rg2_2 = _rg2_filippov(xp, rp, mp, mid, e)
    G2 = _gamma2(M1, M2, Rg1_2, Rg2_2, N, _ac2(rp, mp, s, e), kf, Df)
    if not (G2 > 0.0):
        return 1
    G = np.sqrt(G2)

    # bounding radii about the centroids; contact impossible if too far apart
    Rmax1 = 0.0
    rmax = 0.0
    for i in range(s, mid):
        Rmax1 = max(Rmax1, np.sqrt(xp[i, 0] ** 2 + xp[i, 1] ** 2
                                   + xp[i, 2] ** 2) + rp[i])
        rmax = max(rmax, rp[i])
    Rmax2 = 0.0
    for i in range(mid, e):
        Rmax2 = max(Rmax2, np.sqrt(xp[i, 0] ** 2 + xp[i, 1] ** 2
                                   + xp[i, 2] ** 2) + rp[i])
        rmax = max(rmax, rp[i])
    if G - Rmax1 - Rmax2 >= max_sep:
        return 1

    # cell list of agg1 in its own frame (fixed during the search)
    h = 2.0 * rmax + max_sep
    if use_cells:
        nc, cell_start, cell_idx = _build_cells(xp, rp, s, mid, Rmax1, h)
    else:
        nc = 1
        cell_start = np.zeros(2, dtype=np.int64)
        cell_idx = np.zeros(1, dtype=np.int64)

    p2 = np.empty((N2, 3))  # rotated agg2, shape: (N2, 3)

    for iter1 in range(iter1max):
        # translation of agg1 to a random point on the sphere of radius G
        phi = 2.0 * np.pi * rng.random()
        u = 2.0 * rng.random() - 1.0
        q = np.sqrt(1.0 - u * u)
        c0 = G * q * np.cos(phi)
        c1 = G * q * np.sin(phi)
        c2 = G * u
        nx, ny, nz = _random_axis(rng)

        for iter2 in range(iter2max):
            psi = 2.0 * np.pi * rng.random()
            R = _rotation_matrix(nx, ny, nz, psi)

            overlap = False
            attached = False
            for jj in range(N2):
                j = mid + jj
                p0 = R[0, 0] * xp[j, 0] + R[0, 1] * xp[j, 1] + R[0, 2] * xp[j, 2]
                p1 = R[1, 0] * xp[j, 0] + R[1, 1] * xp[j, 1] + R[1, 2] * xp[j, 2]
                pz = R[2, 0] * xp[j, 0] + R[2, 1] * xp[j, 1] + R[2, 2] * xp[j, 2]
                p2[jj, 0] = p0
                p2[jj, 1] = p1
                p2[jj, 2] = pz
                # position relative to agg1 centroid
                y0 = p0 - c0
                y1 = p1 - c1
                y2 = pz - c2
                if (np.sqrt(y0 * y0 + y1 * y1 + y2 * y2) - rp[j] - Rmax1
                        >= max_sep):
                    continue  # no pair within reach (exact)
                if use_cells:
                    ix = int(np.floor((y0 + Rmax1) / h))
                    iy = int(np.floor((y1 + Rmax1) / h))
                    iz = int(np.floor((y2 + Rmax1) / h))
                    for cx in range(max(ix - 1, 0), min(ix + 1, nc - 1) + 1):
                        for cy in range(max(iy - 1, 0), min(iy + 1, nc - 1) + 1):
                            for cz in range(max(iz - 1, 0),
                                            min(iz + 1, nc - 1) + 1):
                                c = (cx * nc + cy) * nc + cz
                                for k in range(cell_start[c], cell_start[c + 1]):
                                    i = cell_idx[k]
                                    dx = xp[i, 0] - y0
                                    dy = xp[i, 1] - y1
                                    dz = xp[i, 2] - y2
                                    gap = (np.sqrt(dx * dx + dy * dy + dz * dz)
                                           - (rp[i] + rp[j]))
                                    if gap < min_sep:
                                        overlap = True
                                        break
                                    if gap < max_sep:
                                        attached = True
                                if overlap:
                                    break
                            if overlap:
                                break
                        if overlap:
                            break
                else:
                    for i in range(s, mid):
                        dx = xp[i, 0] - y0
                        dy = xp[i, 1] - y1
                        dz = xp[i, 2] - y2
                        gap = (np.sqrt(dx * dx + dy * dy + dz * dz)
                               - (rp[i] + rp[j]))
                        if gap < min_sep:
                            overlap = True
                            break
                        if gap < max_sep:
                            attached = True
                if overlap:
                    break

            if attached and not overlap:
                for i in range(s, mid):
                    xp[i, 0] += c0
                    xp[i, 1] += c1
                    xp[i, 2] += c2
                for jj in range(N2):
                    xp[mid + jj, 0] = p2[jj, 0]
                    xp[mid + jj, 1] = p2[jj, 1]
                    xp[mid + jj, 2] = p2[jj, 2]
                _shift_to_mass_centroid(xp, mp, s, e)
                return 0
    return 2


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@numba.njit
def _tcca_kernel(Np, mean_rp, rel_std_rp, Df, kf, tol, ops, sub_start,
                 max_search, iter1max, iter2max, max_rebuild, use_cells, rng):
    """Run the full TCCA schedule. Returns (Np_final, xp, rp).

    ops: shape (n_ops, 4), rows (type, start, mid, end) in post-order;
         type 0 = SA cluster of monomers start..end-1, type 1 = merge of
         start..mid-1 with mid..end-1.
    sub_start: shape (n_ops,), index of the first op of each op's subtree.

    When an op fails because contact is geometrically impossible (code 1),
    its whole subtree is rebuilt from scratch, radii included, at most
    max_rebuild times.  Redrawing the whole subtree is a rejection step, so
    the accepted aggregates follow the same distribution as restarting the
    whole aggregate after every such failure.  An exhausted search
    (code 2) ends the run with Np_final < Np, as in ptsa.
    """
    rp = _sample_radii(Np, mean_rp, rel_std_rp, rng)  # shape: (Np,)
    mp = (4.0 * np.pi / 3.0) * rp ** 3                # shape: (Np,), density = 1
    xp = np.zeros((Np, 3))                            # shape: (Np, 3)
    min_sep = mean_rp * 1e-6
    max_sep = mean_rp * tol
    n_ops = ops.shape[0]
    n_rebuild = np.zeros(n_ops, dtype=np.int64)

    Np_done = 0
    iop = 0
    while iop < n_ops:
        s = ops[iop, 1]
        mid = ops[iop, 2]
        e = ops[iop, 3]
        if ops[iop, 0] == 0:
            code = _sa_cluster(xp, rp, mp, s, e - s, kf, Df, min_sep, max_sep,
                               max_search, rng)
        else:
            code = _cca_merge(xp, rp, mp, s, mid, e, kf, Df, min_sep, max_sep,
                              iter1max, iter2max, use_cells, rng)
        if code == 0:
            Np_done = max(Np_done, e - s)
            iop += 1
            continue
        if code == 2:
            return Np_done, xp, rp
        # code 1: rebuild the subtree of this op with new radii
        n_rebuild[iop] += 1
        if n_rebuild[iop] > max_rebuild:
            return Np_done, xp, rp
        rp_new = _sample_radii(e - s, mean_rp, rel_std_rp, rng)
        for i in range(s, e):
            rp[i] = rp_new[i - s]
            mp[i] = (4.0 * np.pi / 3.0) * rp[i] ** 3
        for j in range(sub_start[iop], iop):
            n_rebuild[j] = 0
        iop = sub_start[iop]

    return Np_done, xp, rp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def make_schedule(Np, n_SA):
    """Binary-splitting hierarchy for Np monomers.

    Returns
    -------
    ops       : int64 array, shape (n_ops, 4), rows (type, start, mid, end)
                in post-order; type 0 = SA cluster (mid unused, = end),
                type 1 = merge.  SA clusters have at most n_SA monomers.
    sub_start : int64 array, shape (n_ops,), first op of each op's subtree
    """
    if Np < 1 or n_SA < 1:
        raise ValueError("Np and n_SA must be >= 1")
    ops, sub_start = [], []

    def _split(s, n):
        first = len(ops)
        if n <= n_SA:
            ops.append((0, s, s + n, s + n))
            sub_start.append(first)
            return
        n1 = n // 2
        _split(s, n1)
        _split(s + n1, n - n1)
        ops.append((1, s, s + n1, s + n))
        sub_start.append(first)

    _split(0, Np)
    return (np.array(ops, dtype=np.int64).reshape(-1, 4),
            np.array(sub_start, dtype=np.int64))


def aggregate_properties(xp, rp, density=1.0):
    """Rg, Rve, porosity with the same definitions as ``ptsa``.

    Rg uses mass weights and the solid-sphere self term (3/5) r_i**2.
    Returns (V, Rve, Rg, eps_agg) in the unit of rp (volume: unit**3).
    """
    mp = (4 * np.pi / 3) * density * rp ** 3  # shape: (Np,)
    xc = xp - np.sum(mp[:, None] * xp, axis=0) / np.sum(mp)  # shape: (Np, 3)
    Rg = np.sqrt(np.sum(mp * (np.sum(xc ** 2, axis=1) + 3 / 5 * rp ** 2))
                 / np.sum(mp))
    Re = np.sqrt(5 / 3) * Rg
    Vagg = 4 * np.pi / 3 * Re ** 3
    V = np.sum(mp) / density
    Rve = np.cbrt(3 * V / (4 * np.pi))
    eps_agg = 1 - V / Vagg
    return V, Rve, Rg, eps_agg


def tcca(Np, mean_rp, rel_std_rp, Df, k, n_SA=4, tol=1e-2,
         max_search=100_000_000, iter1max=1000, iter2max=10000,
         max_rebuild=100, rng=None, use_cells=True):
    """
    Tunable cluster-cluster aggregation (Filippov et al. 2000) of Np monomers.

    Parameters
    ----------
    Np         : int    number of monomers (any value >= 1)
    mean_rp    : float  mean monomer radius [um]; also sets the contact
                        window.  The fractal law N = k*(Rg_F/a_c)**Df uses the
                        mass-weighted rms radius a_c (= mean_rp if
                        monodisperse)
    rel_std_rp : float  relative std of monomer radius [-] (0: monodisperse)
    Df         : float  fractal dimension [-]
    k          : float  fractal prefactor [-]
    n_SA       : int    max size of the initial SA clusters (C++ num_sph_SA)
    tol        : float  contact window: a pair is in contact when its surface
                        gap is below tol*mean_rp [-] (C++ tol / a)
    max_search : int    max trials per monomer in SA
    iter1max   : int    max random translations per merge (C++ iter1max)
    iter2max   : int    max random rotations per translation (C++ iter2max)
    max_rebuild: int    max rebuilds of a sub-cluster whose next step is
                        geometrically impossible (typical at low Df)
    rng        : numpy.random.Generator
    use_cells  : bool   cell-list acceleration (same result as brute force)

    Returns
    -------
    Same tuple as ``ptsa``: (Np_final, xp, rp, V, Rve, Rg, eps_agg).
    xp: shape (Np, 3) [um], rp: shape (Np,) [um].  Np_final < Np on failure
    (size of the largest cluster completed), and V, Rve, Rg, eps_agg are 0.
    Rg and eps_agg use the ptsa definitions (mass weights, (3/5) r**2 term).
    """
    if rng is None:
        rng = np.random.default_rng()
    ops, sub_start = make_schedule(int(Np), int(n_SA))
    Np_final, xp, rp = _tcca_kernel(
        int(Np), float(mean_rp), float(rel_std_rp), float(Df), float(k),
        float(tol), ops, sub_start, int(max_search), int(iter1max),
        int(iter2max), int(max_rebuild), bool(use_cells), rng)
    if Np_final == Np:
        V, Rve, Rg, eps_agg = aggregate_properties(xp, rp)
    else:
        V = Rve = Rg = eps_agg = 0.0
    return Np_final, xp, rp, V, Rve, Rg, eps_agg


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
@numba.njit
def _pair_scan(xp, rp, max_sep):
    """Min surface gap, per-sphere min gap, and contact-graph component count."""
    n = xp.shape[0]
    parent = np.arange(n)
    min_gap_each = np.full(n, np.inf)
    min_gap = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            dx = xp[i, 0] - xp[j, 0]
            dy = xp[i, 1] - xp[j, 1]
            dz = xp[i, 2] - xp[j, 2]
            gap = np.sqrt(dx * dx + dy * dy + dz * dz) - (rp[i] + rp[j])
            min_gap = min(min_gap, gap)
            min_gap_each[i] = min(min_gap_each[i], gap)
            min_gap_each[j] = min(min_gap_each[j], gap)
            if gap < max_sep:
                ri = i
                while parent[ri] != ri:
                    ri = parent[ri]
                rj = j
                while parent[rj] != rj:
                    rj = parent[rj]
                if ri != rj:
                    parent[ri] = rj
    n_comp = 0
    for i in range(n):
        if parent[i] == i:
            n_comp += 1
    return min_gap, min_gap_each, n_comp


def check_geometry(xp, rp, max_sep):
    """Geometry check of an aggregate (any generator), O(Np**2).

    Returns dict with min_gap (min surface gap over all pairs), max_nn_gap
    (largest nearest-neighbour gap over spheres), n_components (connected
    components of the contact graph with gap < max_sep), and ok
    (min_gap > 0 and n_components == 1).
    """
    min_gap, gap_each, n_comp = _pair_scan(np.ascontiguousarray(xp, dtype=np.float64),
                                           np.ascontiguousarray(rp, dtype=np.float64),
                                           float(max_sep))
    return {"min_gap": float(min_gap), "max_nn_gap": float(np.max(gap_each)),
            "n_components": int(n_comp),
            "ok": bool(min_gap > 0.0 and n_comp == 1)}


def fractal_prefactor_filippov(xp, rp, Df, a=None):
    """Effective prefactor k_eff = N / (Rg_F / a)**Df of a whole aggregate.

    Rg_F: mass-weighted Rg with self term r_i**2 (the convention of the tcca
    placement law).  a: length scale; default is the mass-weighted rms radius
    a_c used by tcca.  For tcca with a=None, k_eff equals k up to rounding,
    because the last merge enforces the law.  Applied to ptsa output it is a
    diagnostic only: ptsa uses a count-based law with the new monomer's
    radius, so agreement with k is an empirical result.
    """
    mp = rp ** 3  # shape: (Np,)
    if a is None:
        a = np.sqrt(np.sum(mp * rp ** 2) / np.sum(mp))
    xc = xp - np.sum(mp[:, None] * xp, axis=0) / np.sum(mp)  # shape: (Np, 3)
    RgF = np.sqrt(np.sum(mp * (np.sum(xc ** 2, axis=1) + rp ** 2)) / np.sum(mp))
    return xp.shape[0] / (RgF / a) ** Df


@numba.njit
def _pair_histogram(xp, edges):
    """Counts of monomer pairs (i < j) per centre-distance bin, shape (len(edges)-1,)."""
    n = xp.shape[0]
    nb = edges.shape[0] - 1
    counts = np.zeros(nb, dtype=np.int64)
    lo = edges[0]
    hi = edges[nb]
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((xp[i, 0] - xp[j, 0]) ** 2 + (xp[i, 1] - xp[j, 1]) ** 2
                        + (xp[i, 2] - xp[j, 2]) ** 2)
            if d < lo or d >= hi:
                continue
            k = np.searchsorted(edges, d, side="right") - 1
            counts[k] += 1
    return counts


def pair_correlation(xp, edges):
    """Two-point correlation function C(r) of Filippov et al. (2000), Eq. [11].

    C(r) = n(r) / (N * V_shell(r)), where n(r) is the number of monomer pairs
    whose centre distance lies in the bin and V_shell = 4*pi/3*(r2**3 - r1**3).
    Normalisation: integral of C(r) 4 pi r**2 dr over all r = (N - 1)/2 (Eq. [12]).

    xp    : shape (N, 3), monomer centres
    edges : shape (n_bins + 1,), bin edges in the unit of xp (ascending)
    Returns C, shape (n_bins,), in the unit of xp**-3.
    """
    edges = np.ascontiguousarray(edges, dtype=np.float64)
    counts = _pair_histogram(np.ascontiguousarray(xp, dtype=np.float64), edges)
    shell = 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
    return counts / (xp.shape[0] * shell)
