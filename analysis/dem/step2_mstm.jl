# Optical properties of the step-2 aggregate classes with MSTMforCAS.jl.
#
# Reads the HDF5 file of step2_generate.py (coordinates in units of the monomer radius a),
# centres each aggregate at the origin, scales it to a = A_UM micrometres, applies one random
# rotation per aggregate (fixed seed),
# and computes the forward amplitudes and efficiencies for incidence along +z.
# The truncation order is chosen per (wavelength, refractive index) by
# contact_truncation_order with eps = TRUNC_EPS, so that touching pairs are resolved.
#
# Usage (CPU; threads parallelise over aggregates):
#   julia -t 20 --project=PATH/TO/MSTMforCAS.jl step2_mstm.jl step2.h5 out.csv [n_max_per_class]

using MSTMforCAS, HDF5, LinearAlgebra, Random, Printf

const A_UM = 0.015                                   # monomer radius [um]
const WAVELENGTHS = [0.453, 0.638, 0.834]            # vacuum wavelength [um], medium = air
const REFINDEX = [1.75 + 0.60im, 1.95 + 0.79im]      # absolute = relative refractive index (air)
const TRUNC_EPS = 1e-2

function random_rotation(rng)
    q, r = qr(randn(rng, 3, 3))
    q = Matrix(q) * Diagonal(sign.(diag(r)))
    return det(q) < 0 ? -q : q                      # proper rotation, shape: (3, 3)
end

function main(h5path, out, nmax = typemax(Int))
    BLAS.set_num_threads(1)                         # parallelism is over aggregates
    classes, xps = String[], Array{Float64,3}[]
    h5open(h5path, "r") do f
        for name in eval(Meta.parse(read_attribute(f, "classes")))   # JSON list of class names
            haskey(f, name) || continue
            xp = read(f[name]["xp"])                 # Julia reads (3, N, n) for h5py (n, N, 3)
            ndims(xp) == 3 || continue
            push!(classes, name)
            push!(xps, xp[:, :, 1:min(size(xp, 3), nmax)])
        end
    end
    jobs = [(c, i) for (c, xp) in zip(classes, xps) for i in 1:size(xp, 3)]
    xpd = Dict(zip(classes, xps))
    orders = Dict{Tuple{Float64,ComplexF64},Int}()
    for wl in WAVELENGTHS, m in REFINDEX
        orders[(wl, m)] = contact_truncation_order(2pi / wl * A_UM, m; eps = TRUNC_EPS)
        @printf("lambda %.3f um, m %s: x = %.4f, truncation order %d\n", wl, m, 2pi / wl * A_UM, orders[(wl, m)])
    end
    rows = Vector{String}(undef, length(jobs))
    done = Threads.Atomic{Int}(0)
    t0 = time()
    Threads.@threads :dynamic for j in eachindex(jobs)
        c, i = jobs[j]
        rot = random_rotation(MersenneTwister(hash((c, i))))
        x = xpd[c][:, :, i]
        x = x .- sum(x, dims = 2) ./ size(x, 2)       # centre at the origin: the phase of
                                                      # S_forward depends on the origin along z
        pos = rot * x .* A_UM                         # shape: (3, N) [um]
        N = size(pos, 2)
        buf = IOBuffer()
        for wl in WAVELENGTHS, m in REFINDEX
            k = 2pi / wl
            res, _ = compute_scattering(pos .* k, fill(A_UM * k, N), m;
                                        truncation_order = orders[(wl, m)], use_fft = false)
            S = res.S_forward
            @printf(buf, "%s,%d,%.3f,%.3f,%.3f,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%d,%d\n",
                    c, i - 1, wl, real(m), imag(m), res.Q_ext, res.Q_abs, res.Q_sca,
                    real(S[1]), imag(S[1]), real(S[2]), imag(S[2]), res.converged, res.truncation_order)
        end
        rows[j] = String(take!(buf))
        n = Threads.atomic_add!(done, 1) + 1
        if n % 50 == 0
            @printf("%d / %d aggregates, %.0f s\n", n, length(jobs), time() - t0)
            flush(stdout)
        end
    end
    open(out, "w") do io
        println(io, "class,idx,wavelength_um,m_re,m_im,Q_ext,Q_abs,Q_sca,S1_re,S1_im,S2_re,S2_im,converged,order")
        foreach(r -> print(io, r), rows)
    end
    @printf("saved %s: %d aggregates, %.0f s\n", out, length(jobs), time() - t0)
end

main(ARGS[1], ARGS[2], length(ARGS) > 2 ? parse(Int, ARGS[3]) : typemax(Int))
