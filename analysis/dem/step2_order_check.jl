# Convergence of the step-2 optics in the VSWF truncation order.
# Recomputes the first n aggregates of selected classes at the base order of step2_mstm.jl
# (contact_truncation_order, eps = 1e-2) and at base + 2 and base + 3, for one wavelength and
# refractive index, with the same centring and random rotation as step2_mstm.jl.
#
# Usage: julia -t 20 --project=PATH/TO/MSTMforCAS.jl step2_order_check.jl step2.h5 out.csv n class...

using MSTMforCAS, HDF5, LinearAlgebra, Random, Printf

const A_UM = 0.015
const WL = 0.638
const M = 1.95 + 0.79im

function random_rotation(rng)
    q, r = qr(randn(rng, 3, 3))
    q = Matrix(q) * Diagonal(sign.(diag(r)))
    return det(q) < 0 ? -q : q
end

function main(h5path, out, n, classes...)
    BLAS.set_num_threads(1)
    k = 2pi / WL
    base = contact_truncation_order(A_UM * k, M; eps = 1e-2)
    jobs = Tuple{String,Int,Matrix{Float64}}[]
    h5open(h5path, "r") do f
        for c in classes
            xp = read(f[c]["xp"])
            for i in 1:min(n, size(xp, 3))
                push!(jobs, (c, i, xp[:, :, i]))
            end
        end
    end
    rows = Vector{String}(undef, length(jobs))
    Threads.@threads :dynamic for j in eachindex(jobs)
        c, i, x = jobs[j]
        x = x .- sum(x, dims = 2) ./ size(x, 2)
        pos = random_rotation(MersenneTwister(hash((c, i)))) * x .* A_UM
        buf = IOBuffer()
        for order in (base, base + 2, base + 3)
            res, _ = compute_scattering(pos .* k, fill(A_UM * k, size(pos, 2)), M; truncation_order = order)
            S = res.S_forward
            @printf(buf, "%s,%d,%d,%.10e,%.10e,%.10e,%.10e,%.10e\n", c, i - 1, order, res.Q_ext,
                    res.Q_abs, res.Q_sca, real(S[1]), imag(S[1]))
        end
        rows[j] = String(take!(buf))
    end
    open(out, "w") do io
        println(io, "class,idx,order,Q_ext,Q_abs,Q_sca,S1_re,S1_im")
        foreach(r -> print(io, r), rows)
    end
    println("saved $out, base order $base")
end

main(ARGS[1], ARGS[2], parse(Int, ARGS[3]), ARGS[4:end]...)
