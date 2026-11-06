# stage2a_breaktime.jl
#
# STAGE 2a (exact many-body, no QMC): quantum break time of Gross-Pitaevskii mean field
# in small Bose-Hubbard rings, across regular, dynamically unstable and chaotic regimes.
#
# For an initial GP product state Ψ_GP(c0), the exact state ψ(t) = e^{-iHt}Ψ_GP(c0) is
# compared with the time-dependent GP state Ψ_GP(c(t)) at fixed g = u(N−1)/J (the
# mean-field limit N → ∞ at fixed g). Observables:
#   F_GP(t) = |⟨Ψ_GP(t)|ψ(t)⟩|²   (mean-field fidelity, = average-phase bound of the
#                                  GP-guided QMC, see Stage 1)
#   depletion(t) = 1 − λ_max(ρ)/N (exact one-body density matrix)
#   site densities, exact vs GP.
# Break time t_b(N): first time F_GP < 1/2.
#
# Cases (trimer M = 3 unless noted):
#   vortex  : c_j = e^{2πij/M}/√M, GP-stationary superflow; Bogoliubov-stable for g < g_c,
#             dynamically unstable above (rate from linear stability, computed here).
#   imb     : c ∝ (√0.6, √0.3, √0.1), Josephson-type; regular for small g, chaotic large g.
#   M4_pi   : 4-site ring, staggered (k = π) condensate.
#
# Exact propagation: sparse Krylov (KrylovKit.exponentiate, Lanczos) in chunks of DT_OUT.
# Submit: cd ~/claude/importance/runs && sbatch ~/claude/slurm_claude/submit.sl stage2a_breaktime.jl

using Rimu
using LinearAlgebra
using SparseArrays
using KrylovKit
using Printf
include(joinpath(@__DIR__, "GPGuidedDynamics.jl"))
using .GPGuidedDynamics

const QUICK = get(ENV, "STAGE2_QUICK", "0") == "1"
const OUTDIR = QUICK ? mktempdir() :
    joinpath(homedir(), "claude", "importance", "data", "stage2a_breaktime")
mkpath(OUTDIR)
const J = 1.0
const DT_OUT = 0.05
const TMAX = QUICK ? 1.0 : 60.0

vortex(M; w=1) = ComplexF64[cis(2π * w * j / M) for j in 0:M-1] ./ sqrt(M)
const CASES = QUICK ? [("vortex", 3, vortex(3), [3.0], [6, 12])] : [
    ("vortex", 3, vortex(3), [1.0, 2.0, 3.0, 4.0, 6.0], [10, 20, 40, 80, 160]),
    ("imb", 3, normalize(ComplexF64[sqrt(0.6), sqrt(0.3), sqrt(0.1)]), [4.0, 8.0, 14.0, 18.0],
     [10, 20, 40, 80, 160]),
    ("M4pi", 4, vortex(4; w=2), [1.0, 3.0], [8, 16, 24, 32]),
]

# ---- GP amplitudes in log space (no overflow for large N) ----------------------------
function gp_vector_log(db::DenseBasis, c, logfact)
    lc = log.(c)
    N = db.N
    out = Vector{ComplexF64}(undef, length(db.basis))
    for (k, o) in enumerate(db.onrs)
        s = 0.5 * (logfact[N+1] - sum(logfact[n+1] for n in o))
        z = complex(s)
        for (m, n) in enumerate(o)
            n > 0 && (z += n * lc[m])
        end
        out[k] = exp(z)
    end
    return out
end

# ---- linear (Bogoliubov) stability of a GP stationary state --------------------------
"Max growth rate of small perturbations around a GP stationary state c (0 if stable)."
function bogoliubov_rate(c, h1, g)
    c = ComplexF64.(c) ./ norm(c)
    μ = real(dot(c, h1 * c)) + g * sum(abs2.(c) .^ 2)
    @assert norm(h1 * c .+ g .* abs2.(c) .* c .- μ .* c) < 1e-10 "not stationary"
    A = h1 - μ * I + 2g * Diagonal(abs2.(c))
    B = g * Diagonal(c .^ 2)
    L = [A B; -conj(B) -conj(A)]
    return maximum(imag, eigvals(Matrix(L)))
end

function run_case(name, M, c0, g, N)
    u = g / (N - 1)
    addr = near_uniform(BoseFS{N,M})
    H = HubbardReal1D(addr; u, t=J)
    bsr = BasisSetRepresentation(H)
    Hs = sparse(bsr.sparse_matrix)
    db = DenseBasis(bsr.basis)
    ops = one_body_ops(db)
    logfact = cumsum([0.0; log.(1:N)])
    h1 = ring_hopping(M, J)
    gp = GPOrbital(c0, h1, u, N; maxdt=2e-3)
    ψ = gp_vector_log(db, gp.c, logfact)
    @assert abs(norm(ψ) - 1) < 1e-8 "GP vector not normalised: $(norm(ψ))"
    E0 = real(dot(ψ, Hs * ψ))
    rows = NTuple{11,Float64}[]
    t = 0.0
    nchunks = round(Int, TMAX / DT_OUT)
    for k in 0:nchunks
        if k > 0
            ψ, info = exponentiate(Hs, -im * DT_OUT, ψ; ishermitian=true, tol=1e-12, krylovdim=40)
            info.converged == 0 && @warn "Krylov not converged" name g N t
            t = k * DT_OUT
            advance_gp!(gp, t)
        end
        Ψ = gp_vector_log(db, gp.c, logfact)
        ov = dot(Ψ, ψ)
        fgp = abs2(ov)
        ρ = obdm(ψ, db, ops)
        dep = 1 - eigmax(Hermitian(ρ)) / N
        n1 = real(ρ[1, 1])
        # exact-state average phase of the sampled vector in three representations
        aψ = abs.(ψ)
        sign_unguided = abs(sum(ψ)) / sum(aψ)
        sign_a0 = abs(sum(conj.(Ψ) ./ abs.(Ψ) .* ψ)) / sum(aψ)     # ψ̃ = e^{-iθ_GP} ψ
        sign_a1 = abs(ov) / sum(abs.(Ψ) .* aψ)                      # ψ̃ = conj(Ψ_GP) ψ
        push!(rows, (t, fgp, dep, n1, N * abs2(gp.c[1]), real(dot(ψ, Hs * ψ)) - E0,
                     gp_energy(gp) - E0, norm(ψ), sign_unguided, sign_a0, sign_a1))
    end
    file = joinpath(OUTDIR, @sprintf("%s_g%.1f_N%d.csv", name, g, N))
    open(file, "w") do io
        println(io, "time,fgp,depletion,n1,n1_gp,dE_exact,dE_gp_minus_exact0,norm,sign_unguided,sign_a0,sign_a1")
        for r in rows
            println(io, join(r, ","))
        end
    end
    i = findfirst(r -> r[2] < 0.5, rows)
    tb = i === nothing ? NaN : rows[i][1]
    j = findfirst(r -> r[3] > 0.1, rows)
    td = j === nothing ? NaN : rows[j][1]
    late = filter(r -> r[1] >= TMAX / 2, rows)
    fl = sum(r[2] for r in late) / length(late)
    return tb, td, fl, length(db.basis)
end

open(joinpath(OUTDIR, "summary.csv"), "w") do io
    println(io, "case,M,g,N,dim,t_break,t_depl10,fgp_late_mean,lyapunov_gp,bogoliubov_rate")
    for (name, M, c0, gs, Ns) in CASES
        h1 = ring_hopping(M, J)
        for g in gs
            λ, _, _ = lyapunov_gp(c0 .+ (name == "imb" ? 0 : 1e-6), h1, g, 2; tmax=QUICK ? 5.0 : 300.0, dt=2e-3)
            rate = name == "imb" ? NaN : bogoliubov_rate(c0, h1, g)
            for N in Ns
                wall = @elapsed tb, td, fl, dim = run_case(name, M, c0, g, N)
                @printf(io, "%s,%d,%.2f,%d,%d,%.4f,%.4f,%.6f,%.6f,%.6f\n", name, M, g, N, dim,
                        tb, td, fl, λ, rate)
                flush(io)
                @printf("%-7s M=%d g=%5.2f N=%4d dim=%6d  t_F=%7.3f  t_dep=%7.3f  <F>late=%.3f  λ_GP=%.4f  bogo=%.4f  (%.1f s)\n",
                        name, M, g, N, dim, tb, td, fl, λ, rate, wall)
                flush(stdout)
            end
        end
    end
end
println("wrote ", OUTDIR)
