# stage2c_threshold.jl
#
# STAGE 2c (exact many-body): mean-field fidelity across the dynamical instability of a
# superflow (vortex) state in the Bose-Hubbard trimer, with finite-N scaling.
#
# Initial state: GP vortex c_j = e^{2πij/3}/√3 (a single Fock state in momentum space,
# all N bosons at k0 = 2π/3); it is GP-stationary for all g = u(N−1)/J.
# Bogoliubov (two-mode, pair channel k0 ± q = {0, −2π/3}):
#     A = ε̄ + g/3,  B = g/3,  ε̄ = [ε(k0+q)+ε(k0−q)]/2 − ε(k0) = −3J/2,
#     Ω² = A² − B²  (Ω² < 0 ⇔ g > g_c = 9/4: dynamical instability, rate κ = |Ω|),
#     F_B(t) = 1 / (1 + (B/Ω)² sin²(Ωt))   (sin(Ωt)/Ω → sinh(κt)/κ when unstable),
#     time average  F̄_B = Ω/|A|  →  0  as  √(g_c − g).
# Exact: e^{-iHt}Ψ_GP by sparse Lanczos (KrylovKit) for N up to 320 (dim 51681).
# Output per (g, N): time, F_exact, F_bogoliubov, depletion; summary: time averages.
#
# Submit: cd ~/claude/importance/runs && sbatch ~/claude/slurm_claude/submit.sl stage2c_threshold.jl

using Rimu
using LinearAlgebra
using SparseArrays
using KrylovKit
using Printf
include(joinpath(@__DIR__, "GPGuidedDynamics.jl"))
using .GPGuidedDynamics

const QUICK = get(ENV, "STAGE2_QUICK", "0") == "1"
const OUTDIR = QUICK ? mktempdir() :
    joinpath(homedir(), "claude", "importance", "data", "stage2c_threshold")
mkpath(OUTDIR)
const J = 1.0
const M = 3
const DT_OUT = 0.05
const TMAX = QUICK ? 1.0 : 80.0
const GS = QUICK ? [2.0] : [1.0, 1.5, 1.8, 2.0, 2.1, 2.2, 2.25, 2.3, 2.4, 2.6, 3.0, 4.0]
const NS = QUICK ? [6] : [20, 40, 80, 160, 320]

const C0 = ComplexF64[cis(2π * j / M) for j in 0:M-1] ./ sqrt(M)
εk(k) = -2J * cos(k)
const K0 = 2π / M
const EBAR = (εk(K0 + 2π / M) + εk(K0 - 2π / M)) / 2 - εk(K0)

function bogoliubov_F(g, t)
    A = EBAR + g / 3
    B = g / 3
    Ω2 = A^2 - B^2
    s = Ω2 > 0 ? sin(sqrt(Ω2) * t) / sqrt(Ω2) : Ω2 < 0 ? sinh(sqrt(-Ω2) * t) / sqrt(-Ω2) : t
    return 1 / (1 + B^2 * s^2)
end
bogoliubov_Fbar(g) = (A = EBAR + g / 3; Ω2 = A^2 - (g / 3)^2; Ω2 > 0 ? sqrt(Ω2) / abs(A) : 0.0)

function gp_vector_log(db, c, logfact)
    lc = log.(c)
    out = Vector{ComplexF64}(undef, length(db.basis))
    for (k, o) in enumerate(db.onrs)
        z = complex(0.5 * (logfact[db.N+1] - sum(logfact[n+1] for n in o)))
        for (m, n) in enumerate(o)
            n > 0 && (z += n * lc[m])
        end
        out[k] = exp(z)
    end
    return out
end

function run_case(g, N)
    u = g / (N - 1)
    H = HubbardReal1D(near_uniform(BoseFS{N,M}); u, t=J)
    bsr = BasisSetRepresentation(H)
    Hs = sparse(bsr.sparse_matrix)
    db = DenseBasis(bsr.basis)
    ops = one_body_ops(db)
    logfact = cumsum([0.0; log.(1:N)])
    gp = GPOrbital(C0, ring_hopping(M, J), u, N; maxdt=2e-3)
    ψ = gp_vector_log(db, gp.c, logfact)
    rows = NTuple{5,Float64}[]
    for k in 0:round(Int, TMAX / DT_OUT)
        if k > 0
            ψ, info = exponentiate(Hs, -im * DT_OUT, ψ; ishermitian=true, tol=1e-12, krylovdim=40)
            advance_gp!(gp, k * DT_OUT)
        end
        t = k * DT_OUT
        Ψ = gp_vector_log(db, gp.c, logfact)
        F = abs2(dot(Ψ, ψ))
        dep = 1 - eigmax(Hermitian(obdm(ψ, db, ops))) / N
        push!(rows, (t, F, bogoliubov_F(g, t), dep, norm(ψ)))
    end
    open(joinpath(OUTDIR, @sprintf("vortex_g%.2f_N%d.csv", g, N)), "w") do io
        println(io, "time,F_exact,F_bogoliubov,depletion,norm")
        foreach(r -> println(io, join(r, ",")), rows)
    end
    Fbar = sum(r[2] for r in rows) / length(rows)
    FbarB = sum(r[3] for r in rows) / length(rows)
    j = findfirst(r -> r[4] > 0.1, rows)
    return Fbar, FbarB, j === nothing ? NaN : rows[j][1], maximum(r[4] for r in rows), length(db.basis)
end

open(joinpath(OUTDIR, "summary.csv"), "w") do io
    println(io, "g,N,dim,Fbar_exact,Fbar_bogoliubov_window,Fbar_bogoliubov_inf,t_depl10,max_depletion")
    for g in GS, N in NS
        wall = @elapsed Fbar, FbarB, td, dmax, dim = run_case(g, N)
        @printf(io, "%.3f,%d,%d,%.6f,%.6f,%.6f,%.4f,%.6f\n", g, N, dim, Fbar, FbarB,
                bogoliubov_Fbar(g), td, dmax)
        flush(io)
        @printf("g=%5.2f N=%4d dim=%6d  F̄=%.4f  F̄_B(window)=%.4f  F̄_B(∞)=%.4f  t_dep=%7.3f  max dep=%.4f (%.1f s)\n",
                g, N, dim, Fbar, FbarB, bogoliubov_Fbar(g), td, dmax, wall)
        flush(stdout)
    end
end
println("wrote ", OUTDIR)
