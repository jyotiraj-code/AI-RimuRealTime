# stage1_verify.jl
#
# STAGE 1: verify time-dependent GP importance sampling (GPGuidedDynamics.jl) in
# RimuRealTime against exact diagonalisation.
#
# System : Bose-Hubbard ring HubbardReal1D, M = 4, N = 8 (dim 165), u = 1, t = 1.
# Initial: the GP product state Ψ_GP(c0) with a generic complex orbital c0.
# Guide  : φ = conj(Ψ_GP(t)), c(t) from the discrete time-dependent GP equation.
# Runs   : unguided / guided ExactEvolution (Krylov), guided without ∂tφ/φ term
#          (ablation), unguided / guided deterministic PEC, and unguided / guided
#          semistochastic PEC at W = 100 and 1000 walkers (3 seeds each).
# Checks : back-transformed ψ = ψ̃/φ vs exact e^{-iHt}Ψ_GP(0) (fidelity, densities,
#          condensate fraction), and the bound sign(ψ̃) ≥ sqrt(F_GP).
#
# Submit : cd ~/claude/importance/runs && sbatch ~/claude/slurm_claude/submit.sl stage1_verify.jl
# Quick local API check: STAGE1_QUICK=1 julia --project=devrimu scripts/stage1_verify.jl

using Rimu
using RimuRealTime
using Gutzwiller
using LinearAlgebra
using DataFrames
using Printf

const QUICK = get(ENV, "STAGE1_QUICK", "0") == "1"
include(joinpath(@__DIR__, "GPGuidedDynamics.jl"))
using .GPGuidedDynamics

const OUTDIR = QUICK ? mktempdir() :
    joinpath(homedir(), "claude", "importance", "data", "stage1_verify")
mkpath(OUTDIR)

# ---- parameters ----------------------------------------------------------------------
const M = QUICK ? 3 : 4
const N = QUICK ? 3 : 8
const U = 1.0
const J = 1.0
const DT = QUICK ? 0.01 : 0.005
const TMAX = QUICK ? 0.1 : 4.0
const NSTEPS = round(Int, TMAX / DT)
const EVERY = QUICK ? 1 : 2
const WALKERS = QUICK ? (50,) : (100, 1000)
const SEEDS = QUICK ? (1,) : (1, 2, 3)
const C0 = let c = ComplexF64[1.0, 0.8cis(0.3), 0.6cis(-0.5), 0.9cis(1.0)][1:M]
    c ./ norm(c)
end

println("STAGE 1 verification (quick = $QUICK)")
for p in (Rimu, RimuRealTime, Gutzwiller)
    println("  ", rpad(string(p), 14), pkgversion(p))
end
println("  Julia ", VERSION, ", threads = ", Threads.nthreads())

addr = near_uniform(BoseFS{N,M})
H = HubbardReal1D(addr; u=U, t=J)
bsr = BasisSetRepresentation(H)
Hd = Matrix(bsr.sparse_matrix)
db = DenseBasis(bsr.basis)
println("  H = ", H, ", dim = ", length(db.basis))
@assert ishermitian(Hd)
E, V = eigen(Hermitian(Hd))

h1 = ring_hopping(M, J)
E_rimu, E_gp, nΨ = check_gp_consistency(H, h1, U, db, Hd)
@printf("  GP consistency: <Ψ|H|Ψ> = %.12f, GP functional = %.12f, |Ψ| = %.12f\n",
        E_rimu, E_gp, nΨ)
@assert abs(E_rimu - E_gp) < 1e-9 * max(1, abs(E_gp)) "h1/u inconsistent with H"
@assert abs(nΨ - 1) < 1e-10

ansatz = GrossPitaevskiiAnsatz(addr; valtype=ComplexF64)
gp0 = GPOrbital(C0, h1, U, N)
Ψ0 = gp_vector(db, ansatz, gp0)
coef0 = V' * Ψ0
ψexact(t) = V * (cis.(-E .* t) .* coef0)
@printf("  <H> = %.6f (GP %.6f), spectral range [%.3f, %.3f]\n",
        real(dot(Ψ0, Hd * Ψ0)), gp_energy(gp0), E[1], E[end])
flush(stdout)

# ---- start vectors -------------------------------------------------------------------
function make_dvec(vals, style)
    v = DVec{typeof(addr),ComplexF64}(; style)
    for (k, a) in enumerate(db.basis)
        x = vals[k]
        iszero(x) || (v[a] = x)
    end
    return v
end
det_style = IsDeterministic{ComplexF64}()
sto_style = IsDynamicSemistochastic{ComplexF64}()
ψ̃0 = abs2.(Ψ0)                      # conj(Ψ_GP)·Ψ_GP, sums to 1
start_guided(W, style) = make_dvec(W .* ψ̃0, style)
start_unguided(W, style) = make_dvec((W / sum(abs, Ψ0)) .* Ψ0, style)

function problem(Hop, v; strategy, seed=1)
    return QuantumDynamicsProblem(
        Hop;
        start_at=v,
        time_step=DT,
        maximum_time=10 * TMAX + 1,        # the lock-step driver decides when to stop
        evolution_strategy=strategy,
        threading=false,
        random_seed=seed,
    )
end

runs = Run[]
# deterministic
push!(runs, Run("exact_unguided", init(problem(H, start_unguided(1000, det_style);
                strategy=ExactEvolution())); scheme=:none))
let gp = GPOrbital(C0, h1, U, N)
    push!(runs, Run("exact_guided", init(problem(TDGPGuided(H, gp), start_guided(1000, det_style);
                    strategy=ExactEvolution())); scheme=:generator, gp, midpoint=true))
end
let gp = GPOrbital(C0, h1, U, N)
    push!(runs, Run("exact_guided_nodt", init(problem(TDGPGuided(H, gp; dt_term=false),
                    start_guided(1000, det_style); strategy=ExactEvolution()));
                    scheme=:generator, gp, midpoint=true))
end
push!(runs, Run("pec_unguided_det", init(problem(H, start_unguided(1000, det_style);
                strategy=PEC())); scheme=:none))
let gp = GPOrbital(C0, h1, U, N)
    push!(runs, Run("pec_guided_det", init(problem(TDGPGuided(H, gp), start_guided(1000, det_style);
                    strategy=PEC())); scheme=:generator, gp))
end
# stochastic
for W in WALKERS, s in SEEDS
    push!(runs, Run("pec_unguided_W$(W)_s$(s)", init(problem(H, start_unguided(W, sto_style);
                    strategy=PEC(), seed=1000W + s)); scheme=:none))
    let gp = GPOrbital(C0, h1, U, N)
        push!(runs, Run("pec_guided_W$(W)_s$(s)", init(problem(TDGPGuided(H, gp),
                        start_guided(W, sto_style); strategy=PEC(), seed=1000W + s));
                        scheme=:generator, gp))
    end
end
println("  runs: ", join((r.label for r in runs), ", "))
println("  dt = $DT, tmax = $TMAX, steps = $NSTEPS")
flush(stdout)

gp_meas = GPOrbital(C0, h1, U, N)
wall = @elapsed run_lockstep!(runs, db, ansatz, gp_meas, ψexact; dt=DT, nsteps=NSTEPS,
                              every=EVERY, obdm_every=5)
@printf("  lock-step finished in %.1f s\n", wall)

# ---- output --------------------------------------------------------------------------
function write_csv(file, rows)
    cols = sort(collect(keys(first(rows))))
    open(file, "w") do io
        println(io, join(cols, ","))
        for r in rows
            println(io, join((string(Float64(r[c])) for c in cols), ","))
        end
    end
end
for r in runs
    write_csv(joinpath(OUTDIR, r.label * ".csv"), r.rows)
end

# long-time exact GP fidelity (no QMC), plus GP Lyapunov exponent of this trajectory
let gp = GPOrbital(C0, h1, U, N), rows = Any[]
    for t in 0:0.02:(QUICK ? 0.2 : 20.0)
        advance_gp!(gp, t)
        Ψ = gp_vector(db, ansatz, gp)
        ψe = ψexact(t)
        push!(rows, Dict{String,Any}("time" => t, "fgp_exact" => abs2(dot(Ψ, ψe)),
                                     "gp_energy" => gp_energy(gp)))
    end
    write_csv(joinpath(OUTDIR, "fgp_long.csv"), rows)
end
λ, _, _ = lyapunov_gp(C0, h1, U, N; tmax=QUICK ? 1.0 : 100.0)

summary = IOBuffer()
println(summary, "script      = stage1_verify.jl")
println(summary, "versions    = Rimu $(pkgversion(Rimu)), RimuRealTime $(pkgversion(RimuRealTime)), Gutzwiller $(pkgversion(Gutzwiller))")
println(summary, "hamiltonian = $H")
println(summary, "dim         = $(length(db.basis))")
println(summary, "c0          = $C0")
println(summary, "dt          = $DT, tmax = $TMAX, every = $EVERY")
println(summary, "walkers     = $WALKERS, seeds = $SEEDS")
@printf(summary, "gp_check    = E_rimu %.12f  E_gp %.12f\n", E_rimu, E_gp)
@printf(summary, "lyapunov_gp = %.5f\n", λ)
println(summary, "\nfinal-time results (t = $TMAX):")
for r in runs
    row = r.rows[end]
    minfid = minimum(x["fid_exact"] for x in r.rows)
    @printf(summary, "  %-26s fid_end = %.8f  min_fid = %.8f  sign_end = %.4f  l1_end = %10.2f  len_end = %4d\n",
            r.label, row["fid_exact"], minfid, row["sign_raw"], row["l1_raw"], row["len"])
end
txt = String(take!(summary))
write(joinpath(OUTDIR, "params.txt"), txt)
print(txt)
println("wrote ", OUTDIR)
