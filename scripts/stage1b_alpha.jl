# stage1b_alpha.jl
#
# STAGE 1b: fix and verify the GP-guided real-time scheme.
#
# Stage 1 showed that the full GP guide φ = conj(Ψ_GP) (α = 1) (i) makes explicit PEC
# inaccurate when the GP orbital passes near a node (|c_m| → 0), and (ii) makes the
# semistochastic walker population explode (walker density ∝ |ψ|² starves the tails,
# then the ∂tφ/φ factor amplifies the rounding noise).
#
# Here we test two remedies on the same system (M = 4, N = 8, u = t = 1, GP initial state):
#   * exact reweighting scheme: step with the frozen H̃(t_n), then ψ̃ ← φ(t_{n+1})/φ(t_n) ψ̃
#     (no splitting error, since D e^{−iH dt} D⁻¹ = e^{−iH̃ dt});
#   * softened guides φ_α = |Ψ_GP|^α e^{−iθ_GP}, α ∈ {1, 0.5, 0}; α = 0 is a pure,
#     unitary GP-phase gauge transformation.
# Deterministic: reweighted PEC must reproduce unguided PEC to rounding error for every α.
# Stochastic: semistochastic PEC at W ∈ {100, 300, 1000}, 4 seeds, unguided vs α-guides.
#
# Submit: cd ~/claude/importance/runs && sbatch ~/claude/slurm_claude/submit.sl stage1b_alpha.jl

using Rimu
using RimuRealTime
using Gutzwiller
using LinearAlgebra
using Printf

const QUICK = get(ENV, "STAGE1_QUICK", "0") == "1"
include(joinpath(@__DIR__, "GPGuidedDynamics.jl"))
using .GPGuidedDynamics

const OUTDIR = QUICK ? mktempdir() :
    joinpath(homedir(), "claude", "importance", "data", "stage1b_alpha")
mkpath(OUTDIR)

const M = QUICK ? 3 : 4
const N = QUICK ? 3 : 8
const U = 1.0
const J = 1.0
const DT = QUICK ? 0.01 : 0.005
const TMAX = QUICK ? 0.1 : 4.0
const NSTEPS = round(Int, TMAX / DT)
const EVERY = QUICK ? 1 : 2
const ALPHAS = (1.0, 0.5, 0.0)
const WALKERS = QUICK ? (50,) : (100, 300, 1000)
const SEEDS = QUICK ? (1,) : (1, 2, 3, 4)
const C0 = let c = ComplexF64[1.0, 0.8cis(0.3), 0.6cis(-0.5), 0.9cis(1.0)][1:M]
    c ./ norm(c)
end

println("STAGE 1b (quick = $QUICK)")
for p in (Rimu, RimuRealTime, Gutzwiller)
    println("  ", rpad(string(p), 14), pkgversion(p))
end

addr = near_uniform(BoseFS{N,M})
H = HubbardReal1D(addr; u=U, t=J)
bsr = BasisSetRepresentation(H)
Hd = Matrix(bsr.sparse_matrix)
db = DenseBasis(bsr.basis)
E, V = eigen(Hermitian(Hd))
h1 = ring_hopping(M, J)
E_rimu, E_gp, _ = check_gp_consistency(H, h1, U, db, Hd)
@assert abs(E_rimu - E_gp) < 1e-9 * max(1, abs(E_gp))

ansatz = GrossPitaevskiiAnsatz(addr; valtype=ComplexF64)
Ψ0 = gp_vector(db, ansatz, GPOrbital(C0, h1, U, N))
coef0 = V' * Ψ0
ψexact(t) = V * (cis.(-E .* t) .* coef0)
println("  H = $H, dim = $(length(db.basis))")

function make_dvec(vals, style)
    v = DVec{typeof(addr),ComplexF64}(; style)
    for (k, a) in enumerate(db.basis)
        iszero(vals[k]) || (v[a] = vals[k])
    end
    return v
end
det_style = IsDeterministic{ComplexF64}()
sto_style = IsDynamicSemistochastic{ComplexF64}()
start_unguided(W, style) = make_dvec((W / sum(abs, Ψ0)) .* Ψ0, style)
function start_guided(α, W, style)
    ψ̃ = abs.(Ψ0) .^ (α + 1)              # φ_α Ψ0 = |Ψ0|^{α+1}
    return make_dvec((W / sum(ψ̃)) .* ψ̃, style)
end
problem(Hop, v; strategy, seed=1) = QuantumDynamicsProblem(
    Hop; start_at=v, time_step=DT, maximum_time=10 * TMAX + 1,
    evolution_strategy=strategy, threading=false, random_seed=seed)

function guided_run(label, α, scheme, strategy, v; seed=1)
    gp = GPOrbital(C0, h1, U, N)
    Hop = TDGPGuided(H, gp; alpha=α, dt_term=(scheme === :generator))
    midpoint = scheme === :generator && strategy isa ExactEvolution
    return Run(label, init(problem(Hop, v; strategy, seed)); scheme, gp, alpha=α, midpoint)
end

astr(α) = replace(string(α), "." => "p")
runs = Run[]
push!(runs, Run("exact_unguided", init(problem(H, start_unguided(1000, det_style);
                strategy=ExactEvolution()))))
push!(runs, Run("pec_unguided_det", init(problem(H, start_unguided(1000, det_style);
                strategy=PEC()))))
for α in ALPHAS
    push!(runs, guided_run("exact_rw_a$(astr(α))", α, :reweight, ExactEvolution(),
                           start_guided(α, 1000, det_style)))
    push!(runs, guided_run("pec_rw_a$(astr(α))_det", α, :reweight, PEC(),
                           start_guided(α, 1000, det_style)))
    push!(runs, guided_run("pec_gen_a$(astr(α))_det", α, :generator, PEC(),
                           start_guided(α, 1000, det_style)))
end
for W in WALKERS, s in SEEDS
    seed = 1000W + s
    push!(runs, Run("sto_unguided_W$(W)_s$(s)", init(problem(H, start_unguided(W, sto_style);
                    strategy=PEC(), seed))))
    for α in ALPHAS
        push!(runs, guided_run("sto_rw_a$(astr(α))_W$(W)_s$(s)", α, :reweight, PEC(),
                               start_guided(α, W, sto_style); seed))
    end
end
println("  $(length(runs)) runs, dt = $DT, tmax = $TMAX")
flush(stdout)

gp_meas = GPOrbital(C0, h1, U, N)
wall = @elapsed run_lockstep!(runs, db, ansatz, gp_meas, ψexact; dt=DT, nsteps=NSTEPS,
                              every=EVERY, obdm_every=5)
@printf("  lock-step finished in %.1f s\n", wall)

function write_csv(file, rows)
    cols = sort(collect(keys(first(rows))))
    open(file, "w") do io
        println(io, join(cols, ","))
        for r in rows
            println(io, join((string(Float64(r[c])) for c in cols), ","))
        end
    end
end
foreach(r -> write_csv(joinpath(OUTDIR, r.label * ".csv"), r.rows), runs)

# deterministic equivalence: reweighted PEC vs unguided PEC (back-transformed states)
ref = Dict(x["step"] => x["fid_exact"] for x in runs[2].rows)
io = IOBuffer()
println(io, "script   = stage1b_alpha.jl")
println(io, "versions = Rimu $(pkgversion(Rimu)), RimuRealTime $(pkgversion(RimuRealTime)), Gutzwiller $(pkgversion(Gutzwiller))")
println(io, "H = $H, dim = $(length(db.basis)), c0 = $C0")
println(io, "dt = $DT, tmax = $TMAX, alphas = $ALPHAS, walkers = $WALKERS, seeds = $SEEDS")
println(io, "\nrun                           min_fid      end_fid    end_sign      end_l1  max|fid-fid(pec_unguided)|")
for r in runs
    mf = minimum(x["fid_exact"] for x in r.rows)
    dev = maximum(abs(x["fid_exact"] - ref[x["step"]]) for x in r.rows)
    e = r.rows[end]
    @printf(io, "%-28s %.8f  %.8f  %8.4f  %12.2f  %.2e\n", r.label, mf, e["fid_exact"],
            e["sign_raw"], e["l1_raw"], dev)
end
txt = String(take!(io))
write(joinpath(OUTDIR, "params.txt"), txt)
print(txt)
