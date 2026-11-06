# stage2b_qmc.jl
#
# STAGE 2b: GP-phase-gauge (α = 0, exact reweighting) vs unguided semistochastic
# real-time FCIQMC (RimuRealTime PEC) in regimes where mean field is good or bad.
#
# Cases (Bose-Hubbard trimer ring, initial GP product state, g = u(N−1)/J):
#   vortex_stable   : vortex c_j ∝ e^{2πij/3}, g = 1.5 (< g_c = 9/4, Bogoliubov stable)
#   vortex_unstable : vortex,                  g = 4.0 (dynamically unstable, rate 1.22)
#   imb_regular     : c ∝ (√.6, √.3, √.1),     g = 4.0 (λ_GP ≈ 0.02)
#   imb_chaotic     : c ∝ (√.6, √.3, √.1),     g = 18  (λ_GP ≈ 0.7)
# for N = 24 (dim 325) and, for the vortex, N = 48 (dim 1225).
# Each case: deterministic PEC unguided + gauge (sanity), and semistochastic PEC at three
# walker numbers × 3 seeds, unguided vs gauge. Fidelity vs exact diagonalisation.
# dt is set from the spectrum: S = ⟨H⟩ of the initial state, max|E−S| dt ≤ 0.1.
# (A first run with S = spectral centre and |E−S| dt ≤ 0.3 was integrator-limited; its
#  data are kept in data/stage2b_qmc_dtcoarse.)
#
# Submit: cd ~/claude/importance/runs && sbatch ~/claude/slurm_claude/submit.sl stage2b_qmc.jl

using Rimu
using RimuRealTime
using Gutzwiller
using LinearAlgebra
using Printf

const QUICK = get(ENV, "STAGE2_QUICK", "0") == "1"
include(joinpath(@__DIR__, "GPGuidedDynamics.jl"))
using .GPGuidedDynamics

const OUTDIR = QUICK ? mktempdir() :
    joinpath(homedir(), "claude", "importance", "data", "stage2b_qmc")
mkpath(OUTDIR)
const J = 1.0
const TMAX = QUICK ? 0.05 : 6.0
const SEEDS = QUICK ? (1,) : (1, 2, 3)

vortex(M) = ComplexF64[cis(2π * j / M) for j in 0:M-1] ./ sqrt(M)
imb() = normalize(ComplexF64[sqrt(0.6), sqrt(0.3), sqrt(0.1)])
const CASES = QUICK ? [("vortex_stable", vortex(3), 1.5, 6, (50,))] : [
    ("vortex_stable", vortex(3), 1.5, 24, (300, 1000, 3000)),
    ("vortex_unstable", vortex(3), 4.0, 24, (300, 1000, 3000)),
    ("imb_regular", imb(), 4.0, 24, (300, 1000, 3000)),
    ("imb_chaotic", imb(), 18.0, 24, (300, 1000, 3000)),
    ("vortex_stable", vortex(3), 1.5, 48, (1000, 3000, 10000)),
    ("vortex_unstable", vortex(3), 4.0, 48, (1000, 3000, 10000)),
]

function write_csv(file, rows)
    cols = sort(collect(keys(first(rows))))
    open(file, "w") do io
        println(io, join(cols, ","))
        for r in rows
            println(io, join((string(Float64(r[c])) for c in cols), ","))
        end
    end
end

summary = open(joinpath(OUTDIR, "summary.txt"), "w")
println(summary, "versions = Rimu $(pkgversion(Rimu)), RimuRealTime $(pkgversion(RimuRealTime)), Gutzwiller $(pkgversion(Gutzwiller))")

for (name, c0, g, N, walkers) in CASES
    M = length(c0)
    u = g / (N - 1)
    addr = near_uniform(BoseFS{N,M})
    H = HubbardReal1D(addr; u, t=J)
    bsr = BasisSetRepresentation(H)
    Hd = Matrix(bsr.sparse_matrix)
    db = DenseBasis(bsr.basis)
    E, V = eigen(Hermitian(Hd))
    h1 = ring_hopping(M, J)
    ansatz = GrossPitaevskiiAnsatz(addr; valtype=ComplexF64)
    Ψ0 = gp_vector(db, ansatz, GPOrbital(c0, h1, u, N))
    # Shift by the (conserved) energy of the initial state and keep x = max|E−S| dt ≤ 0.1:
    # PEC amplifies a mode with frequency ω by (1 + x⁴/4)^{1/2} per step (x = ω dt), which
    # at x = 0.3 (first version of this script) caused O(1) integrator error.
    S = real(dot(Ψ0, Hd * Ψ0))
    dt = min(0.005, 0.1 / max(abs(E[1] - S), abs(E[end] - S)))
    nsteps = round(Int, TMAX / dt)
    every = max(1, round(Int, 0.02 / dt))
    coef0 = V' * Ψ0
    ψexact(t) = V * (cis.(-E .* t) .* coef0)
    tag = "$(name)_N$(N)"
    @printf("\n=== %s: H = %s, dim = %d, dt = %.5f, steps = %d, S = %.3f ===\n",
            tag, H, length(db.basis), dt, nsteps, S)
    flush(stdout)

    function make_dvec(vals, style)
        v = DVec{typeof(addr),ComplexF64}(; style)
        for (k, a) in enumerate(db.basis)
            iszero(vals[k]) || (v[a] = vals[k])
        end
        return v
    end
    det = IsDeterministic{ComplexF64}()
    sto = IsDynamicSemistochastic{ComplexF64}()
    start(W, style) = make_dvec((W / sum(abs, Ψ0)) .* Ψ0, style)                  # unguided
    start_gauge(W, style) = make_dvec((W / sum(abs, Ψ0)) .* abs.(Ψ0), style)     # e^{-iθ}Ψ0
    prob(Hop, v; seed=1) = QuantumDynamicsProblem(Hop; start_at=v, time_step=dt, shift=S,
        maximum_time=10 * TMAX + 1, evolution_strategy=PEC(), threading=false, random_seed=seed)
    function gauge_run(label, v; seed=1)
        gp = GPOrbital(c0, h1, u, N)
        Hop = TDGPGuided(H, gp; alpha=0.0, dt_term=false)
        return Run(label, init(prob(Hop, v; seed)); scheme=:reweight, gp, alpha=0.0)
    end

    runs = Run[]
    push!(runs, Run("det_unguided", init(prob(H, start(1000, det)))))
    push!(runs, gauge_run("det_gauge", start_gauge(1000, det)))
    for W in walkers, s in SEEDS
        seed = 7919 * W + s
        push!(runs, Run("sto_unguided_W$(W)_s$(s)", init(prob(H, start(W, sto); seed))))
        push!(runs, gauge_run("sto_gauge_W$(W)_s$(s)", start_gauge(W, sto); seed))
    end
    gp_meas = GPOrbital(c0, h1, u, N)
    wall = @elapsed run_lockstep!(runs, db, ansatz, gp_meas, ψexact; dt, nsteps, every,
                                  obdm_every=10)
    @printf("  %d runs finished in %.1f s\n", length(runs), wall)
    dir = joinpath(OUTDIR, tag)
    mkpath(dir)
    foreach(r -> write_csv(joinpath(dir, r.label * ".csv"), r.rows), runs)

    @printf(summary, "\n%s  H=%s dim=%d dt=%.5f tmax=%.1f\n", tag, H, length(db.basis), dt, TMAX)
    for r in runs
        f = [x["fid_exact"] for x in r.rows]
        e = r.rows[end]
        @printf(summary, "  %-26s mean_fid=%.5f  end_fid=%.5f  end_sign=%.4f  end_l1=%10.1f  fgp_end=%.4f\n",
                r.label, sum(f) / length(f), f[end], e["sign_raw"], e["l1_raw"], e["fgp_exact"])
    end
    flush(summary)
end
close(summary)
print(read(joinpath(OUTDIR, "summary.txt"), String))
