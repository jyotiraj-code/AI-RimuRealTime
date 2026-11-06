# rrt_smoke_test.jl
#
# Very small RimuRealTime.jl smoke test.
#
# System : 1D Bose-Hubbard ring, M = 4 sites, N = 4 bosons (Hilbert dim 35),
#          t = 1, u = 4, quench from the Mott state |1,1,1,1>.
# Runs   : 1. ExactEvolution  (Krylov exp(-iH dt), deterministic)  -> reference
#          2. PEC, deterministic style                              -> integrator error
#          3. PEC, dynamic semistochastic style, few walkers        -> stochastic noise
# Output : one CSV per run in ~/claude/importance/data/rrt_smoke_test/
#          (complex columns split into <name>_re / <name>_im)
#
# Submit : cd ~/claude/importance/runs && sbatch ~/claude/slurm_claude/submit.sl rrt_smoke_test.jl

using Rimu
using RimuRealTime
using DataFrames
using CSV
using Printf

const OUTDIR = joinpath(homedir(), "claude", "importance", "data", "rrt_smoke_test")
mkpath(OUTDIR)

# ---- model ---------------------------------------------------------------------------
const ADDR = BoseFS(1, 1, 1, 1)
const U = 4.0
const T = 1.0
const H = HubbardReal1D(ADDR; u=U, t=T)

const DT = 0.01
const TMAX = 2.0
const NWALKERS = 200
const SEED = 1234

println("RimuRealTime smoke test")
println("  Hamiltonian : ", H)
println("  dimension   : ", dimension(H))
println("  dt = $DT, tmax = $TMAX, threads = $(Threads.nthreads())")
flush(stdout)

# observables: 2-norm of the state and the amplitude on the initial Fock state
post_step_strategy() = (
    Projector(norm2=Norm2Projector()),
    Projector(c0=DVec(ADDR => 1.0)),
)

# ---- helpers -------------------------------------------------------------------------
"Split complex-valued columns into `<name>_re` and `<name>_im` so the CSV is plain numbers."
function flatten_complex(df::DataFrame)
    out = DataFrame()
    for name in names(df)
        col = df[!, name]
        if !isempty(col) && all(x -> x isa Complex, col)
            out[!, name * "_re"] = real.(col)
            out[!, name * "_im"] = imag.(col)
        else
            out[!, name] = col
        end
    end
    return out
end

function run_case(label; kwargs...)
    println("\n=== $label ===")
    flush(stdout)
    problem = QuantumDynamicsProblem(
        H;
        start_at=ADDR,
        time_step=DT,
        maximum_time=TMAX,
        post_step_strategy=post_step_strategy(),
        kwargs...,
    )
    elapsed = @elapsed simulation = solve(problem)
    df = DataFrame(simulation)
    println("  finished in ", @sprintf("%.2f", elapsed), " s, ", nrow(df), " rows")
    println("  columns: ", join(names(df), ", "))
    flat = flatten_complex(df)
    file = joinpath(OUTDIR, "$label.csv")
    CSV.write(file, flat)
    println("  wrote ", file)
    show(stdout, first(flat, 3); allcols=true)
    println()
    flush(stdout)
    return flat
end

# ---- runs ----------------------------------------------------------------------------
cases = [
    ("exact", (; evolution_strategy=ExactEvolution(),
                 style=IsDeterministic{ComplexF64}())),
    ("pec_det", (; evolution_strategy=PEC(),
                   style=IsDeterministic{ComplexF64}())),
    ("pec_stoch", (; evolution_strategy=PEC(),
                     style=IsDynamicSemistochastic{ComplexF64}(),
                     initial_walkers=NWALKERS,
                     random_seed=SEED)),
]

status = String[]
for (label, kw) in cases
    try
        run_case(label; kw...)
        push!(status, "$label: ok")
    catch err
        println("  FAILED: ")
        showerror(stdout, err, catch_backtrace())
        println()
        push!(status, "$label: FAILED ($(typeof(err)))")
    end
end

open(joinpath(OUTDIR, "params.txt"), "w") do io
    println(io, "script      = rrt_smoke_test.jl")
    println(io, "hamiltonian = HubbardReal1D(BoseFS(1,1,1,1); u=$U, t=$T)")
    println(io, "dimension   = ", dimension(H))
    println(io, "dt          = $DT")
    println(io, "tmax        = $TMAX")
    println(io, "walkers     = $NWALKERS (pec_stoch only)")
    println(io, "seed        = $SEED (pec_stoch only)")
    foreach(s -> println(io, "status      = ", s), status)
end

println("\nSummary:")
foreach(s -> println("  ", s), status)
