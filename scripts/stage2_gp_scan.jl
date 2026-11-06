# stage2_gp_scan.jl
#
# STAGE 2 (parameter scan, mean field only): largest Lyapunov exponent λ of the discrete
# GP dynamics  i ċ = h1 c + g |c|² c  on M-site rings, as a function of g = u(N−1)/J,
# for a few physically motivated initial orbitals. Used to choose regular (λ≈0) and
# chaotic (λ>0) regimes for the many-body break-time study.
#
# Submit: cd ~/claude/importance/runs && sbatch ~/claude/slurm_claude/submit.sl stage2_gp_scan.jl

using LinearAlgebra
using Printf
include(joinpath(@__DIR__, "GPGuidedDynamics.jl"))
using .GPGuidedDynamics

const OUTDIR = joinpath(homedir(), "claude", "importance", "data", "stage2_gp_scan")
mkpath(OUTDIR)

# initial orbitals (normalised inside GPOrbital); N only enters via g, so pass N=2, u=g
inits = Dict(
    "M3_imbalance" => ComplexF64[sqrt(0.6), sqrt(0.3), sqrt(0.1)],          # Josephson-type
    "M3_generic" => ComplexF64[1.0, 0.8cis(0.3), 0.6cis(-0.5)],
    "M3_vortex_pert" => ComplexF64[cis(2π * k / 3) for k in 0:2] .+ ComplexF64[0.05, 0, 0],
    "M4_imbalance" => ComplexF64[sqrt(0.4), sqrt(0.3), sqrt(0.2), sqrt(0.1)],
    "M4_generic" => ComplexF64[1.0, 0.8cis(0.3), 0.6cis(-0.5), 0.9cis(1.0)],
    "M5_imbalance" => ComplexF64[sqrt(0.3), sqrt(0.25), sqrt(0.2), sqrt(0.15), sqrt(0.1)],
)
gs = collect(0.0:0.25:20.0)
const TL = 300.0

open(joinpath(OUTDIR, "lyapunov_scan.csv"), "w") do io
    println(io, "init,M,g,lambda,lambda_half")
    for name in sort(collect(keys(inits)))
        c0 = inits[name]
        M = length(c0)
        h1 = ring_hopping(M, 1.0)
        for g in gs
            λ, ts, lam = lyapunov_gp(c0, h1, g, 2; tmax=TL, dt=2e-3)
            λhalf = lam[findfirst(>=(TL / 2), ts)]   # convergence check
            @printf(io, "%s,%d,%.4f,%.6e,%.6e\n", name, M, g, λ, λhalf)
            @printf("%-16s g = %6.2f  λ = %8.4f  (λ(T/2) = %8.4f)\n", name, g, λ, λhalf)
            flush(stdout)
        end
    end
end
println("wrote ", OUTDIR)
