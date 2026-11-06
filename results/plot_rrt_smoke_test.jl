# plot_rrt_smoke_test.jl
#
# Plots the RimuRealTime smoke test (data/rrt_smoke_test, produced on ctcp by
# scripts/rrt_smoke_test.jl): 4-site Bose-Hubbard ring, u = 4, quench from |1,1,1,1>.
#
# Run from the repo root:
#   julia --project=devrimu results/plot_rrt_smoke_test.jl
#
# Output: results/rrt_smoke_test.png and results/rrt_smoke_test_summary.txt

ENV["GKSwstype"] = "100"   # headless GR

using DelimitedFiles
using Plots
using Printf

const ROOT = dirname(@__DIR__)
const DATA = joinpath(ROOT, "data", "rrt_smoke_test")
const OUT = @__DIR__

"Read one of the smoke-test CSV files into a Dict column name => Vector{Float64}."
function load(name)
    raw, header = readdlm(joinpath(DATA, "$name.csv"), ',', Float64; header=true)
    return Dict(String(h) => raw[:, j] for (j, h) in enumerate(vec(header)))
end

runs = ("exact", "pec_det", "pec_stoch")
labels = Dict("exact" => "Exact (Krylov)",
              "pec_det" => "PEC deterministic",
              "pec_stoch" => "PEC semistochastic (200 walkers)")
colors = Dict("exact" => :black, "pec_det" => :dodgerblue, "pec_stoch" => :orangered)
styles = Dict("exact" => :solid, "pec_det" => :dash, "pec_stoch" => :solid)

d = Dict(r => load(r) for r in runs)

# return probability P0(t) = |<1111|psi(t)>|^2 / ||psi(t)||^2
p0(x) = @. (x["c0_re"]^2 + x["c0_im"]^2) / x["norm2"]^2
# relative 2-norm drift, measured from the first reported step
normdrift(x) = x["norm2"] ./ x["norm2"][1] .- 1

t = d["exact"]["time"]
P = Dict(r => p0(d[r]) for r in runs)

pa = plot(title="(a) Return probability", xlabel="t", ylabel="P₀(t)",
          legend=:right, legendfontsize=7)
for r in runs
    plot!(pa, d[r]["time"], P[r]; label=labels[r], color=colors[r], ls=styles[r], lw=2)
end

pb = plot(title="(b) |P₀ − P₀(exact)|", xlabel="t", ylabel="abs. error",
          yscale=:log10, legend=:bottomright, legendfontsize=7)
for r in ("pec_det", "pec_stoch")
    err = abs.(P[r] .- P["exact"])
    plot!(pb, d[r]["time"], max.(err, 1e-16); label=labels[r], color=colors[r], lw=2)
end

pc = plot(title="(c) 2-norm drift", xlabel="t", ylabel="‖ψ(t)‖₂ / ‖ψ(t₁)‖₂ − 1",
          legend=:topleft, legendfontsize=7)
for r in runs
    plot!(pc, d[r]["time"], normdrift(d[r]); label=labels[r], color=colors[r],
          ls=styles[r], lw=2)
end

# 1-norm / 2-norm ratio: grows as the state spreads over configurations;
# for a stochastic run it also tracks the phase noise.
pd = plot(title="(d) Spread: ‖ψ‖₁ / ‖ψ‖₂ and occupied configs",
          xlabel="t", ylabel="‖ψ‖₁ / ‖ψ‖₂", legend=:bottomright, legendfontsize=7)
for r in runs
    plot!(pd, d[r]["time"], d[r]["walkers"] ./ d[r]["norm2"]; label=labels[r],
          color=colors[r], ls=styles[r], lw=2)
end
pd2 = twinx(pd)
plot!(pd2, d["pec_stoch"]["time"], d["pec_stoch"]["len"]; label="len (stoch)",
      color=colors["pec_stoch"], ls=:dot, lw=1.5, ylabel="occupied configs (of 35)",
      legend=:right, legendfontsize=7, ylims=(0, 36))

fig = plot(pa, pb, pc, pd; layout=(2, 2), size=(1150, 800), margin=4Plots.mm,
           right_margin=14Plots.mm,
           plot_title="RimuRealTime smoke test: 4-site Bose-Hubbard, u = 4, |1111⟩ quench")
png_path = joinpath(OUT, "rrt_smoke_test.png")
savefig(fig, png_path)

# ---- numbers -------------------------------------------------------------------------
summary = IOBuffer()
println(summary, "RimuRealTime smoke test summary (data/rrt_smoke_test)")
println(summary, "steps reported: ", length(t), ", t = ", t[1], " … ", t[end])
for r in ("pec_det", "pec_stoch")
    err = abs.(P[r] .- P["exact"])
    @printf(summary, "%-10s max|ΔP0| = %.3e   mean|ΔP0| = %.3e   final norm drift = %+.3e\n",
            r, maximum(err), sum(err) / length(err), normdrift(d[r])[end])
end
@printf(summary, "%-10s final norm drift = %+.3e\n", "exact", normdrift(d["exact"])[end])
@printf(summary, "min P0(exact) = %.4f at t = %.2f\n", minimum(P["exact"]),
        t[argmin(P["exact"])])
@printf(summary, "pec_stoch occupied configs: min %d, max %d (of 35)\n",
        minimum(d["pec_stoch"]["len"]), maximum(d["pec_stoch"]["len"]))
txt = String(take!(summary))
write(joinpath(OUT, "rrt_smoke_test_summary.txt"), txt)
print(txt)
println("saved ", png_path)
