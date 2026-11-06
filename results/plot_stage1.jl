# plot_stage1.jl
#
# STAGE 1 figures: verification of time-dependent GP importance sampling in RimuRealTime
# (data/stage1_verify, produced on ctcp by scripts/stage1_verify.jl).
#
# Run from the repo root:   julia --project=devrimu results/plot_stage1.jl
# Output: results/stage1_verify.png, results/stage1_summary.txt

ENV["GKSwstype"] = "100"
using DelimitedFiles
using Plots
using Printf
using Statistics

const ROOT = dirname(@__DIR__)
const DATA = joinpath(ROOT, "data", "stage1_verify")
const OUT = @__DIR__

function load(name)
    raw, header = readdlm(joinpath(DATA, "$name.csv"), ',', Float64; header=true)
    return Dict(String(h) => raw[:, j] for (j, h) in enumerate(vec(header)))
end
files = filter(f -> endswith(f, ".csv"), readdir(DATA))
d = Dict(replace(f, ".csv" => "") => load(replace(f, ".csv" => "")) for f in files)
walkers = sort(unique(parse(Int, m.captures[1]) for k in keys(d)
                      for m in (match(r"_W(\d+)_s", k),) if m !== nothing))
seeds(prefix) = sort([k for k in keys(d) if startswith(k, prefix)])

infid(x) = max.(1 .- x["fid_exact"], 1e-16)
t = d["exact_guided"]["time"]
M = count(k -> occursin(r"^n\d+$", k), keys(d["exact_guided"]))

default(lw=2, legendfontsize=7, guidefontsize=9, titlefontsize=10, framestyle=:box)

# (a) deterministic infidelity of back-transformed ψ vs exact diagonalisation
pa = plot(title="(a) Deterministic: 1 − |⟨ψ_exact|ψ⟩|²", xlabel="t", yscale=:log10,
          ylims=(1e-16, 1), legend=:bottomright)
for (k, lab, c, ls) in (("exact_unguided", "Krylov, unguided", :gray, :solid),
                        ("exact_guided", "Krylov, GP-guided", :black, :solid),
                        ("pec_unguided_det", "PEC, unguided", :dodgerblue, :dash),
                        ("pec_guided_det", "PEC, GP-guided", :orangered, :solid),
                        ("exact_guided_nodt", "Krylov, guided, no ∂ₜφ/φ", :purple, :dot))
    plot!(pa, d[k]["time"], infid(d[k]); label=lab, color=c, ls=ls)
end

# (b) site density: exact vs guided QMC (deterministic) vs GP mean field
pb = plot(title="(b) Site density n₁(t)", xlabel="t", ylabel="⟨n₁⟩", legend=:topright)
plot!(pb, t, d["exact_unguided"]["n1"]; label="exact", color=:black)
plot!(pb, t, d["pec_guided_det"]["n1"]; label="PEC GP-guided (back-transformed)",
      color=:orangered, ls=:dash)
plot!(pb, t, d["exact_guided"]["n1_gp"]; label="GP mean field N|c₁|²", color=:green, ls=:dot)
s = seeds("pec_guided_W$(walkers[end])_")
plot!(pb, t, mean(d[k]["n1"] for k in s); label="guided QMC W=$(walkers[end]) (seed mean)",
      color=:orangered, alpha=0.5, lw=1)

# (c) stochastic infidelity: guided vs unguided
pc = plot(title="(c) Semistochastic PEC: 1 − fidelity", xlabel="t", yscale=:log10,
          legend=:bottomright)
cols = Dict(walkers[1] => :dodgerblue, walkers[end] => :orangered)
for W in walkers, (g, ls) in (("unguided", :dash), ("guided", :solid))
    ks = seeds("pec_$(g)_W$(W)_")
    m = mean(infid(d[k]) for k in ks)
    plot!(pc, t, m; label="$g, W=$W", color=cols[W], ls=ls)
end

# (d) average sign of the sampled vector vs the bound sqrt(F_GP)
pd = plot(title="(d) Average phase |Σv|/Σ|v| of sampled vector", xlabel="t",
          ylims=(0, 1.02), legend=:bottomleft)
plot!(pd, t, d["exact_unguided"]["sign_raw"]; label="unguided (exact)", color=:gray)
plot!(pd, t, d["exact_guided"]["sign_raw"]; label="GP-guided ψ̃ (exact)", color=:black)
plot!(pd, t, sqrt.(d["exact_guided"]["fgp_exact"]); label="bound √F_GP", color=:green, ls=:dot)
for W in walkers
    ks = seeds("pec_guided_W$(W)_")
    plot!(pd, t, mean(d[k]["sign_raw"] for k in ks); label="guided QMC W=$W",
          color=cols[W], lw=1)
end

# (e) mean-field fidelity at long times
fl = load("fgp_long")
pe = plot(fl["time"], fl["fgp_exact"]; title="(e) Mean-field fidelity F_GP = |⟨Ψ_GP|ψ⟩|²",
          xlabel="t", label="exact many-body", color=:green, ylims=(0, 1.02))
hline!(pe, [0.5]; color=:gray, ls=:dot, label="")

# (f) condensate fraction
mask = .!isnan.(d["exact_guided"]["cfrac"])
pf = plot(title="(f) Condensate fraction λ_max(ρ)/N", xlabel="t", legend=:bottomleft)
plot!(pf, t[mask], d["exact_unguided"]["cfrac_exact"][mask]; label="exact", color=:black)
plot!(pf, t[mask], d["pec_guided_det"]["cfrac"][mask]; label="PEC GP-guided",
      color=:orangered, ls=:dash, marker=:circle, ms=2)
hline!(pf, [1.0]; label="GP mean field", color=:green, ls=:dot)

fig = plot(pa, pb, pc, pd, pe, pf; layout=(3, 2), size=(1150, 1200), margin=4Plots.mm,
           plot_title="Stage 1: time-dependent GP importance sampling, Bose-Hubbard ring M=$M")
savefig(fig, joinpath(OUT, "stage1_verify.png"))

io = IOBuffer()
println(io, "Stage 1 summary (data/stage1_verify)")
for k in sort(collect(keys(d)))
    k == "fgp_long" && continue
    x = d[k]
    @printf(io, "  %-26s max(1-fid) = %.3e   end(1-fid) = %.3e   end sign = %.4f   end l1 = %9.2f\n",
            k, maximum(1 .- x["fid_exact"]), 1 - x["fid_exact"][end], x["sign_raw"][end], x["l1_raw"][end])
end
ok = all(d["exact_guided"]["sign_raw"] .>= sqrt.(d["exact_guided"]["fgp_exact"]) .- 1e-12)
println(io, "  bound sign(ψ̃) ≥ √F_GP holds for exact guided run: ", ok)
i = findfirst(<(0.5), fl["fgp_exact"])
println(io, "  F_GP first < 1/2 at t = ", i === nothing ? "never (t ≤ $(fl["time"][end]))" : fl["time"][i])
txt = String(take!(io))
write(joinpath(OUT, "stage1_summary.txt"), txt)
print(txt)
