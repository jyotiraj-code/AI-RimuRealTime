# plot_stage1b.jl
#
# STAGE 1b figures: exact reweighting scheme and α-family guides φ_α = |Ψ_GP|^α e^{−iθ_GP}
# (data/stage1b_alpha, produced on ctcp by scripts/stage1b_alpha.jl).
#
# Run from the repo root:   julia --project=devrimu results/plot_stage1b.jl
# Output: results/stage1b_alpha.png, results/stage1b_summary.txt

ENV["GKSwstype"] = "100"
using DelimitedFiles
using Plots
using Printf
using Statistics

const ROOT = dirname(@__DIR__)
const DATA = joinpath(ROOT, "data", "stage1b_alpha")
const OUT = @__DIR__

function load(name)
    raw, header = readdlm(joinpath(DATA, "$name.csv"), ',', Float64; header=true)
    return Dict(String(h) => raw[:, j] for (j, h) in enumerate(vec(header)))
end
d = Dict(replace(f, ".csv" => "") => load(replace(f, ".csv" => ""))
         for f in readdir(DATA) if endswith(f, ".csv"))
t = d["pec_unguided_det"]["time"]
alphas = ("1p0", "0p5", "0p0")
alab = Dict("1p0" => "α=1 (full GP)", "0p5" => "α=½", "0p0" => "α=0 (GP phase gauge)")
acol = Dict("1p0" => :orangered, "0p5" => :goldenrod, "0p0" => :seagreen)
Ws = (100, 300, 1000)
seedkeys(prefix) = sort([k for k in keys(d) if startswith(k, prefix)])
infid(x) = max.(1 .- x, 1e-16)

series1000 = [("unguided", "sto_unguided_W1000_", :black);
              [("reweight, " * alab[a], "sto_rw_a$(a)_W1000_", acol[a]) for a in alphas]]

default(lw=2, legendfontsize=7, guidefontsize=9, titlefontsize=10, framestyle=:box)

pa = plot(title="(a) Deterministic PEC: 1 − fidelity", xlabel="t", yscale=:log10,
          ylims=(1e-8, 2), legend=:bottomright)
plot!(pa, t, infid(d["pec_unguided_det"]["fid_exact"]); label="unguided", color=:black, lw=3)
for a in alphas
    plot!(pa, t, infid(d["pec_rw_a$(a)_det"]["fid_exact"]); label="reweight, " * alab[a],
          color=acol[a], ls=:dash)
    plot!(pa, t, infid(d["pec_gen_a$(a)_det"]["fid_exact"]); label="generator, " * alab[a],
          color=acol[a], ls=:dot)
end

pb = plot(title="(b) Reweighted PEC vs unguided PEC", xlabel="t",
          ylabel="|F(reweight) − F(unguided)|", yscale=:log10, ylims=(1e-17, 1e-12),
          legend=:topleft)
for a in alphas
    dev = abs.(d["pec_rw_a$(a)_det"]["fid_exact"] .- d["pec_unguided_det"]["fid_exact"])
    plot!(pb, t, max.(dev, 1e-17); label=alab[a], color=acol[a])
end

pc = plot(title="(c) Semistochastic, W = 1000: fidelity (seed mean)", xlabel="t",
          ylabel="|⟨ψ_exact|ψ⟩|²", ylims=(0, 1.02), legend=:right)
for (lab, pre, col) in series1000
    plot!(pc, t, mean(d[k]["fid_exact"] for k in seedkeys(pre)); label=lab, color=col)
end

pd = plot(title="(d) Walker 1-norm, W = 1000 (seed mean)", xlabel="t", yscale=:log10,
          legend=:topleft)
for (lab, pre, col) in series1000
    plot!(pd, t, mean(d[k]["l1_raw"] for k in seedkeys(pre)); label=lab, color=col)
end

pe = plot(title="(e) Time-averaged fidelity vs walker number", xlabel="W", xscale=:log10,
          ylabel="⟨F⟩_t (mean ± sd over 4 seeds)", ylims=(0, 1), legend=:topleft,
          xticks=(collect(Ws), string.(Ws)))
for (lab, pre, col, sh) in [("unguided", "sto_unguided", :black, :circle);
                            [("reweight, " * alab[a], "sto_rw_a$(a)", acol[a], :diamond) for a in alphas]]
    m = Float64[]; s = Float64[]
    for W in Ws
        vals = [mean(d[k]["fid_exact"]) for k in seedkeys("$(pre)_W$(W)_")]
        push!(m, mean(vals)); push!(s, std(vals))
    end
    plot!(pe, collect(Ws), m; yerror=s, label=lab, color=col, marker=sh, ms=4)
end

pf = plot(title="(f) Average phase of the exact ψ̃ = φ_α ψ", xlabel="t", ylims=(0, 1.02),
          legend=:topright)
plot!(pf, t, d["pec_unguided_det"]["sign_raw"]; label="unguided ψ", color=:black)
for a in alphas
    plot!(pf, t, d["exact_rw_a$(a)"]["sign_raw"]; label=alab[a], color=acol[a])
end
plot!(pf, t, sqrt.(d["exact_rw_a1p0"]["fgp_exact"]); label="√F_GP", color=:gray, ls=:dot)

fig = plot(pa, pb, pc, pd, pe, pf; layout=(3, 2), size=(1150, 1200), margin=4Plots.mm,
           plot_title="Stage 1b: exact reweighting and softened GP guides (M=4, N=8, u=t=1)")
savefig(fig, joinpath(OUT, "stage1b_alpha.png"))

io = IOBuffer()
println(io, "Stage 1b summary (data/stage1b_alpha)")
for a in alphas
    dev = maximum(abs.(d["pec_rw_a$(a)_det"]["fid_exact"] .- d["pec_unguided_det"]["fid_exact"]))
    @printf(io, "  reweight %-22s max|F - F(unguided PEC)| = %.2e ; generator min F = %.4f\n",
            alab[a], dev, minimum(d["pec_gen_a$(a)_det"]["fid_exact"]))
end
println(io, "  time-averaged fidelity, mean ± sd over seeds:")
for W in Ws, (lab, pre) in [("unguided", "sto_unguided"); [("α=" * a, "sto_rw_a$(a)") for a in alphas]]
    vals = [mean(d[k]["fid_exact"]) for k in seedkeys("$(pre)_W$(W)_")]
    l1 = [maximum(d[k]["l1_raw"]) for k in seedkeys("$(pre)_W$(W)_")]
    @printf(io, "    W=%5d  %-10s <F> = %.4f ± %.4f   max l1 = %.3g\n", W, lab, mean(vals), std(vals), maximum(l1))
end
txt = String(take!(io))
write(joinpath(OUT, "stage1b_summary.txt"), txt)
print(txt)
