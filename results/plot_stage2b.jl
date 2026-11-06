# plot_stage2b.jl
#
# STAGE 2b figure: GP-phase-gauge vs unguided semistochastic real-time FCIQMC
# (data/stage2b_qmc, produced on ctcp by scripts/stage2b_qmc.jl).
#
# Run from the repo root:   julia --project=devrimu results/plot_stage2b.jl
# Output: results/stage2b_qmc.png, results/stage2b_summary.txt

ENV["GKSwstype"] = "100"
using DelimitedFiles
using Plots
using Printf
using Statistics

const ROOT = dirname(@__DIR__)
const DATA = joinpath(ROOT, "data", "stage2b_qmc")
const OUT = @__DIR__

function load(path)
    raw, header = readdlm(path, ',', Float64; header=true)
    return Dict(String(h) => raw[:, j] for (j, h) in enumerate(vec(header)))
end
tags = filter(t -> isdir(joinpath(DATA, t)), readdir(DATA))
order = ["vortex_stable_N24", "vortex_unstable_N24", "imb_regular_N24", "imb_chaotic_N24",
         "vortex_stable_N48", "vortex_unstable_N48"]
tags = [t for t in order if t in tags]
D = Dict(t => Dict(replace(f, ".csv" => "") => load(joinpath(DATA, t, f))
                   for f in readdir(joinpath(DATA, t)) if endswith(f, ".csv")) for t in tags)
walkers(t) = sort(unique(parse(Int, m.captures[1]) for k in keys(D[t])
                         for m in (match(r"_W(\d+)_s", k),) if m !== nothing))
runs(t, kind, W) = sort([k for k in keys(D[t]) if startswith(k, "sto_$(kind)_W$(W)_")])
meanfid(t, kind, W) = mean(D[t][k]["fid_exact"] for k in runs(t, kind, W))
titles = Dict("vortex_stable_N24" => "vortex g=1.5 (stable), N=24",
              "vortex_unstable_N24" => "vortex g=4 (unstable), N=24",
              "imb_regular_N24" => "imbalance g=4 (regular), N=24",
              "imb_chaotic_N24" => "imbalance g=18 (chaotic), N=24",
              "vortex_stable_N48" => "vortex g=1.5 (stable), N=48",
              "vortex_unstable_N48" => "vortex g=4 (unstable), N=48")

default(lw=1.8, legendfontsize=6, guidefontsize=9, titlefontsize=9, framestyle=:box)
cols = [:dodgerblue, :darkorange, :purple]
panels = Any[]
for (i, t) in enumerate(tags)
    p = plot(title="($(Char('a' + i - 1))) " * titles[t], xlabel="t", ylabel="fidelity",
             ylims=(0, 1.02), legend=:bottomleft)
    tt = D[t]["det_unguided"]["time"]
    plot!(p, tt, D[t]["det_unguided"]["fgp_exact"]; color=:gray, ls=:dot, label="F_GP (exact)")
    for (j, W) in enumerate(walkers(t))
        plot!(p, tt, meanfid(t, "unguided", W); color=cols[j], ls=:dash, label="unguided W=$W")
        plot!(p, tt, meanfid(t, "gauge", W); color=cols[j], label="GP gauge W=$W")
    end
    push!(panels, p)
end

# (g) gain: time-averaged fidelity, gauge vs unguided, all cases
pg = plot(title="(g) ⟨F⟩_t: GP gauge (filled) vs unguided (open)", xlabel="W", xscale=:log10,
          ylabel="⟨fidelity⟩_t", ylims=(0, 1.02), legend=:outerright, size=(400, 300))
ccase = Dict(zip(tags, [:blue, :red, :green, :orange, :navy, :darkred]))
io = IOBuffer()
println(io, "Stage 2b summary: time-averaged fidelity <F>_t (mean ± sd over seeds)")
for t in tags
    Ws = walkers(t)
    mg = Float64[]; mu = Float64[]
    for W in Ws
        g = [mean(D[t][k]["fid_exact"]) for k in runs(t, "gauge", W)]
        u = [mean(D[t][k]["fid_exact"]) for k in runs(t, "unguided", W)]
        push!(mg, mean(g)); push!(mu, mean(u))
        sg = [mean(D[t][k]["sign_raw"]) for k in runs(t, "gauge", W)]
        su = [mean(D[t][k]["sign_raw"]) for k in runs(t, "unguided", W)]
        @printf(io, "  %-20s W=%6d  gauge %.4f ± %.4f   unguided %.4f ± %.4f   sign gauge %.3f  unguided %.3f\n",
                t, W, mean(g), std(g), mean(u), std(u), mean(sg), mean(su))
    end
    plot!(pg, Ws, mg; color=ccase[t], marker=:circle, label=t)
    plot!(pg, Ws, mu; color=ccase[t], marker=:circle, markercolor=:white, ls=:dash, label="")
end

# (h) sampled-vector average phase, stable vortex
t = "vortex_stable_N48" in tags ? "vortex_stable_N48" : tags[1]
ph = plot(title="(h) Average phase of sampled vector, " * titles[t], xlabel="t",
          ylims=(0, 1.02), legend=:right)
tt = D[t]["det_unguided"]["time"]
plot!(ph, tt, D[t]["det_unguided"]["sign_raw"]; color=:black, ls=:dash, label="unguided, exact")
plot!(ph, tt, D[t]["det_gauge"]["sign_raw"]; color=:black, label="GP gauge, exact")
for (j, W) in enumerate(walkers(t))
    plot!(ph, tt, mean(D[t][k]["sign_raw"] for k in runs(t, "gauge", W)); color=cols[j],
          label="gauge QMC W=$W", lw=1)
end

fig = plot(panels..., pg, ph; layout=(4, 2), size=(1150, 1500), margin=4Plots.mm,
           plot_title="Stage 2b: GP-phase-gauge real-time FCIQMC (trimer, semistochastic PEC)")
savefig(fig, joinpath(OUT, "stage2b_qmc.png"))
txt = String(take!(io))
write(joinpath(OUT, "stage2b_summary.txt"), txt)
print(txt)
