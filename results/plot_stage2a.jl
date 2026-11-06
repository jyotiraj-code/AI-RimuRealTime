# plot_stage2a.jl
#
# STAGE 2a figure: quantum break time of GP mean field in the Bose-Hubbard trimer
# (data/stage2a_breaktime, produced on ctcp by scripts/stage2a_breaktime.jl).
#
# Run from the repo root:   julia --project=devrimu results/plot_stage2a.jl
# Output: results/stage2a_breaktime.png, results/stage2a_fits.txt

ENV["GKSwstype"] = "100"
using DelimitedFiles
using Plots
using Printf
using Statistics

const ROOT = dirname(@__DIR__)
const DATA = joinpath(ROOT, "data", "stage2a_breaktime")
const OUT = @__DIR__

function loadcsv(path)
    raw, header = readdlm(path, ',', Any; header=true)
    return Dict(String(h) => raw[:, j] for (j, h) in enumerate(vec(header)))
end
num(x) = Float64.(x)
load(case, g, N) = (d = loadcsv(joinpath(DATA, @sprintf("%s_g%.1f_N%d.csv", case, g, N)));
                    Dict(k => num(v) for (k, v) in d))
S = loadcsv(joinpath(DATA, "summary.csv"))
rowsel(case, g) = findall(i -> S["case"][i] == case && isapprox(Float64(S["g"][i]), g), eachindex(S["case"]))
Ns(case, g) = Int.(S["N"][rowsel(case, g)])

# Bogoliubov two-mode prediction for the trimer vortex (see stage2c_threshold.jl)
const EBAR = -1.5
function FB(g, t)
    A = EBAR + g / 3; B = g / 3; Ω2 = A^2 - B^2
    s = Ω2 > 0 ? sin(sqrt(Ω2) * t) / sqrt(Ω2) : sinh(sqrt(-Ω2) * t) / sqrt(-Ω2)
    return 1 / (1 + B^2 * s^2)
end
κ(g) = (A = EBAR + g / 3; sqrt(max(0, (g / 3)^2 - A^2)))

default(lw=1.8, legendfontsize=7, guidefontsize=9, titlefontsize=10, framestyle=:box)
pal(n) = cgrad(:viridis, max(n, 2); categorical=true)

# (a) stable vortex: N-independent fidelity, Bogoliubov collapse
pa = plot(title="(a) Stable vortex g=1: F_GP(t)", xlabel="t", ylabel="|⟨Ψ_GP|ψ⟩|²",
          ylims=(0.88, 1.005), xlims=(0, 15), legend=:bottomright)
NN = Ns("vortex", 1.0)
for (i, N) in enumerate(NN)
    d = load("vortex", 1.0, N)
    plot!(pa, d["time"], d["fgp"]; label="N=$N", color=pal(length(NN))[i])
end
tt = range(0, 15; length=600)
plot!(pa, tt, FB.(1.0, tt); label="Bogoliubov", color=:red, ls=:dash, lw=2)

# (b) unstable vortex: N-independent fidelity decay
pb = plot(title="(b) Unstable vortex g=4: F_GP(t)", xlabel="t", ylims=(0, 1.02),
          xlims=(0, 3), legend=:topright)
NN = Ns("vortex", 4.0)
for (i, N) in enumerate(NN)
    d = load("vortex", 4.0, N)
    plot!(pb, d["time"], d["fgp"]; label="N=$N", color=pal(length(NN))[i])
end
tt = range(0, 3; length=300)
plot!(pb, tt, FB.(4.0, tt); label="Bogoliubov", color=:red, ls=:dash, lw=2)

# (c) unstable vortex: depletion vs t, shifted by ln N
pc = plot(title="(c) Unstable vortex g=4: depletion 1−λ_max/N", xlabel="t", yscale=:log10,
          ylims=(1e-3, 1), xlims=(0, 4), legend=:bottomright)
for (i, N) in enumerate(NN)
    d = load("vortex", 4.0, N)
    plot!(pc, d["time"], max.(d["depletion"], 1e-6); label="N=$N", color=pal(length(NN))[i])
end
hline!(pc, [0.1]; color=:gray, ls=:dot, label="10%")

# (d) break times vs ln N: unstable vortex (theory slope 1/2κ) ; F-break (flat)
pd = plot(title="(d) Break times, vortex", xlabel="ln N", ylabel="t", legend=:topleft)
fits = IOBuffer()
println(fits, "Stage 2a fits (t_dep = a + b ln N for unstable; t_dep = c N^β for regular)")
for (g, col) in ((3.0, :purple), (4.0, :orangered), (6.0, :goldenrod))
    idx = rowsel("vortex", g)
    x = log.(Float64.(S["N"][idx])); td = Float64.(S["t_depl10"][idx]); tF = Float64.(S["t_break"][idx])
    b = cov(x, td) / var(x); a = mean(td) - b * mean(x)
    scatter!(pd, x, td; color=col, label="t_dep, g=$g", ms=4)
    plot!(pd, x, a .+ x ./ (2κ(g)) .- mean(x) / (2κ(g)) .+ mean(td) .- a; color=col, ls=:dash,
          label="slope 1/2κ = $(round(1/(2κ(g)), digits=3))")
    scatter!(pd, x, tF; color=col, marker=:utriangle, label="t_F (F<½)", ms=3, alpha=0.6)
    @printf(fits, "  vortex g=%.1f  fitted slope b = %.3f   Bogoliubov 1/(2κ) = %.3f   t_F = %s\n",
            g, b, 1 / (2κ(g)), string(round.(tF, digits=2)))
end

# (e) imbalance: regular vs chaotic, t_dep vs N (log-log)
pe = plot(title="(e) Imbalanced trimer: t_dep vs N", xlabel="N", ylabel="t_dep",
          xscale=:log10, yscale=:log10, legend=:topleft)
for (g, col) in ((4.0, :seagreen), (8.0, :teal), (14.0, :orange), (18.0, :red))
    idx = rowsel("imb", g)
    isempty(idx) && continue
    N = Float64.(S["N"][idx]); td = Float64.(S["t_depl10"][idx]); λ = Float64(S["lyapunov_gp"][idx[1]])
    ok = .!isnan.(td)
    count(ok) < 2 && continue
    β = cov(log.(N[ok]), log.(td[ok])) / var(log.(N[ok]))
    scatter!(pe, N[ok], td[ok]; color=col, label=@sprintf("g=%g (λ_GP=%.2f), β=%.2f", g, λ, β), ms=4)
    plot!(pe, N[ok], td[ok]; color=col, label="", alpha=0.5)
    @printf(fits, "  imb    g=%4.1f  λ_GP = %.3f   t_dep ∝ N^β with β = %.3f   (t_dep = %s)\n",
            g, λ, β, string(round.(td, digits=2)))
end
NN2 = [10.0, 160.0]
plot!(pe, NN2, 3.85 .* sqrt.(NN2 ./ 10); color=:gray, ls=:dot, label="∝ √N")

# (f) late-time average phase of the exact state in three representations
pf = plot(title="(f) Average phase of exact ψ̃, t-average over [0, 10]", xlabel="N",
          xscale=:log10, ylabel="⟨|Σψ̃|/Σ|ψ̃|⟩", ylims=(0, 1.02), legend=:outerbottom,
          legend_columns=2)
for (case, g, col) in (("vortex", 1.0, :blue), ("vortex", 4.0, :red), ("imb", 4.0, :green), ("imb", 18.0, :orange))
    NN = Ns(case, g)
    isempty(NN) && continue
    av(key) = [mean(load(case, g, N)[key][1:201]) for N in NN]
    plot!(pf, NN, av("sign_a0"); color=col, marker=:circle, label="$case g=$g, GP phase gauge")
    plot!(pf, NN, av("sign_unguided"); color=col, marker=:x, ls=:dot, label="$case g=$g, unguided")
end

fig = plot(pa, pb, pc, pd, pe, pf; layout=(3, 2), size=(1150, 1200), margin=4Plots.mm,
           plot_title="Stage 2a: mean-field break time in the Bose-Hubbard trimer (exact, N ≤ 160)")
savefig(fig, joinpath(OUT, "stage2a_breaktime.png"))
txt = String(take!(fits))
write(joinpath(OUT, "stage2a_fits.txt"), txt)
print(txt)
