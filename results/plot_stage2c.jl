# plot_stage2c.jl
#
# STAGE 2c figure: mean-field fidelity and depletion across the dynamical instability of
# the trimer vortex (data/stage2c_threshold, produced on ctcp by scripts/stage2c_threshold.jl).
#
# Two-mode Bogoliubov theory (k0 = 2π/3, pair channel {0, −2π/3}):
#   A = −3/2 + g/3,  B = g/3,  Ω² = A² − B² = g_c − g,  g_c = 9/4   (J = 1)
#   F(t) = 1/(1 + (B/Ω)² sin²Ωt),   F̄_∞ = Ω/|A| ∝ (g_c − g)^{1/2}
#   N·depletion_max = 2B²/Ω² = 2g²/[9(g_c − g)]   (diverges as (g_c − g)^{−1})
#   above g_c: growth rate κ = √(g − g_c), t_dep ≈ ln N / (2κ) + const.
#
# Run from the repo root:   julia --project=devrimu results/plot_stage2c.jl
# Output: results/stage2c_threshold.png, results/stage2c_fits.txt

ENV["GKSwstype"] = "100"
using DelimitedFiles
using Plots
using Printf
using Statistics

const ROOT = dirname(@__DIR__)
const DATA = joinpath(ROOT, "data", "stage2c_threshold")
const OUT = @__DIR__
const GC = 9 / 4

raw, hdr = readdlm(joinpath(DATA, "summary.csv"), ',', Float64; header=true)
S = Dict(String(h) => raw[:, j] for (j, h) in enumerate(vec(hdr)))
gs = sort(unique(S["g"])); Ns = sort(unique(Int.(S["N"])))
sel(g, N) = only(findall(i -> isapprox(S["g"][i], g) && S["N"][i] == N, eachindex(S["g"])))
has(g, N) = any(i -> isapprox(S["g"][i], g) && S["N"][i] == N, eachindex(S["g"]))
function run(g, N)
    r, h = readdlm(joinpath(DATA, @sprintf("vortex_g%.2f_N%d.csv", g, N)), ',', Float64; header=true)
    return Dict(String(x) => r[:, j] for (j, x) in enumerate(vec(h)))
end

A(g) = -1.5 + g / 3
B(g) = g / 3
Fbar_inf(g) = g < GC ? sqrt(GC - g) / abs(A(g)) : 0.0
function FB(g, t)
    Ω2 = GC - g
    s = Ω2 > 0 ? sin(sqrt(Ω2) * t) / sqrt(Ω2) : Ω2 < 0 ? sinh(sqrt(-Ω2) * t) / sqrt(-Ω2) : t
    return 1 / (1 + B(g)^2 * s^2)
end
Ndep_max(g) = 2 * B(g)^2 / (GC - g)
tw = 0:0.05:80
Fbar_window(g) = mean(FB.(g, tw))

default(lw=1.8, legendfontsize=7, guidefontsize=9, titlefontsize=10, framestyle=:box)
pal = cgrad(:viridis, length(Ns); categorical=true)
fits = IOBuffer()
println(fits, "Stage 2c: trimer vortex across the dynamical instability (g_c = 9/4)")

# (a) time-averaged fidelity vs g
gg = range(0.9, 4.1; length=400)
pa = plot(title="(a) Time-averaged mean-field fidelity F̄ (t ≤ 80)", xlabel="g = u(N−1)/J",
          ylabel="F̄", ylims=(0, 1.02), legend=:topright)
plot!(pa, gg, Fbar_inf.(gg); color=:red, lw=2.5, label="Bogoliubov F̄_∞ = Ω/|A|")
plot!(pa, gg, Fbar_window.(gg); color=:red, ls=:dash, label="Bogoliubov, same window")
for (i, N) in enumerate(Ns)
    gN = [g for g in gs if has(g, N)]
    scatter!(pa, gN, [S["Fbar_exact"][sel(g, N)] for g in gN]; color=pal[i], label="N=$N", ms=4)
end
vline!(pa, [GC]; color=:gray, ls=:dot, label="g_c = 9/4")

# (b) F(t) just below threshold
gb = 2.2
pb = plot(title="(b) F_GP(t) at g = $gb (Ω = $(round(sqrt(GC-gb), digits=3)))", xlabel="t",
          ylims=(0, 1.02), legend=:topright)
for (i, N) in enumerate(Ns)
    has(gb, N) || continue
    d = run(gb, N)
    plot!(pb, d["time"], d["F_exact"]; color=pal[i], label="N=$N")
end
plot!(pb, tw, FB.(gb, tw); color=:red, ls=:dash, lw=2, label="Bogoliubov")

# (c) finite-N approach to Bogoliubov: |F̄(N) − F̄_B(window)| vs N
pc = plot(title="(c) Finite-N deviation |F̄ − F̄_Bogoliubov|", xlabel="N", xscale=:log10,
          yscale=:log10, legend=:bottomleft)
gcol = cgrad(:plasma, length(gs); categorical=true)
for (k, g) in enumerate(gs)
    g < GC || continue
    NN = [N for N in Ns if has(g, N)]
    dev = [abs(S["Fbar_exact"][sel(g, N)] - Fbar_window(g)) for N in NN]
    ok = dev .> 0
    count(ok) ≥ 2 || continue
    p = cov(log.(NN[ok]), log.(dev[ok])) / var(log.(NN[ok]))
    plot!(pc, NN[ok], dev[ok]; marker=:circle, color=gcol[k], label=@sprintf("g=%.2f (∝N^%.2f)", g, p))
    @printf(fits, "  g=%.2f  |F̄−F̄_B| ∝ N^%.3f   values %s\n", g, p, string(round.(dev, sigdigits=3)))
end

# (d) N × max depletion vs g: divergence at g_c, rounded at finite N
pd = plot(title="(d) N × max depletion", xlabel="g", yscale=:log10, legend=:topleft)
gl = range(0.9, GC - 0.01; length=300)
plot!(pd, gl, Ndep_max.(gl); color=:red, lw=2.5, label="Bogoliubov 2g²/[9(g_c−g)]")
for (i, N) in enumerate(Ns)
    gN = [g for g in gs if has(g, N)]
    plot!(pd, gN, [N * S["max_depletion"][sel(g, N)] for g in gN]; marker=:circle,
          color=pal[i], label="N=$N")
end
vline!(pd, [GC]; color=:gray, ls=:dot, label="")

# (e) scaling collapse: max depletion vs x = (g − g_c) N
pe = plot(title="(e) Collapse: max depletion vs (g − g_c)·N", xlabel="(g − g_c) N",
          ylabel="max depletion", yscale=:log10, legend=:bottomright, xlims=(-120, 120))
for (i, N) in enumerate(Ns)
    gN = [g for g in gs if has(g, N)]
    scatter!(pe, [(g - GC) * N for g in gN], [S["max_depletion"][sel(g, N)] for g in gN];
             color=pal[i], label="N=$N", ms=4)
end
xs = range(-120, -1; length=200)
plot!(pe, xs, 2 * GC^2 / 9 ./ abs.(xs); color=:red, ls=:dash, label="Bogoliubov (9/8)/|x|")

# (f) unstable side: t_dep vs ln N, Bogoliubov slope 1/(2κ)
pf = plot(title="(f) Unstable side: t_dep (10%) vs ln N", xlabel="ln N", ylabel="t_dep",
          legend=:topleft)
for (k, g) in enumerate(gs)
    g > GC || continue
    NN = [N for N in Ns if has(g, N)]
    td = [S["t_depl10"][sel(g, N)] for N in NN]
    ok = .!isnan.(td)
    count(ok) ≥ 3 || continue
    x = log.(NN[ok]); y = td[ok]
    b = cov(x, y) / var(x)
    κ = sqrt(g - GC)
    scatter!(pf, x, y; color=gcol[k], label=@sprintf("g=%.2f: fit %.2f, 1/2κ=%.2f", g, b, 1 / (2κ)), ms=4)
    plot!(pf, x, mean(y) .+ (x .- mean(x)) ./ (2κ); color=gcol[k], ls=:dash, label="")
    @printf(fits, "  g=%.2f  t_dep slope vs ln N: fit %.3f, Bogoliubov 1/(2κ) = %.3f\n", g, b, 1 / (2κ))
end

fig = plot(pa, pb, pc, pd, pe, pf; layout=(3, 2), size=(1150, 1200), margin=4Plots.mm,
           plot_title="Stage 2c: mean-field fidelity across the vortex instability (trimer, exact, N ≤ 320)")
savefig(fig, joinpath(OUT, "stage2c_threshold.png"))
for g in gs
    NN = [N for N in Ns if has(g, N)]
    @printf(fits, "  g=%.2f  F̄(N=%s) = %s ; Bogoliubov window %.4f, ∞ %.4f\n", g, string(NN),
            string(round.([S["Fbar_exact"][sel(g, N)] for N in NN], digits=4)), Fbar_window(g), Fbar_inf(g))
end
txt = String(take!(fits))
write(joinpath(OUT, "stage2c_fits.txt"), txt)
print(txt)
