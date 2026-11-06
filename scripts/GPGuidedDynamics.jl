# GPGuidedDynamics.jl
#
# Time-dependent Gross-Pitaevskii (GP) importance sampling for RimuRealTime.jl.
#
# Theory (ideation.tex, Eqs. 5 and 9): with a guiding function φ(x,t) and ψ̃ = φψ,
#
#     ∂t ψ̃ = [∂t φ/φ − i H̃(t)] ψ̃ ≡ −i K(t) ψ̃,     H̃_xy = φ(x,t)/φ(y,t) H_xy,
#     K_xy = H̃_xy + δ_xy · i ∂t log φ(x,t).
#
# Here φ = conj(Ψ_GP(t)), where Ψ_GP is Gutzwiller.jl's GrossPitaevskiiAnsatz
#     ⟨n|Ψ_GP⟩ = sqrt(N!/∏ n_m!) ∏ c_m^{n_m},
# and the orbital c(t) obeys the discrete time-dependent GP (Hartree) equation, i.e. the
# Dirac-Frenkel variational dynamics of the N-boson product state:
#     i ċ = h1 c + U (N−1) |c|² c.
# With this choice ψ̃ = conj(Ψ_GP)·ψ ≈ |ψ|² while mean field is accurate, so the
# transformed amplitudes are close to real-positive (reduced phase problem), and
#     ∑_x ψ̃(x) = ⟨Ψ_GP(t)|ψ(t)⟩   (the mean-field overlap, for free).
#
# The GP orbital lives in a mutable `GPOrbital` that the Hamiltonian reads live, so the
# immutable RimuRealTime state can carry a time-dependent K(t). The driver `run_lockstep!`
# moves each orbital to the time at which the integrator evaluates K:
#   PEC   (trapezoidal corrector, x_{n+1} = K w_{n+1}) → t_{n+1}
#   Exact (Krylov exp(−iK dt))                         → t_n + dt/2 (exponential midpoint)

module GPGuidedDynamics

using LinearAlgebra
using SparseArrays
using Printf
using Rimu
using Rimu.Hamiltonians: ModifiedHamiltonian
import Rimu.Hamiltonians: parent_operator, modify_diagonal, modify_offdiagonal
using RimuRealTime
using Gutzwiller: GrossPitaevskiiAnsatz

export ring_hopping, GPOrbital, advance_gp!, gp_energy, gp_rhs
export TDGPGuided, guide_value, guide_fn, reweight!
export DenseBasis, dense, one_body_ops, obdm, gp_vector
export check_gp_consistency, Run, run_lockstep!, lyapunov_gp

# ---------------------------------------------------------------------------------------
# single-particle matrix
# ---------------------------------------------------------------------------------------
"""
    ring_hopping(M, t; potential=zeros(M))

One-body matrix of `HubbardReal1D` / `HubbardReal1DEP` on an M-site ring:
h[i, i±1] = −t (periodic), h[i,i] = potential[i]. M ≥ 3.
"""
function ring_hopping(M, t; potential=zeros(M))
    M ≥ 3 || throw(ArgumentError("use M ≥ 3 (M = 2 double-counts the ring bond)"))
    h = zeros(Float64, M, M)
    for i in 1:M
        j = mod1(i + 1, M)
        h[i, j] -= t
        h[j, i] -= t
        h[i, i] = potential[i]
    end
    return h
end

# ---------------------------------------------------------------------------------------
# time-dependent GP orbital
# ---------------------------------------------------------------------------------------
"""
    GPOrbital(c0, h1, u, N; maxdt=1e-3)

Mutable discrete GP orbital c(t) (normalised to 1) with i ċ = h1 c + u(N−1)|c|²c,
integrated with RK4 sub-steps of at most `maxdt`. Caches `cconj = conj(c)` and
`lrate = ċ ./ c`, so that ∂t log Ψ_GP(x) = ∑_m n_m lrate[m].
"""
mutable struct GPOrbital
    c::Vector{ComplexF64}
    h1::Matrix{Float64}
    g::Float64          # u (N − 1)
    u::Float64
    N::Int
    t::Float64
    maxdt::Float64
    cconj::Vector{ComplexF64}
    lrate::Vector{ComplexF64}
end

gp_rhs(c, h1, g) = -im .* (h1 * c .+ g .* abs2.(c) .* c)

function GPOrbital(c0, h1, u, N; maxdt=1e-3)
    c = ComplexF64.(c0)
    c ./= norm(c)
    gp = GPOrbital(c, Matrix{Float64}(h1), u * (N - 1), u, N, 0.0, maxdt,
                   similar(c), similar(c))
    return refresh!(gp)
end

function refresh!(gp::GPOrbital)
    any(iszero, gp.c) && error("GP orbital has a zero component; guide undefined")
    gp.cconj .= conj.(gp.c)
    gp.lrate .= gp_rhs(gp.c, gp.h1, gp.g) ./ gp.c
    return gp
end

"Advance the orbital to `t_target` (forward or backward) with RK4."
function advance_gp!(gp::GPOrbital, t_target)
    Δ = t_target - gp.t
    abs(Δ) < 1e-14 && return gp
    n = max(1, ceil(Int, abs(Δ) / gp.maxdt))
    h = Δ / n
    c = gp.c
    for _ in 1:n
        k1 = gp_rhs(c, gp.h1, gp.g)
        k2 = gp_rhs(c .+ (h / 2) .* k1, gp.h1, gp.g)
        k3 = gp_rhs(c .+ (h / 2) .* k2, gp.h1, gp.g)
        k4 = gp_rhs(c .+ h .* k3, gp.h1, gp.g)
        c = c .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    gp.c = c
    gp.t = t_target
    return refresh!(gp)
end

"Mean-field energy ⟨Ψ_GP|H|Ψ_GP⟩ = N c†h1c + (u/2)N(N−1)∑|c|⁴ (conserved by GP)."
gp_energy(gp::GPOrbital) =
    gp.N * real(dot(gp.c, gp.h1 * gp.c)) + gp.u / 2 * gp.N * (gp.N - 1) * sum(abs2.(gp.c) .^ 2)

"""
    lyapunov_gp(c0, h1, u, N; tmax, dt=1e-3, d0=1e-8, renorm=0.1)

Largest Lyapunov exponent of the GP trajectory from `c0` (two-trajectory method with
periodic renormalisation, perturbation kept on the unit sphere). Returns (λ, ts, λ(t)).
"""
function lyapunov_gp(c0, h1, u, N; tmax, dt=1e-3, d0=1e-8, renorm=0.1, seed=7)
    a = GPOrbital(c0, h1, u, N; maxdt=dt)
    pert = ComplexF64.(c0) ./ norm(c0)
    # deterministic pseudo-random direction (no RNG dependency)
    dirv = ComplexF64[sin(seed * k + 0.3) + im * cos(1.7 * seed * k) for k in eachindex(pert)]
    dirv .-= dot(pert, dirv) .* pert         # orthogonal to c: not a global-phase/norm mode
    b = GPOrbital(pert .+ d0 .* dirv ./ norm(dirv), h1, u, N; maxdt=dt)
    d0 = norm(b.c .- a.c)
    sumlog = 0.0
    ts = Float64[]
    lam = Float64[]
    t = 0.0
    while t < tmax - 1e-12
        t += renorm
        advance_gp!(a, t)
        advance_gp!(b, t)
        # remove relative global phase (U(1) symmetry direction is neutral)
        ph = dot(a.c, b.c)
        bc = b.c .* (conj(ph) / abs(ph))
        d = norm(bc .- a.c)
        sumlog += log(d / d0)
        b.c = a.c .+ (d0 / d) .* (bc .- a.c)
        b.c ./= norm(b.c)
        refresh!(b)
        push!(ts, t)
        push!(lam, sumlog / t)
    end
    return lam[end], ts, lam
end

# ---------------------------------------------------------------------------------------
# time-dependent GP-guided Hamiltonian  K(t) = H̃(t) + i ∂t log φ
# ---------------------------------------------------------------------------------------
"""
    guide_fn(ansatz, c, α, addr)

Guiding function of the α-family  φ_α(x) = conj(Ψ(x)) |Ψ(x)|^(α−1) = |Ψ(x)|^α e^{−iθ(x)},
Ψ(x) = ⟨x|Ψ_GP(c)⟩. α = 1: full GP importance sampling (ψ̃ ≈ |ψ|²);
α = 0: pure GP-phase gauge transformation (unitary, ψ̃ ≈ |ψ|).
"""
function guide_fn(ansatz, c, α, addr)
    Ψ = ansatz(addr, c)
    return α == 1 ? conj(Ψ) : conj(Ψ) * abs(Ψ)^(α - 1)
end

"""
    TDGPGuided(H, gp::GPOrbital; alpha=1.0, dt_term=true) <: ModifiedHamiltonian{ComplexF64}

Similarity-transformed generator for ψ̃ = φ_α ψ (see [`guide_fn`](@ref)):
off-diagonal  H̃_xy = φ(x)/φ(y) H_xy,
diagonal      H_xx + i ∂t log φ(x) = H_xx + i α Re L(x) + Im L(x),  L = ∑ n_m ċ_m/c_m.
`dt_term=false` gives the frozen-guide H̃ used by the exact reweighting scheme.
Non-Hermitian for α ≠ 0 → `LOStructure` stays `AdjointUnknown` (Krylov uses Arnoldi).
"""
struct TDGPGuided{H,G} <: ModifiedHamiltonian{ComplexF64}
    hamiltonian::H
    ansatz::G
    gp::GPOrbital
    alpha::Float64
    dt_term::Bool
end
function TDGPGuided(H, gp::GPOrbital; alpha=1.0, dt_term=true)
    ansatz = GrossPitaevskiiAnsatz(starting_address(H); valtype=ComplexF64)
    return TDGPGuided(H, ansatz, gp, Float64(alpha), dt_term)
end
Base.show(io::IO, h::TDGPGuided) =
    print(io, "TDGPGuided(", h.hamiltonian, "; alpha=", h.alpha, ", dt_term=", h.dt_term, ")")

parent_operator(h::TDGPGuided) = h.hamiltonian

"φ_α(x, t) for the current orbital"
guide_value(h::TDGPGuided, addr) = guide_fn(h.ansatz, h.gp.c, h.alpha, addr)

function modify_offdiagonal(h::TDGPGuided, src, dst, value)
    return dst => value * (guide_value(h, dst) / guide_value(h, src))
end

function modify_diagonal(h::TDGPGuided, addr, value)
    h.dt_term || return ComplexF64(value)
    L = zero(ComplexF64)
    for (n, m, _) in occupied_modes(addr)
        L += n * h.gp.lrate[m]
    end
    return value + im * h.alpha * real(L) + imag(L)
end

"""
    reweight!(dv, ansatz, c_old, c_new, α)

Exact guide update ψ̃(x) ← φ_new(x)/φ_old(x) · ψ̃(x). Together with a step under the
frozen H̃(t) this integrates Eq. (5) without splitting error, because
D e^{−iH dt} D⁻¹ = e^{−i D H D⁻¹ dt}.
"""
function reweight!(dv, ansatz, c_old, c_new, α)
    for (k, v) in collect(pairs(dv))
        dv[k] = v * (guide_fn(ansatz, c_new, α, k) / guide_fn(ansatz, c_old, α, k))
    end
    return dv
end

# ---------------------------------------------------------------------------------------
# dense-basis tools for measurement and exact reference
# ---------------------------------------------------------------------------------------
struct DenseBasis{A}
    basis::Vector{A}
    index::Dict{A,Int}
    onrs::Vector{Vector{Int}}
    M::Int
    N::Int
end
function DenseBasis(basis::AbstractVector)
    b = collect(basis)
    index = Dict(a => i for (i, a) in enumerate(b))
    onrs = Vector{Int}[collect(Int, Tuple(onr(a))) for a in b]
    return DenseBasis(b, index, onrs, length(first(onrs)), sum(first(onrs)))
end

function dense(db::DenseBasis, v)
    x = zeros(ComplexF64, length(db.basis))
    for (k, val) in pairs(v)
        x[db.index[k]] = val
    end
    return x
end

"Sparse matrices of a†_i a_j (i ≠ j) on the dense basis."
function one_body_ops(db::DenseBasis)
    n = length(db.basis)
    key = Dict(o => i for (i, o) in enumerate(db.onrs))
    ops = Dict{Tuple{Int,Int},SparseMatrixCSC{Float64,Int}}()
    for i in 1:db.M, j in 1:db.M
        i == j && continue
        I = Int[]; J = Int[]; V = Float64[]
        for (col, o) in enumerate(db.onrs)
            o[j] == 0 && continue
            y = copy(o)
            amp = sqrt(o[j] * (o[i] + 1))
            y[j] -= 1
            y[i] += 1
            push!(I, key[y]); push!(J, col); push!(V, amp)
        end
        ops[(i, j)] = sparse(I, J, V, n, n)
    end
    return ops
end

"One-body density matrix ρ_ij = ⟨a†_i a_j⟩ of a normalised dense vector."
function obdm(ψ, db::DenseBasis, ops)
    ρ = zeros(ComplexF64, db.M, db.M)
    w = abs2.(ψ)
    for i in 1:db.M
        ρ[i, i] = sum(w[k] * db.onrs[k][i] for k in eachindex(ψ))
    end
    for ((i, j), A) in ops
        ρ[i, j] = dot(ψ, A * ψ)
    end
    return ρ
end

"Dense vector ⟨x|Ψ_GP⟩ for the current orbital."
function gp_vector(db::DenseBasis, ansatz, gp::GPOrbital)
    return ComplexF64[ansatz(a, gp.c) for a in db.basis]
end

"""
    check_gp_consistency(H, h1, u, db, Hdense; c=random) -> (E_rimu, E_gp)

Checks that `h1`, `u` reproduce ⟨Ψ_GP|H|Ψ_GP⟩ for a generic complex orbital.
"""
function check_gp_consistency(H, h1, u, db::DenseBasis, Hdense)
    M = db.M
    c = ComplexF64[(1 + 0.3k) * cis(0.7k^2) for k in 1:M]
    gp = GPOrbital(c, h1, u, db.N)
    ansatz = GrossPitaevskiiAnsatz(starting_address(H); valtype=ComplexF64)
    Ψ = gp_vector(db, ansatz, gp)
    E_rimu = real(dot(Ψ, Hdense * Ψ)) / real(dot(Ψ, Ψ))
    return E_rimu, gp_energy(gp), norm(Ψ)
end

# ---------------------------------------------------------------------------------------
# lock-step driver
# ---------------------------------------------------------------------------------------
"""
    Run(label, sim; scheme=:none, gp=nothing, alpha=1.0, midpoint=false)

One RimuRealTime simulation in the lock-step driver.
* `scheme = :none`      — unguided.
* `scheme = :generator` — K(t) = H̃(t) + i∂t log φ (H built with `dt_term=true`); the
  orbital is moved to t_{n+1} (PEC) or t_n + dt/2 (`midpoint=true`, Krylov) before `step!`.
* `scheme = :reweight`  — frozen H̃(t_n) (H built with `dt_term=false`), then the exact
  guide update [`reweight!`](@ref) of the state (and PEC's stored H·w vector).
"""
mutable struct Run
    label::String
    sim::Any
    scheme::Symbol
    gp::Union{Nothing,GPOrbital}
    alpha::Float64
    midpoint::Bool
    rows::Vector{Any}
end
function Run(label, sim; scheme=:none, gp=nothing, alpha=1.0, midpoint=false)
    scheme in (:none, :generator, :reweight) || throw(ArgumentError("unknown scheme $scheme"))
    return Run(label, sim, scheme, gp, Float64(alpha), midpoint, Any[])
end
isguided(r::Run) = r.scheme !== :none

statevec(r::Run) = r.sim.state.single_states[1].state_vector

"""
    run_lockstep!(runs, db, ansatz, gp_meas, ψexact; dt, nsteps, every, obdm_every, ops)

Steps all runs together. After every `every` steps measures, for each run,
back-transformed observables vs the exact reference `ψexact(t)::Vector`.
"""
function run_lockstep!(runs, db, ansatz, gp_meas::GPOrbital, ψexact;
                       dt, nsteps, every=1, obdm_every=10, ops=one_body_ops(db),
                       log_every=max(1, nsteps ÷ 10))
    N = db.N
    measure!(runs, db, ansatz, gp_meas, ψexact, 0, 0.0, ops, true)
    wall = time()
    for n in 1:nsteps
        t_prev = (n - 1) * dt
        t_next = n * dt
        for r in runs
            if r.scheme === :generator
                advance_gp!(r.gp, r.midpoint ? t_prev + dt / 2 : t_next)
                step!(r.sim)
            elseif r.scheme === :reweight
                step!(r.sim)                       # frozen guide at t_prev
                c_old = copy(r.gp.c)
                advance_gp!(r.gp, t_next)
                s = r.sim.state.single_states[1]
                reweight!(s.state_vector, r.sim.state.hamiltonian.ansatz, c_old, r.gp.c, r.alpha)
                if hasproperty(s, :h_predictor_old)
                    reweight!(s.h_predictor_old, r.sim.state.hamiltonian.ansatz, c_old, r.gp.c, r.alpha)
                end
            else
                step!(r.sim)
            end
        end
        advance_gp!(gp_meas, t_next)
        if n % every == 0
            do_obdm = (n ÷ every) % obdm_every == 0
            measure!(runs, db, ansatz, gp_meas, ψexact, n, t_next, ops, do_obdm)
        end
        if n % log_every == 0
            @printf("  step %6d  t = %7.3f  wall = %7.1f s\n", n, t_next, time() - wall)
            flush(stdout)
        end
    end
    return runs
end

function measure!(runs, db, ansatz, gp_meas, ψexact, n, t, ops, do_obdm)
    N = db.N
    Ψ = gp_vector(db, ansatz, gp_meas)       # ⟨x|Ψ_GP(t)⟩, normalised
    ψe = ψexact(t)
    ψe ./= norm(ψe)
    for r in runs
        v = statevec(r)
        raw = dense(db, v)
        ψ = isguided(r) ? raw ./ (conj.(Ψ) .* abs.(Ψ) .^ (r.alpha - 1)) : raw
        nψ = norm(ψ)
        ψn = ψ ./ nψ
        l1 = sum(abs, raw)
        row = Dict{String,Any}(
            "step" => n, "time" => t,
            "len" => length(v),
            "l1_raw" => l1,
            "l2_raw" => norm(raw),
            "norm2_phys" => nψ,
            "sign_raw" => abs(sum(raw)) / l1,
            "fid_exact" => abs2(dot(ψe, ψn)),
            "fgp" => abs2(dot(Ψ, ψn)),
            "fgp_exact" => abs2(dot(Ψ, ψe)),
            "gp_energy" => gp_energy(gp_meas),
        )
        for i in 1:db.M
            row["n$(i)"] = sum(abs2(ψn[k]) * db.onrs[k][i] for k in eachindex(ψn))
            row["n$(i)_gp"] = N * abs2(gp_meas.c[i])
        end
        if do_obdm
            ρ = obdm(ψn, db, ops)
            row["cfrac"] = eigmax(Hermitian(ρ)) / N
            ρe = obdm(ψe, db, ops)
            row["cfrac_exact"] = eigmax(Hermitian(ρe)) / N
        else
            row["cfrac"] = NaN
            row["cfrac_exact"] = NaN
        end
        push!(r.rows, row)
    end
end

end # module
