# Interpolating polynomial used to represent the differential state
# across each interval, evaluated at the collocation points.

"""
    AbstractPolynomial

Abstract type for the interpolating polynomial family used within each interval.
A concrete subtype supplies the basis functions and the weights for the
colocation and continuity constraints.
"""
abstract type AbstractPolynomial end

"""
    Lagrange()

Lagrange interpolation polynomials.
"""
struct Lagrange <: AbstractPolynomial end

"""
    dljk(::Lagrange, nodes) -> Matrix

`[dℓⱼ(τₖ)]`: derivative of Lagrange basis `j` (over `nodes = [0, τ₁..τ_K]`) at each
collocation point `τₖ`. Rows `k = 1..K`, columns `j = 0..K`; the `K × (K+1)` `StateForm` `A`.
"""
function dljk(::Lagrange, nodes)
    error("dljk (StateForm collocation matrix) not implemented") # TODO: barycentric differentiation matrix
end

"""
    lj1(::Lagrange, nodes) -> Vector

`[ℓⱼ(1)]`: Lagrange basis `j` (over `nodes = [0, τ₁..τ_K]`) at the right endpoint `τ = 1`;
the length-`K+1` `StateForm` `b`. `= δⱼ_K` when a collocation point sits at `τ = 1` (Radau/Lobatto).
"""
function lj1(::Lagrange, nodes)
    error("lj1 (StateForm endpoint weights) not implemented") # TODO: ℓⱼ(1) via barycentric interpolation
end

"""
    omegajk(::Lagrange, roots) -> Matrix

`[Ωⱼ(τₖ)]` with `Ωⱼ(τ) = ∫₀^τ ℓ̄ⱼ` over the collocation points `roots = τ₁..τ_K`;
the `K × K` `DerivativeForm` `A`.
"""
function omegajk(::Lagrange, roots)
    error("omegajk (DerivativeForm collocation matrix) not implemented") # TODO: integrated basis at τₖ
end

"""
    omegaj1(::Lagrange, roots) -> Vector

`[Ωⱼ(1)]`: the integrated basis at `τ = 1`; the length-`K` `DerivativeForm` `b`.
"""
function omegaj1(::Lagrange, roots)
    error("omegaj1 (DerivativeForm endpoint weights) not implemented") # TODO: integrated basis at 1
end
