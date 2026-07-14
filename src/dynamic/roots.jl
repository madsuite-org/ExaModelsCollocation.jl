# Roots of Gauss-Jacobi polynomials are interpolation points within each interval
# (L. T. Biegler, Nonlinear Programming, Theorem 10.1)

"""
    AbstractRoots

Abstract type for collocation-point families. A concrete subtype selects where the `K`
interior collocation points τ₁,…,τ_K ∈ (0,1] sit within each finite element and
supplies the associated quadrature weights.
"""
abstract type AbstractRoots end

"""
    GaussRadau()

Radau collocation points: `K` roots including the right endpoint `τ_K = 1`, giving
`2K-1` order and stiff-accurate (L-stable) behavior. Default `roots` for [`add_dae`](@ref).
"""
struct GaussRadau <: AbstractRoots end

"""
    GaussLegendre()

Legendre (Gauss) collocation points: `K` interior roots in `(0,1)`, giving the maximal
`2K` order but not stiff-accurate.
"""
struct GaussLegendre <: AbstractRoots end

"""
    GaussLobatto()

Lobatto collocation points: `K` roots including both endpoints `τ = 0` and `τ = 1`.
"""
struct GaussLobatto <: AbstractRoots end

function _get_roots(::AbstractRoots, degree::T)::Vector{T} where {T <: Real}

    return taus
end