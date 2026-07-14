# ----- COMPLETE 7/13/2026 -----
# Interpolation points (taus) within each interval are the 
# roots of Gauss-Jacobi polynomials
# (L. T. Biegler, Nonlinear Programming, Theorem 10.1)

"""
    AbstractRoots

Abstract type for collocation point families. A concrete subtype contains the `K+1`
interpolation points tau_0 = 0, tau_j in (0,1], for j = 1,...,K.
"""
abstract type AbstractRoots end

"""
    GaussRadau()

Roots of the Gauss-Radau polynomial as collocation points. Default `roots` for [`add_dae`](@ref).
"""
struct GaussRadau <: AbstractRoots end

"""
    GaussLegendre()

Roots of the Gauss-Legendre polynomial as collocation points.
"""
struct GaussLegendre <: AbstractRoots end

"""
    GaussLotto()

Roots of the Gauss-Lobatto polynomial as collocation points.
"""
struct GaussLobatto <: AbstractRoots end

function _get_taus(family::AbstractRoots, K::Integer)
    # Calculate roots of Gauss-Jacobi polynomials (from FastGaussQuadrature.jl)
    roots = _get_roots(family, K) # in [-1, 1]

    # Shift from [-1,1] to [0,1]
    taus = (roots .+ 1) ./ 2

    # Append tau0 = 0
    iszero(first(taus)) || pushfirst!(taus, zero(eltype(taus)))

    return taus
end

_get_roots(::GaussRadau,    K::Integer) = -reverse(FastGaussQuadrature.gaussradau(K)[1])
_get_roots(::GaussLegendre, K::Integer) = FastGaussQuadrature.gausslegendre(K)[1]
_get_roots(::GaussLobatto,  K::Integer) = 
    K >= 2 ? FastGaussQuadrature.gaussLobatto(K+1)[1] : error("GaussLobatto requires K ≥ 2")