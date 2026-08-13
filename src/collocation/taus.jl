# The collocation points are roots of Gauss-Jacobi polynomials
# (L. T. Biegler, Nonlinear Programming, Theorem 10.1):
# default = GaussRadau()
#   GaussRadau()    : alpha = 1, beta = 0 (default; includes tau = 1)
#   GaussLegendre() : alpha = 0, beta = 0 (tau_j in (0,1))
#   GaussLobatto()  : alpha = 1, beta = 1 (includes both tau = 0 and tau = 1)
"""
    AbstractRoots

Abstract type for collocation point families. A concrete subtype contains the `K` true
collocation points tau_j, for j = 1,…,K. tau_0 = 0 is added per basis in basis.jl.
"""
abstract type AbstractRoots end

"""
    GaussRadau()

Roots of the Gauss-Radau polynomial as collocation points. The default `roots` for
[`CollocationExaCore`](@ref).
"""
struct GaussRadau <: AbstractRoots end

"""
    GaussLegendre()

Roots of the Gauss-Legendre polynomial as collocation points.
"""
struct GaussLegendre <: AbstractRoots end

"""
    GaussLobatto()

Roots of the Gauss-Lobatto polynomial as collocation points.
"""
struct GaussLobatto <: AbstractRoots end

# Calculate taus for j=1,...,K
function _get_taus(family::AbstractRoots, K::Integer)
    # Roots of the Gauss-Jacobi polynomial, from FastGaussQuadrature.jl
    roots = _get_roots(family, K) # in [-1, 1]

    # Shift from [-1,1] to [0,1]
    taus = (roots .+ 1) ./ 2
    
    return taus
end

_get_roots(::GaussRadau,    K::Integer) = -reverse(FastGaussQuadrature.gaussradau(K)[1])
_get_roots(::GaussLegendre, K::Integer) = FastGaussQuadrature.gausslegendre(K)[1]
_get_roots(::GaussLobatto,  K::Integer) = FastGaussQuadrature.gausslobatto(K)[1]