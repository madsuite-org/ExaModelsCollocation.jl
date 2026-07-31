# Step 1 of the mode: where inside a reference interval [0,1] the equations are enforced.
# The collocation points are roots of Gauss-Jacobi polynomials
# (L. T. Biegler, Nonlinear Programming, Theorem 10.1):
#   GaussRadau()    : alpha = 1, beta = 0   (default; includes tau = 1)
#   GaussLegendre() : alpha = 0, beta = 0
#   GaussLobatto()  : alpha = 1, beta = 1   (includes both tau = 0 and tau = 1)
"""
    AbstractRoots

Abstract type for collocation point families. A concrete subtype contains the `K` true
collocation points tau_j, for j = 1,...,K. The tau0 = 0 anchor is added per basis in basis.jl
(StateForm interpolates it, DerivativeForm treats it as an actual collocation point).
"""
abstract type AbstractRoots end

"""
    GaussRadau()

Roots of the Gauss-Radau polynomial as collocation points. The default `roots` for
[`CollocationExaCore`](@ref); `tau_K = 1`, so the last collocation point sits on the interval's right edge.
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

# K is validated in _set_mesh before this is reached
function _get_taus(family::AbstractRoots, K::Integer)
    # Roots of the Gauss-Jacobi polynomial, from FastGaussQuadrature.jl
    roots = _get_roots(family, K) # in [-1, 1]

    # Shift from [-1,1] to [0,1]
    taus = (roots .+ 1) ./ 2

    # True collocation points only, tau0 = 0 is added per depending on basis in basis.jl
    return taus
end

_get_roots(::GaussRadau,    K::Integer) = -reverse(FastGaussQuadrature.gaussradau(K)[1])
_get_roots(::GaussLegendre, K::Integer) = FastGaussQuadrature.gausslegendre(K)[1]
_get_roots(::GaussLobatto,  K::Integer) =
    K >= 2 ? FastGaussQuadrature.gausslobatto(K)[1] : error("GaussLobatto requires K ≥ 2")