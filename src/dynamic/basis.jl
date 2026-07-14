# ----- COMPLETE 7/14/2026 -----
# Basis representations for the differential state
# default = ExaModelsDAE.StateForm()
#   ExaModelsDAE.StateForm()      : the differential state is represented by the interpolating polynomial
#   ExaModelsDAE.DerivativeForm() : the time derivative of the differential state is represented by the interpolating polynomial
"""
    AbstractBasis

Abstract type for the basis representation of the differential state.
A concrete subtype determines whether the interpolating polynomial
represents the state `z` ([`StateForm`](@ref)) or its time derivative
`dz/dt` ([`DerivativeForm`](@ref)).
"""
abstract type AbstractBasis end

"""
    StateForm()

The interpolating polynomial represents the differential state `z`.
"""
struct StateForm <: AbstractBasis end

"""
    DerivativeForm()

The interpolating polynomial represents the time derivative of the
differential state `dz/dt` (Runge-Kutta).
"""
struct DerivativeForm <: AbstractBasis end

"""
    BasisWeights{T}

Contains the collocation and continuity weights

# Fields
- `A`: collocation constraint weights A[j,k], j=0,...,K, k=1,...,K. jth basis polynomial evaluated at tau[k]
- `b`: continuity constraint weights b[j], j=0,...,K. jth basis polynomial evaluated at tau=1
"""
struct BasisWeights{T}
    A::Matrix{T}
    b::Vector{T}
end

function _get_weights(basis::AbstractBasis, polynomial::AbstractPolynomial, taus::AbstractVector{T}) where {T <: Real}
    # Get derivative interpolation weights A[i,j]
    A = _get_weights_A(basis, polynomial, taus)
    
    # Get continuity weights b[j]
    b = _get_weights_b(basis, polynomial, taus)
    
    return BasisWeights(A, b)
end

_get_weights_A(::StateForm, ::Lagrange, taus) = delljk(taus)
_get_weights_b(::StateForm, ::Lagrange, taus) = ell1j(taus)

_get_weights_A(::DerivativeForm, ::Lagrange, taus) = Omegajk(taus)
_get_weights_b(::DerivativeForm, ::Lagrange, taus) = Omega1j(taus)
