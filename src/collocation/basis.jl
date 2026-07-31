# Basis representations for the differential state
# default = StateForm()
#   StateForm()      : the differential state is represented by the interpolating polynomial
#   DerivativeForm() : the time derivative of the differential state is represented by the interpolating polynomial
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

The interpolating polynomial represents the differential state, `z` (classic collocation).
"""
struct StateForm <: AbstractBasis end

"""
    DerivativeForm()

The interpolating polynomial represents the time derivative of the differential state,
`dz/dt` (implicit Runge-Kutta).
"""
struct DerivativeForm <: AbstractBasis end

"""
    BasisWeights{T}

Contains the collocation and continuity weights

# Fields
- `A`: collocation constraint weights A[j,k], j=0,...,K,
- `b`: continuity constraint weights b[j], j=0,...,K.
- `taus` : true collocation points taus[j], j=1,...,K.
"""
struct BasisWeights{T, MA <: AbstractMatrix{T}, VB <: AbstractVector{T}, VT <: AbstractVector{T}}
    A::MA
    b::VB
    taus::VT
end

function _get_weights(polynomial::AbstractPolynomial, basis::AbstractBasis, taus::AbstractVector{T}) where {T <: Real}
    # Get derivative interpolation weights A[i,j]
    A = _get_weights_A(polynomial, basis, taus)
    
    # Get continuity weights b[j]
    b = _get_weights_b(polynomial, basis, taus)
    
    return BasisWeights(A, b, taus)
end

# StateForm weights, A[j=0:K,k=1:K], b[j=0:K] (+1 to account for 1-based indexing)
_get_weights_A(::Lagrange, ::StateForm, taus) = delljk(_add_tau0(taus), taus)
_get_weights_b(::Lagrange, ::StateForm, taus) = ell1j(_add_tau0(taus))

# DerivativeForm weights, A[j=1:K,k=1:K], b[j=1:K]
_get_weights_A(::Lagrange, ::DerivativeForm, taus) = Omegajk(taus, taus)
_get_weights_b(::Lagrange, ::DerivativeForm, taus) = Omega1j(taus)

# Prepend the tau0 = 0 interpolation anchor to the collocation points
_add_tau0(taus) = vcat(zero(eltype(taus)), taus)
