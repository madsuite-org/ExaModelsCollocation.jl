# Step 2 of the mode: turn the collocation points into the weights the constraints use.
#   StateForm()      : the interpolating polynomial represents the state z
#   DerivativeForm() : it represents dz/dt instead (Runge-Kutta)
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

The interpolating polynomial represents the time derivative of the differential state
`dz/dt` (Runge-Kutta).

Its weights are the integrated basis `Ω_j = ∫ ℓ_j` over the `K` collocation points, giving
the residual `z_{i,k} = z_{i,0} + h Σ_j Ω_j(τ_k) ż_{i,j}` — the implicit Runge-Kutta step,
with `A` the Butcher tableau and `b` its quadrature weights.

The variables are the same as [`StateForm`](@ref)'s: `ż_{i,j}` is never a decision variable,
it is the right-hand side evaluated at `z_{i,j}`. So [`add_var_collocation`](@ref) allocates
the same block for either basis, and only the residual the `add_con_*` helpers build changes.
"""
struct DerivativeForm <: AbstractBasis end

"""
    BasisWeights{T}

Contains the collocation and continuity weights

# Fields
- `A`: collocation constraint weights A[j,k], j=0,...,K, k=1,...,K. jth basis polynomial evaluated at taus[k]
- `b`: continuity constraint weights b[j], j=0,...,K. jth basis polynomial evaluated at tau=1
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

# StateForm builds the basis over the anchored nodes [0, tau_1..tau_K] and differentiates at
# the K collocation points, so A is (K+1) x K and b is length K+1 -- indexed A[j+1,k] and
# b[j+1] for j = 0,...,K.
_get_weights_A(::Lagrange, ::StateForm, taus) = delljk(_anchor(taus), taus)
_get_weights_b(::Lagrange, ::StateForm, taus) = ell1j(_anchor(taus))

# DerivativeForm needs no anchor: it builds the basis over the K collocation points alone and
# integrates, so A is K x K and b is length K -- indexed A[j,k] and b[j] for j = 1,...,K.
_get_weights_A(::Lagrange, ::DerivativeForm, taus) = Omegajk(taus, taus)
_get_weights_b(::Lagrange, ::DerivativeForm, taus) = Omega1j(taus)

# Prepend the tau0 = 0 interpolation anchor to the collocation points
_anchor(taus) = vcat(zero(eltype(taus)), taus)
