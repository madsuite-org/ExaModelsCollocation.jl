# Basis representations for the differential state
# default = ExaModelsDAE.StateForm(), ExaModelsDAE.DerivativeForm()
#   ExaModelsDAE.StateForm()      : the differential state is represented by the interpolating polynomial
#   ExaModelsDAE.DerivativeForm() : the time derivative of the differential state is represented by the interpolating polynomial

# TODO: note: for StateForm() weights, need dljdtau(tauk) + lj(1). for DerivativeForm() weights, need Omegaj(tauk) + omegaj(1)
# TODO: contains: collocation wieghts +

"""
    AbstractBasis

Abstract type for the basis representation of the differential state.
A concrete subtype determines whether the interpolating polynomial
represents the state `z` ([`StateForm`](@ref)) or its time derivative
`dz/dt` ([`DerivativeForm`](@ref)), which sets the algebraic form of
the collocation constraint.
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
    _get_weights(basis, polynomial, tau) -> BasisWeights

Reference-element collocation weights for `basis`/`polynomial` over `tau = [0, τ₁..τ_K]`.
`StateForm` uses the full node set; `DerivativeForm` drops the `0` anchor. `BasisWeights.tau`
stores the roots `τ₁..τ_K` only.
"""
function _get_weights(::StateForm, polynomial::Lagrange, tau)
    BasisWeights(dljk(polynomial, tau), lj1(polynomial, tau), tau[2:end])
end

function _get_weights(::DerivativeForm, polynomial::Lagrange, tau)
    roots = tau[2:end]
    BasisWeights(omegajk(polynomial, roots), omegaj1(polynomial, roots), roots)
end
