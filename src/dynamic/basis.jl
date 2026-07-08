# Basis representations for the differential state
# default = ExaModelsDynamic.StateForm(), ExaModelsDynamic.DerivativeForm()
#   ExaModelsDynamic.StateForm()      : the differential state is represented by the interpolating polynomial
#   ExaModelsDynamic.DerivativeForm() : the time derivative of the differential state is represented by the interpolating polynomial

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
