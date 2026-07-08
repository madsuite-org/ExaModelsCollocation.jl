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
