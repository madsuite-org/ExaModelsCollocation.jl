"""
    ExaModelsCollocation

Collocation-aware helpers layered over ExaModels.jl.

Each `add_*_collocation` helper mirrors its `ExaModels.add_*` counterpart, adding only what
collocation requires: the residual form of the [`DAEta`](@ref)'s `CollocationMode`, the
basis-polynomial sum as a constraint augmentation, and iteration over the interpolation
points. The caller assembles the model directly.
"""
module ExaModelsCollocation

import ExaModels
import ExaModels: ExaCore
import FastGaussQuadrature

for file in [
        "taus",
        "polynomial",
        "basis",
        "mesh",
        "daeta",
    ]
    include("collocation/$file.jl")
end

include("add_var_collocation.jl")
include("add_con_collocation.jl")
include("add_con_continuity.jl")


export DAEta, set_mesh!, block
export add_var_collocation, @add_var_collocation
export @add_con_collocation, @add_con_continuity
export collocation_itr, continuity_itr
export StateForm, DerivativeForm
export Lagrange
export GaussRadau, GaussLegendre, GaussLobatto

end
