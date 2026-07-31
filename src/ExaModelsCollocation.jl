"""
    ExaModelsCollocation

Collocation-aware helpers layered over ExaModels.jl.

Each `add_*_collocation` helper mirrors its `ExaModels.add_*` counterpart, adding only what
collocation requires: the residual form of the [`CollocationExaCore`](@ref)'s
`CollocationMode`, the basis-polynomial sum as a constraint augmentation, and iteration over
the interpolation points. The caller assembles the model directly.
"""
module ExaModelsCollocation

import ExaModels
import ExaModels: ExaCore
import FastGaussQuadrature

for file in [
        "collocation/taus",
        "collocation/polynomial",
        "collocation/basis",
        "collocation/mesh",
        "core/handles",
        "core/core",
    ]
    include("$file.jl")
end

include("add_var_collocation.jl")
include("add_con_collocation.jl")
include("add_con_continuity.jl")


export CollocationExaCore, CollocationExaModel, CollocationVariable, Collocation
export add_var_collocation, @add_var_collocation
export add_con_collocation, @add_con_collocation
export add_con_continuity, @add_con_continuity
export set_nodes!
export StateForm, DerivativeForm
export GaussRadau, GaussLegendre, GaussLobatto

end
