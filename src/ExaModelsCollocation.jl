"""
    ExaModelsCollocation

Helper functions for implementing orthogonal collocation in [ExaModels.jl](https://github.com/exanauts/ExaModels.jl).
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
