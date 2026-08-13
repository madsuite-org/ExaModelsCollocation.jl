"""
    ExaModelsCollocation

Helper functions for orthogonal collocation in [ExaModels.jl](https://github.com/exanauts/ExaModels.jl).
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
        "api/macros",
        "api/add_var_collocation",
        "api/add_con_collocation",
        "api/add_con_continuity",
    ]
    include("$file.jl")
end

export CollocationExaCore, CollocationExaModel, CollocationVariable, Collocation
export add_var_collocation, @add_var_collocation
export add_con_collocation, @add_con_collocation
export add_con_continuity, @add_con_continuity
export set_nodes!
export StateForm, DerivativeForm
export GaussRadau, GaussLegendre, GaussLobatto

end
