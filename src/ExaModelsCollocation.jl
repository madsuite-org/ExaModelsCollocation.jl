"""
    ExaModelsCollocation

Helper functions for orthogonal collocation in [ExaModels.jl](https://github.com/exanauts/ExaModels.jl).
"""
module ExaModelsCollocation

import ExaModels
import ExaModels: ExaCore
import FastGaussQuadrature

include("collocation/taus.jl")
include("collocation/polynomial.jl")
include("collocation/basis.jl")
include("collocation/mesh.jl")
include("core/handles.jl")
include("core/core.jl")
include("api/macros.jl")
include("api/add_var_collocation.jl")
include("api/add_con_collocation.jl")
include("api/add_con_continuity.jl")
include("api/set_nodes.jl")
include("api/interpolate.jl")

export CollocationExaCore, CollocationExaModel, CollocationVariable, Collocation
export add_var_collocation, @add_var_collocation
export add_con_collocation, @add_con_collocation
export add_con_continuity, @add_con_continuity
export set_nodes!, interpolate
export StateForm, DerivativeForm
export GaussRadau, GaussLegendre, GaussLobatto

end
