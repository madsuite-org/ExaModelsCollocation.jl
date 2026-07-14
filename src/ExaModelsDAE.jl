"""
    ExaModelsDAE

Automatically transcribes differential equations as algebraic constraints in ExaModels.jl
"""
module ExaModelsDAE

import ExaModels: 
    ExaCore
import FastGaussQuadrature

include("utils.jl")

for file in [
        "taus",
        "polynomial",
        "basis",
        "initialize",
        "mesh",
        "daeta",
        "variables",
        "collocation",
        "continuity",
        "initialcons",
        "algebraic",
        "pathcons",
        "terminalcons",
    ]
    include("dynamic/$file.jl")
end

include("exports.jl")
export add_dae

end