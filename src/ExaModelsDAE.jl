"""
    ExaModelsDAE

Automatically transcribes differential equations as algebraic constraints in ExaModels.jl
"""
module ExaModelsDAE

import ExaModels: 
    ExaCore, 
    ExaModels
import FastGaussQuadrature

include("structs.jl")
include("utils.jl")

for file in [
        "basis",    
        "polynomial",
        "taus",
        "mesh",
        "initialization",
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