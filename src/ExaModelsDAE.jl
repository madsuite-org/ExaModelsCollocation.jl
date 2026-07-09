"""
    ExaModelsDAE

Automatically transcribes differential equations as algebraic constraints in ExaModels.jl
"""
module ExaModelsDAE

import ExaModels: 
    ExaCore, 
    ExaModels

include("structs.jl")
include("utils.jl")

for file in [
        "roots",
        "polynomial",
        "basis",
        "nodes",
        "variables",
        "collocation",
        "algebraic",
        "continuity",
        "initialcons",
        "terminalcons",
        "pathcons",
        "initialization",
    ]
    include("dynamic/$file.jl")
end

include("exports.jl")
export add_dae

end