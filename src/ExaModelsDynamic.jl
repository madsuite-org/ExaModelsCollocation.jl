"""
    ExaModelsDyanmic

Automatically transcribes differential equations as algebraic constraints in ExaModels.jl
"""
module ExaModelsDynamic

import ExaModels: 
    ExaCore, 
    ExaModels

include("structs.jl")
include("utils.jl")
include("initialize.jl")
include("nodes.jl")
include("basis.jl")
include("polynomial.jl")
include("roots.jl")

include("exports.jl")
export add_dae

end