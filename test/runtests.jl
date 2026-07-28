using ExaModels
using ExaModelsCollocation
using Test

@testset "ExaModelsCollocation" begin
    include("collocation.jl")             # taus, polynomial, basis, mesh
    include("daeta.jl")
    include("add_var_collocation.jl")
    include("add_con_collocation.jl")     # both residual forms, against theory
    include("vanderpol.jl")               # the helpers end to end on a control problem
    include("bruno.jl")                   # and on a PEtab parameter-estimation problem
end
