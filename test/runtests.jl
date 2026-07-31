using ExaModels
using ExaModelsCollocation
using Test

@testset "ExaModelsCollocation" begin
    include("collocation.jl")             # taus, polynomial, basis, mesh
    include("api.jl")                     # the helpers, and both residual forms against theory
    include("vanderpol.jl")               # the helpers end to end on a control problem
    include("bruno.jl")                   # and on a PEtab parameter-estimation problem
end
