using ExaModels
using ExaModelsCollocation
using MadNLP
using Test

include("backends.jl")

const ROOTS = (GaussRadau(), GaussLegendre(), GaussLobatto())
const BASES = (StateForm(), DerivativeForm())
const DEGREES = 1:5

@testset "ExaModelsCollocation" begin
    @testset "collocation" begin
        include("collocation/taus.jl")
        include("collocation/polynomial.jl")
        include("collocation/basis.jl")
        include("collocation/mesh.jl")
    end

    @testset "core" begin
        include("core/core.jl")
        include("core/handles.jl")
    end

    @testset "api" begin
        include("api/add_var_collocation.jl")
        include("api/add_con_collocation.jl")
        include("api/add_con_continuity.jl")
    end
end
