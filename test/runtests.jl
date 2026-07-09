using ExaModelsDAE
using Test

const EMD = ExaModelsDAE

@testset "ExaModelsDAE.jl" begin
    @testset "strategy types" begin
        @test EMD.GaussRadau() isa EMD.AbstractRoots
        @test EMD.GaussLegendre() isa EMD.AbstractRoots
        @test EMD.GaussLobatto() isa EMD.AbstractRoots

        @test EMD.Lagrange() isa EMD.AbstractPolynomial

        @test EMD.StateForm() isa EMD.AbstractBasis
        @test EMD.DerivativeForm() isa EMD.AbstractBasis
    end

    @testset "public API" begin
        @test isdefined(ExaModelsDAE, :add_dae)
        # add_dae is exported; strategy types are accessed qualified (EMD.…)
        @test :add_dae in names(ExaModelsDAE)
    end

    @testset "DAEta" begin
        dae = EMD.DAEta(
            nothing, nothing, nothing, nothing, nothing, nothing, nothing,  # handles
            2, 0, 1, 1, 0, 10, 3,                                           # dims
            collect(0.0:0.1:1.0), fill(0.1, 10), [0.3, 0.7, 1.0],           # nodes, h, tau
            zeros(10, 3), zeros(4, 4), [0.0, 0.0, 0.0, 1.0],                # t, A, b
            (basis = EMD.StateForm(), polynomial = EMD.Lagrange(), roots = EMD.GaussRadau()),
            (;),
        )
        @test dae.nz == 2
        @test dae.ny == 0
        @test dae.N == 10
        @test dae.K == 3
        @test dae.method.roots isa EMD.GaussRadau
        @test occursin("DAEta", sprint(show, dae))
    end
end
