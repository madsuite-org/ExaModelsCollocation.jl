using ExaModelsDynamic
using Test

const EMD = ExaModelsDynamic

@testset "ExaModelsDynamic.jl" begin
    @testset "strategy types" begin
        @test EMD.GaussRadau() isa EMD.AbstractRoots
        @test EMD.GaussLegendre() isa EMD.AbstractRoots
        @test EMD.GaussLobatto() isa EMD.AbstractRoots

        @test EMD.Lagrange() isa EMD.AbstractPolynomial

        @test EMD.StateForm() isa EMD.AbstractBasis
        @test EMD.DerivativeForm() isa EMD.AbstractBasis
    end

    @testset "public API" begin
        @test isdefined(ExaModelsDynamic, :add_dae)
        # add_dae is exported; strategy types are accessed qualified (EMD.…)
        @test :add_dae in names(ExaModelsDynamic)
    end

    @testset "CollocationData" begin
        dae = EMD.CollocationData(
            nothing, nothing, nothing, nothing, nothing, nothing, nothing,  # handles
            2, 0, 1, 1, 0, 10, 3,                                           # dims
            collect(0.0:0.1:1.0), fill(0.1, 10), [0.3, 0.7, 1.0],           # nodes, h, ρ
            zeros(10, 3), zeros(4, 4), [0.0, 0.0, 0.0, 1.0],                # t, D, ω1
            (basis = EMD.StateForm(), polynomial = EMD.Lagrange(), roots = EMD.GaussRadau()),
            (;),
        )
        @test dae.nz == 2
        @test dae.ny == 0
        @test dae.N == 10
        @test dae.K == 3
        @test dae.method.roots isa EMD.GaussRadau
        @test occursin("CollocationData", sprint(show, dae))
    end
end
