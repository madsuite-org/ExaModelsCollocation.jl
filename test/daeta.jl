@testset "DAEta construction" begin
    nodes = range(0.0, 5.0; length = 21)
    K = 3

    @testset "mesh geometry" begin
        dae = DAEta(nodes, K)
        @test dae.N == 20
        @test dae.K == 3
        @test length(dae.mesh.h) == 20
        @test size(dae.mesh.t) == (20, K)
        @test dae.mesh.t[1, end] <= dae.nodes[2]      # collocation times stay inside their interval
        @test all(dae.mesh.h .≈ 0.25)
        @test length(dae.weights.taus) == K

        # Derived properties read flat but are not stored twice
        @test dae.nodes === dae.mesh.nodes
        @test dae.weights === dae.mode.weights
        @test dae.roots === dae.mode.roots
        @test fieldnames(DAEta) == (:mode, :mesh, :vars, :cons, :blocks)
    end

    @testset "empty container" begin
        # Internal: DAEta(nodes, K) is the only public way to build one, so a mesh-less
        # container never escapes the constructor. The guards are kept as insurance.
        dae = ExaModelsCollocation.DAEta()
        @test dae.mesh === nothing
        @test dae.N == 0
        @test sprint(show, dae) == "DAEta (no mesh)"

        core = ExaModels.ExaCore(; concrete = Val(true))
        @test_throws ErrorException add_var_collocation(core, dae, 1:2)

        ExaModelsCollocation._set_mesh!(dae, nodes, K)
        @test dae.N == 20
    end

    @testset "mode selection" begin
        for r in (GaussRadau(), GaussLegendre())
            dae = DAEta(nodes, K; roots = r)
            @test dae.roots === r
            @test dae.basis isa StateForm
            # StateForm prepends the tau0 = 0 anchor: A is (K+1) x K and b is length K+1
            @test size(dae.weights.A) == (K + 1, K)
            @test length(dae.weights.b) == K + 1
            @test length(dae.weights.taus) == K

            # DerivativeForm needs no anchor: A is K x K and b is length K
            dae = DAEta(nodes, K; roots = r, basis = DerivativeForm())
            @test size(dae.weights.A) == (K, K)
            @test length(dae.weights.b) == K
        end
    end

    @testset "input validation" begin
        @test_throws ArgumentError DAEta([0.0], K)              # too few boundaries
        @test_throws ArgumentError DAEta(nodes, 0)              # degree below 1
        @test_throws ArgumentError DAEta([0.0, 2.0, 1.0], K)    # nodes out of order
    end

    @testset "unknown property" begin
        dae = DAEta(nodes, K)
        @test_throws ArgumentError dae.nope
    end
end
