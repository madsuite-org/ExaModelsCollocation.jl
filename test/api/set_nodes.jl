@testset "set_nodes!" begin
    @testset "moves the mesh without rebuilding" begin
        function build(adaptive; nodes = range(0.0, 1.0; length = 11), K = 3)
            core = CollocationExaCore(nodes, K; adaptive)
            @add_var_collocation(core, z, 1:1)
            @add_con_collocation(core, coll, z[v], -2 * t * z[v] for v in 1:1)
            @add_con_continuity(core, cont, z)
            ExaModels.@add_con(core, ic, z[v, 1, 0] - 1.0 for v in 1:1)
            ExaModels.ExaModel(core), z
        end
        terminal_of(m, z) =
            ExaModels.solution(madnlp(m; print_level = MadNLP.ERROR, tol = 1e-12), z)[1, end, end]

        m, z = build(true)
        graded = [(i / 10)^2 for i in 0:10]
        set_nodes!(m, graded)
        @test m.nodes ≈ graded
        @test ExaModelsCollocation._hval(m.mesh) ≈ diff(graded)

        mr, zr = build(false; nodes = graded)
        @test terminal_of(m, z) ≈ terminal_of(mr, zr) rtol = 1e-10
    end

    @testset "moves a core, not just a model" begin
        core = CollocationExaCore(range(0.0, 1.0; length = 6), 3; adaptive = true)
        @add_var_collocation(core, z)
        graded = [(i / 5)^2 for i in 0:5]
        set_nodes!(core, graded)

        @test core.nodes ≈ graded
        @test ExaModelsCollocation._hval(core.mesh) ≈ diff(graded)
        @test core.θ[1:5] ≈ diff(graded)
        @test ExaModels.ExaModel(core).θ[1:5] ≈ diff(graded)
    end

    @testset "moves one mesh, leaving the rest" begin
        tfs = (1.0, 4.0)
        nodes = [collect(range(0.0, tf; length = 6)) for tf in tfs]
        core = CollocationExaCore(nodes, 3; adaptive = true)
        @add_var_collocation(core, z)

        graded = [4 * (i / 5)^2 for i in 0:5]
        set_nodes!(core, graded; mesh = 2)
        ref = CollocationExaCore([nodes[1], graded], 3; adaptive = true)

        @test core.nodes[1, :] ≈ nodes[1]
        @test core.nodes[2, :] ≈ graded
        @test ExaModelsCollocation._hval(core.mesh) ≈ ExaModelsCollocation._hval(ref.mesh)
        @test ExaModelsCollocation._tval(core.mesh) ≈ ExaModelsCollocation._tval(ref.mesh)
        @test core.θ[1:10] ≈ vec(ExaModelsCollocation._hval(core.mesh))

        @test_throws ArgumentError set_nodes!(core, graded; mesh = 3)
        @test_throws DimensionMismatch set_nodes!(core, graded[1:(end - 1)]; mesh = 2)

        single = CollocationExaCore(range(0.0, 1.0; length = 6), 3; adaptive = true)
        @test_throws ArgumentError set_nodes!(single, collect(range(0.0, 1.0; length = 6)); mesh = 1)
    end

    @testset "rejects what it cannot do" begin
        core = CollocationExaCore(range(0.0, 1.0; length = 6), 3)
        @test_throws ArgumentError set_nodes!(core, collect(range(0.0, 2.0; length = 6)))

        acore = CollocationExaCore(range(0.0, 1.0; length = 6), 3; adaptive = true)
        @test_throws DimensionMismatch set_nodes!(acore, collect(range(0.0, 1.0; length = 7)))
        @test_throws ArgumentError set_nodes!(acore, [0.0, 0.4, 0.2, 0.6, 0.8, 1.0])
        @test_throws ArgumentError set_nodes!(acore, [0.0, 0.2, 0.2, 0.6, 0.8, 1.0])
    end
end
