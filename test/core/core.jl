@testset "CollocationExaCore construction" begin
    nodes = range(0.0, 5.0; length = 21)
    K = 3

    @testset "mesh geometry" begin
        core = CollocationExaCore(nodes, K)
        @test core.N == 20
        @test core.K == 3
        @test length(core.mesh.h) == 20
        @test size(core.mesh.t) == (20, K)
        @test core.mesh.t[1, end] <= core.nodes[2]
        @test all(core.mesh.h .≈ 0.25)
        @test length(core.weights.taus) == K
        @test core.adaptive == false

        # Derived properties read flat but are not stored twice
        @test core.nodes === core.mesh.nodes
        @test core.weights === core.mode.weights
        @test core.roots === core.mode.roots
        @test fieldnames(ExaModelsCollocation.CollocationTag) ==
            (:mode, :mesh, :block, :resid)
    end

    @testset "it is an ExaCore" begin
        core = CollocationExaCore(nodes, K)
        @test core isa ExaModels.ExaCore
        @test core isa ExaModelsCollocation.CollocationExaCore

        # every plain ExaModels call works on it, unforwarded
        core, x = ExaModels.add_var(core, 1:3; name = Val(:x))
        core, p = ExaModels.add_par(core, 1:2; value = [1.0, 2.0])
        ExaModels.@add_var(core, w, 1:4)
        ExaModels.@add_con(core, g, x[i] - 1.0 for i in 1:3)
        @test core.nvar == 7 && core.npar == 2 && core.ncon == 3
        @test core.x === x && core.w === w && core.g === g

        # and the tag survives into the model, which is what set_nodes! needs
        m = ExaModels.ExaModel(core)
        @test m isa ExaModelsCollocation.CollocationExaModel
        @test m.N == 20 && m.K == 3
        @test m.mesh === core.mesh
        @test m.basis isa StateForm
    end

    @testset "both entry points agree" begin
        # CollocationExaCore is sugar over ExaCore(T; tag = Collocation(...)); Collocation
        # holds the validation so the sugar cannot get around it.
        c1 = CollocationExaCore(nodes, K; roots = GaussLegendre())
        c2 = ExaModels.ExaCore(concrete = Val(true);
                               tag = Collocation(nodes, K; roots = GaussLegendre()))
        @test typeof(c1) === typeof(c2)
        @test c1.N == c2.N && c1.K == c2.K && c1.roots === c2.roots

        plain = ExaModels.ExaCore(concrete = Val(true))
        plain, p = ExaModels.add_var(plain, 1:4; name = Val(:p))
        attached = ExaModels.ExaCore(plain; tag = Collocation(nodes, K))
        @test attached.nvar == 4
        @test attached.p === p
        @test attached.N == 20

        # a LegacyExaCore cannot carry the tag, so `concrete` is refused outright
        @test_throws ArgumentError CollocationExaCore(nodes, K; concrete = Val(false))
    end

    @testset "unknown property" begin
        # falls through to getfield, so the exact type moved to FieldError in 1.12
        core = CollocationExaCore(nodes, K)
        @test_throws Exception core.nope
    end

    @testset "builds on $(something(b, "CPU"))" for b in BACKENDS
        N, Nz = 5, 2
        core = CollocationExaCore(range(0.0, 1.0; length = N + 1), 3; backend = b)
        @add_var_collocation(core, z, 1:Nz)
        @add_con_collocation(core, coll, z[v], -z[v] for v in 1:Nz)
        @add_con_continuity(core, cont, z)

        m = ExaModels.ExaModel(core)
        @test m isa ExaModelsCollocation.CollocationExaModel
        @test m.meta.ncon == Nz * N * 3 + Nz * (N - 1)
    end
end

@testset "set_nodes!" begin
    @testset "moves the mesh without rebuilding" begin
        # A rhs that genuinely depends on t, so a mesh update that moved h but not t would
        # show up here: dz/dt = -2t z, z(0) = 1 => exp(-t^2).
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

    @testset "rejects what it cannot do" begin
        core = CollocationExaCore(range(0.0, 1.0; length = 6), 3)
        @test_throws ArgumentError set_nodes!(core, collect(range(0.0, 2.0; length = 6)))

        acore = CollocationExaCore(range(0.0, 1.0; length = 6), 3; adaptive = true)
        # changing the interval count changes the variable count, so it needs a rebuild
        @test_throws DimensionMismatch set_nodes!(acore, collect(range(0.0, 1.0; length = 7)))
        @test_throws ArgumentError set_nodes!(acore, [0.0, 0.4, 0.2, 0.6, 0.8, 1.0])
        # and a repeated boundary, which would collapse an interval to zero width
        @test_throws ArgumentError set_nodes!(acore, [0.0, 0.2, 0.2, 0.6, 0.8, 1.0])
    end
end
