@testset "add_var_collocation" begin
    nodes = range(0.0, 5.0; length = 21)
    N, K = 20, 3
    nz, nu, Nc = 3, 1, 2

    @testset "mesh indices are appended" begin
        core = CollocationExaCore(nodes, K)

        core, z = add_var_collocation(core, 1:nz, 1:Nc)
        @test core.nvar == nz * Nc * N * (K + 1)
        @test z.krange == 0:K
        @test z.dims == (1:nz, 1:Nc)
        @test ExaModels.size(z.size) == (nz, Nc, N, K + 1)

        core, u = add_var_collocation(core, 1:nu, 1:Nc; include_boundary = false)
        @test core.nvar == nz * Nc * N * (K + 1) + nu * Nc * N * K
        @test u.krange == 1:K
    end

    @testset "Integer dims are normalized to ranges" begin
        core = CollocationExaCore(nodes, K)
        core, z = add_var_collocation(core, nz)
        @test z.dims == (1:nz,)
        @test ExaModels.size(z.size) == (nz, N, K + 1)

        @add_con_collocation(core, coll, z[nz], -z[nz])
        @test_throws ArgumentError add_con_continuity(core, z)
    end

    @testset "start, lvar and uvar are length-checked" begin
        single() = CollocationExaCore(nodes, K)
        both() = CollocationExaCore([nodes, nodes], K)
        len = nz * N * (K + 1)

        core, z = add_var_collocation(single(), 1:nz; start = fill(0.5, len))
        @test all(==(0.5), core.x0)
        @test_throws DimensionMismatch add_var_collocation(single(), 1:nz; start = zeros(len - 1))
        @test_throws DimensionMismatch add_var_collocation(both(), 1:nz; start = fill(0.5, len))
        @test_throws DimensionMismatch add_var_collocation(both(), 1:nz; lvar = zeros(len))
        @test_throws DimensionMismatch add_var_collocation(both(), 1:nz; uvar = ones(len))

        core, z = add_var_collocation(both(), 1:nz; start = fill(0.5, nz, 2, N, K + 1))
        @test length(core.x0) == 2 * len && all(==(0.5), core.x0)

        core, zp = add_var_collocation(both(), 1:nz; mesh = 2, start = fill(0.5, len))
        @test zp.dims == (1:nz,) && length(core.x0) == len

        core, zs = add_var_collocation(both(), 1:nz; start = 0.5)
        @test length(core.x0) == 2 * len
    end

    @testset "the name is an optional Val keyword, as in add_var" begin
        core = CollocationExaCore(nodes, K)

        core, y = add_var_collocation(core, 1:nz, 1:Nc)
        @test core.refs == (;)
        @test y.dims == (1:nz, 1:Nc)
        @test y in core.block

        core, z = add_var_collocation(core, 1:nz, 1:Nc; name = Val(:z))
        @test core.z === z
        @test :z in propertynames(core)
        @test ExaModels.ExaModel(core).z === z
    end

    @testset "macro form binds locally and on the core" begin
        core = CollocationExaCore(nodes, K)

        @add_var_collocation(core, z, 1:nz, 1:Nc)
        @add_var_collocation(core, u, 1:nu, 1:Nc; include_boundary = false)

        @test z isa CollocationVariable
        @test u isa CollocationVariable
        @test core.z === z
        @test core.u === u
        @test core.nvar == nz * Nc * N * (K + 1) + nu * Nc * N * K
        @test z.krange == 0:K
        @test u.krange == 1:K
    end

    @testset "keyword spellings agree" begin
        c1 = CollocationExaCore(nodes, K)
        @add_var_collocation(c1, z, 1:nz; include_boundary = false)

        c2 = CollocationExaCore(nodes, K)
        @add_var_collocation(c2, z, 1:nz, include_boundary = false)

        @test c1.nvar == c2.nvar == nz * N * K
    end

    @testset "anonymous form registers nothing" begin
        core = CollocationExaCore(nodes, K)
        z = @add_var_collocation(core, 1:nz)

        @test z isa CollocationVariable
        @test core.refs == (;)
        @test z in core.block
        @test z.krange == 0:K
    end
end
