@testset "add_var_collocation" begin
    nodes = range(0.0, 5.0; length = 21)
    N, K = 20, 3
    nz, nu, Nc = 3, 1, 2

    @testset "mesh indices are appended" begin
        core = CollocationExaCore(nodes, K)

        core, z = add_var_collocation(core, 1:nz, 1:Nc)
        # z[v,c] declared -> z[v,c,i,k] allocated, k = 0,...,K
        @test core.nvar == nz * Nc * N * (K + 1)
        @test z.krange == 0:K
        @test z.dims == (1:nz, 1:Nc)
        @test ExaModels.size(z.size) == (nz, Nc, N, K + 1)

        core, u = add_var_collocation(core, 1:nu, 1:Nc; include_boundary = false)
        # u[v,c] declared -> u[v,c,i,k] allocated, k = 1,...,K
        @test core.nvar == nz * Nc * N * (K + 1) + nu * Nc * N * K
        @test u.krange == 1:K
    end

    @testset "Integer dims are normalized to ranges" begin
        # add_var takes an Integer or a UnitRange per dimension, `n` meaning 1:n. The block
        # stores ranges either way, so add_con_continuity can enumerate its slots -- left as
        # an Integer, the coverage check would see the single slot z[nz] and pass vacuously.
        core = CollocationExaCore(nodes, K)
        core, z = add_var_collocation(core, nz)
        @test z.dims == (1:nz,)
        @test ExaModels.size(z.size) == (nz, N, K + 1)

        @add_con_collocation(core, coll, z[nz], -z[nz])
        @test_throws ArgumentError add_con_continuity(core, z)
    end

    @testset "the name is an optional Val keyword, as in add_var" begin
        core = CollocationExaCore(nodes, K)

        core, y = add_var_collocation(core, 1:nz, 1:Nc)
        @test core.refs == (;)
        @test y.dims == (1:nz, 1:Nc)
        @test y in core.block

        # named, exactly the way ExaModels.add_var takes it -- and registered upstream, so
        # the collocation handle is what core.z and model.z give back
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
