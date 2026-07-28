@testset "add_var_collocation" begin
    nodes = range(0.0, 5.0; length = 21)
    N, K = 20, 3
    nz, nu, Nc = 3, 1, 2

    @testset "mesh indices are appended" begin
        core = ExaModels.ExaCore(; concrete = Val(true))
        dae = DAEta(nodes, K)

        core, z = add_var_collocation(core, dae, 1:nz, 1:Nc)
        # z[v,c] declared -> z[v,c,i,k] allocated, k = 0,...,K
        @test core.nvar == nz * Nc * N * (K + 1)
        @test block(dae, z).krange == 0:K

        core, u = add_var_collocation(core, dae, 1:nu, 1:Nc; include_boundary = false)
        # u[v,c] declared -> u[v,c,i,k] allocated, k = 1,...,K
        @test core.nvar == nz * Nc * N * (K + 1) + nu * Nc * N * K
        @test block(dae, u).krange == 1:K
    end

    @testset "the name is an optional Val keyword, as in add_var" begin
        core = ExaModels.ExaCore(; concrete = Val(true))
        dae = DAEta(nodes, K)

        # anonymous: laid out and tracked, but not registered under any name
        core, y = add_var_collocation(core, dae, 1:nz, 1:Nc)
        @test dae.vars == (;)
        @test block(dae, y).dims == (1:nz, 1:Nc)

        # named, exactly the way ExaModels.add_var takes it
        core, z = add_var_collocation(core, dae, 1:nz, 1:Nc; name = Val(:z))
        @test haskey(dae.vars, :z)
        @test dae.vars isa NamedTuple
        @test dae.z === dae.vars.z === z
        @test :z in propertynames(dae)
        @test core.z === z                      # registered upstream too

        # a handle from some other container is not a block of this one
        @test_throws ArgumentError block(dae, ExaModels.add_var(core, 3)[2])
    end

    @testset "start and bounds pass through" begin
        core = ExaModels.ExaCore(; concrete = Val(true))
        dae = DAEta(nodes, K)

        start = fill(0.5, nz, Nc, N, K + 1)
        core, z = add_var_collocation(core, dae, 1:nz, 1:Nc; start = start, lvar = -2.0, uvar = 2.0)

        @test all(core.x0 .== 0.5)
        @test all(core.lvar .== -2.0)
        @test all(core.uvar .== 2.0)
    end

    @testset "macro form binds locally and on dae" begin
        core = ExaModels.ExaCore(; concrete = Val(true))
        dae = DAEta(nodes, K)

        @add_var_collocation(core, dae, z, 1:nz, 1:Nc)
        @add_var_collocation(core, dae, u, 1:nu, 1:Nc; include_boundary = false)

        # bound in this scope
        @test z isa ExaModels.Variable
        @test u isa ExaModels.Variable

        # and registered on the container
        @test dae.z === z
        @test dae.u === u
        @test core.nvar == nz * Nc * N * (K + 1) + nu * Nc * N * K

        # dae is mutable and was updated in place, so it never needed rebinding
        @test block(dae, u).krange == 1:K
    end

    @testset "keyword spellings agree" begin
        c1 = ExaModels.ExaCore(; concrete = Val(true))
        d1 = DAEta(nodes, K)
        @add_var_collocation(c1, d1, z, 1:nz; include_boundary = false)

        c2 = ExaModels.ExaCore(; concrete = Val(true))
        d2 = DAEta(nodes, K)
        @add_var_collocation(c2, d2, z, 1:nz, include_boundary = false)

        @test c1.nvar == c2.nvar == nz * N * K
    end
end
