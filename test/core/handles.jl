@testset "the handle is a Variable that carries its layout" begin
    core = CollocationExaCore(range(0.0, 5.0; length = 21), 3)
    core, z = add_var_collocation(core, 1:3)

    @test z isa ExaModels.AbstractVariable
    @test z isa CollocationVariable
    @test z[1, 1, 0] isa ExaModels.AbstractNode

    @test ExaModelsCollocation._nleading(z) == 1
end

@testset "a block indexed by its dimensions alone names a slot" begin
    core = CollocationExaCore(range(0.0, 5.0; length = 21), 3)
    core, z = add_var_collocation(core, 1:3, 1:2)
    core, y = add_var_collocation(core)
    core, u = add_var_collocation(core, 1:3; include_boundary = false)

    @test z[2, 1] isa ExaModelsCollocation.CollocationSlot
    @test z[2, 1].var === z
    @test z[2, 1].idx == (2, 1)
    @test z[2, 1, 4, 0] isa ExaModels.AbstractNode

    @test y[] isa ExaModelsCollocation.CollocationSlot
    @test y[].idx == ()
    @test y[4, 0] isa ExaModels.AbstractNode

    @test u[3] isa ExaModelsCollocation.CollocationSlot
    @test u[3, 4, 1] isa ExaModels.AbstractNode
end

@testset "start and bounds pass through" begin
    core = CollocationExaCore(range(0.0, 5.0; length = 21), 3)
    N, K, nz, Nc = 20, 3, 3, 2
    start = fill(0.5, nz, Nc, N, K + 1)
    core, z = add_var_collocation(core, 1:nz, 1:Nc; start = start, lvar = -2.0, uvar = 2.0)

    @test all(core.x0 .== 0.5)
    @test all(core.lvar .== -2.0)
    @test all(core.uvar .== 2.0)

    m = ExaModels.ExaModel(core)
    @test all(ExaModels.get_start(m, z) .== 0.5)
    @test all(ExaModels.get_lvar(m, z) .== -2.0)
    ExaModels.set_start!(m, z, fill(0.25, z.length))
    @test all(ExaModels.get_start(m, z) .== 0.25)
    @test_throws DimensionMismatch ExaModels.set_start!(m, z, [1.0])
end
