# The original Van der Pol oscillator control problem.
#
# One add_con call per structurally distinct algebraic expression, and no more: the three
# state equations have different right-hand sides so they need three; the initial conditions
# collapse to two (the numeric ones, then the one tied to p). Continuity is a single call.

using MadNLP

@testset "Van der Pol optimal control" begin
    # min z3(tf)  s.t.  z1' = z2
    #                   z2' = th1 z2 (1 - z1^2) - z1 + u + p2 cos(t)
    #                   z3' = z1^2 + z2^2 + u^2          (running cost as an augmented state)
    #                   z(t0) = (0, p1, 0)
    nz, nu, Nc = 3, 1, 1
    tf, N, K = 5.0, 19, 3

    function build(; p1_bounds = (-2.0, 2.0))
        dae = DAEta(range(0.0, tf; length = N + 1), K)
        core = ExaModels.ExaCore(; concrete = Val(true))

        @add_var_collocation(core, dae, z, 1:nz, 1:Nc)                            # k = 0,...,K
        @add_var_collocation(core, dae, u, 1:nu, 1:Nc; include_boundary = false)  # k = 1,...,K
        ExaModels.@add_var(core, p, 1:2;
            lvar = [p1_bounds[1], -1.0], uvar = [p1_bounds[2], 1.0], start = [1.0, 0.0])
        ExaModels.@add_par(core, th, [1.0])

        # Three structurally distinct right-hand sides, so three calls. Each pins v in the
        # slice and varies c through the iterator.
        itr = collocation_itr(dae, 1:Nc)

        @add_con_collocation(core, dae, coll1, z[1, c],
            z[2, c, i, k] for (c, i, k, t) in itr)

        @add_con_collocation(core, dae, coll2, z[2, c],
            th[1] * z[2, c, i, k] * (1 - z[1, c, i, k]^2)
                - z[1, c, i, k] + u[1, c, i, k] + p[2] * cos(t)
            for (c, i, k, t) in itr)

        @add_con_collocation(core, dae, coll3, z[3, c],
            z[1, c, i, k]^2 + z[2, c, i, k]^2 + u[1, c, i, k]^2
            for (c, i, k, t) in itr)

        # One call for every component and condition
        @add_con_continuity(core, dae, cont,
            z[v, c] for (v, c) in continuity_itr(dae, 1:nz, 1:Nc))

        # z(t0) = (0, p1, 0). The zero entries are one expression, the p-linked one another.
        ExaModels.@add_con(core, ic_fix,
            z[v, c, 1, 0] - val for (v, c, val) in [(1, 1, 0.0), (3, 1, 0.0)])
        ExaModels.@add_con(core, ic_p,
            z[v, c, 1, 0] - p[m] for (v, c, m) in [(2, 1, 1)])

        ExaModels.@add_obj(core, z[3, 1, N, K] for _ in 1:1)

        return core, dae, z, u, p
    end

    @testset "as originally posed" begin
        core, dae, z, u, p = build()
        @test core.nvar == nz * Nc * N * (K + 1) + nu * Nc * N * K + 2
        @test :coll2 in propertynames(dae)

        result = madnlp(ExaModels.ExaModel(core); print_level = MadNLP.ERROR)
        @test result.status == MadNLP.SOLVE_SUCCEEDED
        @test isfinite(result.objective)

        # p1 is a free decision variable and nothing forces motion, so the optimum is
        # the rest solution: p1 = 0, u = 0, cost 0.
        @test result.objective ≈ 0.0 atol = 1e-6
        @test ExaModels.solution(result, p)[1] ≈ 0.0 atol = 1e-4
    end

    @testset "with a nonzero initial velocity" begin
        # Pin p1 = 1 so the oscillator actually has to be driven to rest.
        core, dae, z, u, p = build(p1_bounds = (1.0, 1.0))

        result = madnlp(ExaModels.ExaModel(core); print_level = MadNLP.ERROR)
        @test result.status == MadNLP.SOLVE_SUCCEEDED
        @test result.objective > 0.1                     # real control effort was spent

        zsol = ExaModels.solution(result, z)
        @test zsol[2, 1, 1, 1] ≈ 1.0 atol = 1e-6         # initial condition honored
        @test zsol[3, 1, 1, 1] ≈ 0.0 atol = 1e-6

        # The augmented cost state is nondecreasing, its RHS being a sum of squares
        @test all(diff(vec(zsol[3, 1, :, K + 1])) .>= -1e-8)

        # Terminal cost equals the objective
        @test zsol[3, 1, N, K + 1] ≈ result.objective rtol = 1e-8
    end
end
