# Van der Pol oscillator control problem

using MadNLP

@testset "Van der Pol optimal control" begin
    # min z3(tf)  s.t.  z1' = z2
    #                   z2' = th1 z2 (1 - z1^2) - z1 + u + p2 cos(t)
    #                   z3' = z1^2 + z2^2 + u^2
    #                   z(t0) = (0, p1, 0)
    tf, N, K = 5.0, 20, 3

    function build(; p1_bounds = (-2.0, 2.0))
        # Create CollocationExaCore
        core = CollocationExaCore(range(0.0, tf; length = N + 1), K)

        # Create CollocationVariables
        @add_var_collocation(core, z1)  # z1[i,k]
        @add_var_collocation(core, z2)  # z2[i,k]
        @add_var_collocation(core, z3)  # z3[i,k]
        @add_var_collocation(core, u; include_boundary = false)  # k = 1,...,K

        # Create standard ExaModels variables and parameters
        ExaModels.@add_var(core, p, 1:2;
            lvar = [p1_bounds[1], -1.0], 
            uvar = [p1_bounds[2], 1.0], 
            start = [1.0, 0.0]
        )
        ExaModels.@add_par(core, theta, [1.0])

        # Create iterator for @add_con_collocation
        mesh_t = core.mesh.t
        iter_coll = [(i,k,mesh_t[i,k]) for i in 1:N, k in 1:K]

        # Create collocation constraints
        # Create Constraint named coll1, to CollocationVariable z1, with right-hand side function dz1/dt = z2
        @add_con_collocation(core, coll1, z1,
            z2[i,k]
            for (i,k,t) in iter_coll
        )
        # Create Constraint named coll2, to CollocationVariable z2, with right-hand side function dz2/dt = ...
        @add_con_collocation(core, coll2, z2,
            theta[1]*z2[i,k]*(1 - z1[i,k]^2) - z1[i,k] + u[i,k] + p[2]*cos(t)
            for (i,k,t) in iter_coll
        )
        # Create Constraint named coll3, to CollocationVariable z3, with right-hand side function dz3/dt = ...
        @add_con_collocation(core, coll3, z3,
            z1[i,k]^2 + z2[i,k]^2 + u[i,k]^2
            for (i,k,t) in iter_coll
        )

        # Create continuity constraints 
        # NOTE: we need to do this separately, one for each CollocationVariable
        # if we created z[1:3,i,k], we would only need @add_con_continuity(core, cont, z)
        @add_con_continuity(core, cont1, z1)
        @add_con_continuity(core, cont2, z2)
        @add_con_continuity(core, cont3, z3)

        # Create initial condition constraints
        # NOTE: we need to do this separately, one for each CollocationVariable
        # if we created z[1:3,i,k], we would only need two @add_con calls
        ExaModels.@add_con(core, ic1, z1[1, 0] for _ in 1:1)
        ExaModels.@add_con(core, ic2, z2[1, 0] - p[1] for _ in 1:1)
        ExaModels.@add_con(core, ic3, z3[1, 0] for _ in 1:1)

        # Create objective function
        ExaModels.@add_obj(core, z3[N, K] for _ in 1:1)

        return core, (z1, z2, z3), u, p
    end

    @testset "as originally posed" begin
        core, _, _, p = build()
        # three state blocks over k = 0,…,K, one control block over k = 1,…,K, and p
        @test core.nvar == 3 * N * (K + 1) + N * K + 2
        @test :coll2 in propertynames(core)

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
        core, (z1, z2, z3), _, _ = build(p1_bounds = (1.0, 1.0))

        result = madnlp(ExaModels.ExaModel(core); print_level = MadNLP.ERROR)
        @test result.status == MadNLP.SOLVE_SUCCEEDED
        @test result.objective > 0.1                     # real control effort was spent

        # 1-based, so k = 0,...,K lands on 1,...,K+1
        z2sol, z3sol = ExaModels.solution(result, z2), ExaModels.solution(result, z3)
        @test z2sol[1, 1] ≈ 1.0 atol = 1e-6              # initial condition honored
        @test z3sol[1, 1] ≈ 0.0 atol = 1e-6

        # The augmented cost state is nondecreasing, its RHS being a sum of squares
        @test all(diff(vec(z3sol[:, K + 1])) .>= -1e-8)

        # Terminal cost equals the objective
        @test z3sol[N, K + 1] ≈ result.objective rtol = 1e-8
    end
end
