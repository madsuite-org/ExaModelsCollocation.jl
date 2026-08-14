@testset "interpolate" begin
    N, K = 10, 3

    @testset "$(nameof(typeof(b))), $(nameof(typeof(r)))" for (b, r) in MODES
        result, zsol, _, m, z = solve_decay(b, r, N, K)

        @test interpolate(m, result, z, 0.0) ≈ [1.0, 2.0] atol = 1e-8
        @test interpolate(m, result, z, TF / 2)[1] ≈ exp(-TF / 2) rtol = 1e-3
        @test interpolate(m, result, z, TF)[1] ≈ exp(-TF) rtol = 1e-3
        @test interpolate(m, result, z, TF / 2)[2] ≈ 2 * interpolate(m, result, z, TF / 2)[1] rtol = 1e-9

        @test interpolate(m, result, z, m.nodes[4]) ≈ zsol[:, 4, 1] atol = 1e-12
        @test interpolate(m, zsol, z, TF / 2) ≈ interpolate(m, result, z, TF / 2)

        ts = [0.0, TF / 4, TF]
        vs = interpolate(m, result, z, ts)
        @test length(vs) == length(ts)
        @test all(vs[j] ≈ interpolate(m, result, z, ts[j]) for j in eachindex(ts))
    end

    @testset "shape follows the block" begin
        core = CollocationExaCore(range(0.0, 1.0; length = N + 1), K)
        core, y = add_var_collocation(core)
        core, w = add_var_collocation(core, 1:3, 1:2)
        core, u = add_var_collocation(core, 1:3; include_boundary = false)

        @test interpolate(core, fill(2.0, N, K + 1), y, 0.5) isa Real
        @test interpolate(core, fill(2.0, N, K + 1), y, 0.5) ≈ 2.0
        @test size(interpolate(core, fill(1.0, 3, 2, N, K + 1), w, 0.5)) == (3, 2)
        @test size(interpolate(core, fill(1.0, 3, N, K), u, 0.5)) == (3,)
    end

    @testset "rejects a time off the mesh" begin
        core = CollocationExaCore(range(0.0, 1.0; length = N + 1), K)
        core, z = add_var_collocation(core, 1:1)
        zsol = fill(1.0, 1, N, K + 1)

        @test_throws ArgumentError interpolate(core, zsol, z, -0.1)
        @test_throws ArgumentError interpolate(core, zsol, z, 1.1)
        @test interpolate(core, zsol, z, 1.0) ≈ [1.0]
    end
end
