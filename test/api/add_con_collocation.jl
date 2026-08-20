
const TF = 5.0

order(::GaussLegendre, K) = 2K
order(::GaussRadau, K) = 2K - 1
order(::GaussLobatto, K) = 2K - 2

const MODES = push!(
    vec([(b, r) for b in BASES, r in (GaussRadau(), GaussLegendre())]),
    (DerivativeForm(), GaussLobatto()),
)

function terminal(core, zsol, N, K, Nz)
    last(core.weights.taus) ≈ 1 && return zsol[:, N, K + 1]
    if core.basis isa StateForm
        return [sum(core.weights.b[j + 1] * zsol[v, N, j + 1] for j in 0:K) for v in 1:Nz]
    end
    h = ExaModelsCollocation._hval(core.mesh)
    return [
        zsol[v, N, 1] + h[N] * sum(core.weights.b[j] * -zsol[v, N, j + 1] for j in 1:K)
        for v in 1:Nz
    ]
end

function solve_decay(basis, roots, N, K; Nz = 2, adaptive = false)
    core = CollocationExaCore(range(0.0, TF; length = N + 1), K; basis, roots, adaptive)
    @add_var_collocation(core, z, 1:Nz)
    @add_con_collocation(core, coll, z[v], -z[v] for v in 1:Nz)

    @add_con_continuity(core, cont, z)

    ExaModels.@add_con(core, ic, z[v, 1, 0] - val for (v, val) in [(1, 1.0), (2, 2.0)])

    model = ExaModels.ExaModel(core)
    result = madnlp(model; print_level = MadNLP.ERROR, tol = 1e-12)
    zsol = ExaModels.solution(result, z)
    return result, zsol, terminal(core, zsol, N, K, Nz), model, z
end

function same_residual(b1, b2)
    m1, m2 = b1(), b2()
    m1.meta.ncon == m2.meta.ncon || return false
    m1.meta.nvar == m2.meta.nvar || return false
    x = collect(range(0.1, 2.0; length = m1.meta.nvar))
    return ExaModels.NLPModels.cons(m1, x) ≈ ExaModels.NLPModels.cons(m2, x)
end

@testset "collocation residual" begin
    K, N = 3, 20

    @testset "$(nameof(typeof(b))), $(nameof(typeof(r)))" for (b, r) in MODES
        result, zsol, zf = solve_decay(b, r, N, K)

        @test result.status == MadNLP.SOLVE_SUCCEEDED
        @test zsol[1, 1, 1] ≈ 1.0 atol = 1e-8
        @test zsol[2, 1, 1] ≈ 2.0 atol = 1e-8

        @test zf[1] ≈ exp(-TF) rtol = 1e-3
        @test zf[2] ≈ 2 * zf[1] rtol = 1e-9
    end

    @testset "the two bases are the same discretization" begin
        for r in (GaussRadau(), GaussLegendre())
            _, zstate, sf = solve_decay(StateForm(), r, N, K)
            _, zderiv, df = solve_decay(DerivativeForm(), r, N, K)
            @test sf ≈ df rtol = 1e-8
            @test zstate ≈ zderiv rtol = 1e-8
        end
    end

    @testset "a row entry that varies with the collocation point" begin
        function decay_t(basis; N = 10, K = 3)
            core = CollocationExaCore(range(0.0, 1.0; length = N + 1), K; basis)
            @add_var_collocation(core, z, 1:1)
            mt = core.mesh.t
            itr = [(v, 2 * mt[i, k], i, k) for v in 1:1, i in 1:N, k in 1:K]
            @add_con_collocation(core, coll, z[v], -g * z[v] for (v, g, i, k) in itr)
            @add_con_continuity(core, cont, z)
            ExaModels.@add_con(core, ic, z[v, 1, 0] - 1.0 for v in 1:1)
            result = madnlp(ExaModels.ExaModel(core); print_level = MadNLP.ERROR, tol = 1e-12)
            return ExaModels.solution(result, z)[1, end, end]
        end

        sf, df = decay_t(StateForm()), decay_t(DerivativeForm())
        @test sf ≈ df rtol = 1e-8
        @test df ≈ exp(-1) rtol = 1e-6
    end

    @testset "convergence order" begin
        @testset "$(nameof(typeof(b))), $(nameof(typeof(r)))" for (b, r) in MODES
            errs = [abs(solve_decay(b, r, n, K)[3][1] - exp(-TF)) for n in (5, 10, 20)]
            @test issorted(errs; rev = true)
            rates = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
            @test all(≈(order(r, K); atol = 0.25), rates)
        end
    end
end

@testset "the spellings agree" begin
    N, K, Nz = 5, 3, 2
    nodes = range(0.0, 1.0; length = N + 1)

    written(basis, adaptive, nd = nodes) = () -> begin
        core = CollocationExaCore(nd, K; basis, adaptive)
        @add_var_collocation(core, z, 1:Nz)
        itr = [(v, i, k) for v in 1:Nz, i in 1:N, k in 1:K]
        @add_con_collocation(core, coll, z[v], -2 * t * z[v, i, k] for (v, i, k) in itr)
        @add_con_continuity(core, cont, z)
        ExaModels.ExaModel(core)
    end

    short(basis, adaptive) = () -> begin
        core = CollocationExaCore(nodes, K; basis, adaptive)
        @add_var_collocation(core, z, 1:Nz)
        @add_con_collocation(core, coll, z[v], -2 * t * z[v] for v in 1:Nz)
        @add_con_continuity(core, cont, z)
        ExaModels.ExaModel(core)
    end

    unknown(basis, nd = nodes) = () -> begin
        core = CollocationExaCore(nd, K; basis, unknown_horizon = true)
        @add_var_collocation(core, z, 1:Nz)
        @add_con_collocation(core, coll, z[v], -2 * t * z[v] for v in 1:Nz)
        @add_con_continuity(core, cont, z)
        ExaModels.ExaModel(core)
    end

    reordered(basis, adaptive) = () -> begin
        core = CollocationExaCore(nodes, K; basis, adaptive)
        @add_var_collocation(core, z, 1:Nz)
        itr = [(2.0, v) for v in 1:Nz]
        @add_con_collocation(core, coll, z[v], -a * t * z[v] for (a, v) in itr)
        @add_con_continuity(core, cont, z)
        ExaModels.ExaModel(core)
    end

    called(basis, adaptive) = () -> begin
        core = CollocationExaCore(nodes, K; basis, adaptive)
        core, z = add_var_collocation(core, 1:Nz)
        itr = [(v, i, k) for v in 1:Nz, i in 1:N, k in 1:K]
        core, _ = add_con_collocation(
            core, (z[v] => -2 * t * z[v, i, k] for (v, i, k, t) in itr),
        )
        core, _ = add_con_continuity(core, z)
        ExaModels.ExaModel(core)
    end

    crossed(basis, adaptive) = () -> begin
        core = CollocationExaCore(nodes, K; basis, adaptive)
        core, z = add_var_collocation(core, 1:Nz)
        itr = [(v,) for v in 1:Nz]
        core, _ = add_con_collocation(
            core, (z[v] => -2 * t * z[v, i, k] for (v, i, k, t) in itr),
        )
        core, _ = add_con_continuity(core, z)
        ExaModels.ExaModel(core)
    end

    @testset "$(nameof(typeof(b))), adaptive = $ad" for b in BASES, ad in (false, true)
        @test same_residual(written(b, ad), short(b, ad))
        @test same_residual(written(b, ad), reordered(b, ad))
        @test same_residual(written(b, ad), called(b, ad))
        @test same_residual(written(b, ad), crossed(b, ad))
    end

    @testset "an adaptive mesh reproduces the numeric one" begin
        for b in BASES
            @test same_residual(written(b, false), written(b, true))
            _, zn, fn = solve_decay(b, GaussRadau(), N, K)
            _, za, fa = solve_decay(b, GaussRadau(), N, K; adaptive = true)
            @test fa ≈ fn rtol = 1e-12
            @test za ≈ zn rtol = 1e-12
        end

        core = CollocationExaCore(nodes, K; adaptive = true)
        @test core.adaptive
        @test core.npar == N * K + N
    end

    @testset "an unknown horizon reproduces the numeric mesh" begin
        tnom = last(nodes) - first(nodes)
        for b in BASES
            m0, m1 = written(b, false)(), unknown(b)()
            @test m1.meta.nvar == m0.meta.nvar + 1
            x = collect(range(0.1, 2.0; length = m0.meta.nvar))
            @test ExaModels.NLPModels.cons(m1, [tnom; x]) ≈ ExaModels.NLPModels.cons(m0, x)
        end

        core = CollocationExaCore(nodes, K; unknown_horizon = true)
        @test core.unknown_horizon
        @test core.nvar == 1 && ExaModels.get_start(ExaModels.ExaModel(core), core.tscale) ≈ [tnom]

        @test_throws ArgumentError CollocationExaCore(nodes, K; adaptive = true, unknown_horizon = true)
    end

    @testset "an unknown horizon off t0 = 0, span = 1" begin
        t0, tnom, ts = 2.0, 3.0, 4.5
        nom = collect(range(t0, t0 + tnom; length = N + 1))
        stretched = t0 .+ (nom .- t0) .* (ts / tnom)

        for b in BASES
            m1 = unknown(b, nom)()
            x = collect(range(0.1, 2.0; length = m1.meta.nvar - 1))
            at(nd) = ExaModels.NLPModels.cons(written(b, false, nd)(), x)

            @test ExaModels.NLPModels.cons(m1, [ts; x]) ≈ at(stretched)
            @test ExaModels.NLPModels.cons(m1, [tnom; x]) ≈ at(nom)
            @test !isapprox(ExaModels.NLPModels.cons(m1, [tnom; x]), at(stretched))
        end
    end

    @testset "three horizons, built together" begin
        tfs = [1.0, 2.0, 5.0]
        function fam(basis)
            core = CollocationExaCore([range(0.0, tf; length = N + 1) for tf in tfs], K; basis)
            @add_var_collocation(core, z, 1:Nz)
            @add_con_collocation(core, coll, z[v,m], -2 * t * z[v,m] for v in 1:Nz, m in 1:3)
            @add_con_continuity(core, cont, z)
            ExaModels.@add_con(core, ic, z[v,m,1,0] - v for v in 1:Nz, m in 1:3)
            model = ExaModels.ExaModel(core)
            r = madnlp(model; print_level = MadNLP.ERROR, tol = 1e-12)
            return model, ExaModels.solution(r, z)
        end
        function one(basis, tf)
            core = CollocationExaCore(range(0.0, tf; length = N + 1), K; basis)
            @add_var_collocation(core, z, 1:Nz)
            @add_con_collocation(core, coll, z[v], -2 * t * z[v] for v in 1:Nz)
            @add_con_continuity(core, cont, z)
            ExaModels.@add_con(core, ic, z[v,1,0] - v for v in 1:Nz)
            model = ExaModels.ExaModel(core)
            r = madnlp(model; print_level = MadNLP.ERROR, tol = 1e-12)
            return model, ExaModels.solution(r, z)
        end

        for b in BASES
            m, zf = fam(b)
            singles = [one(b, tf) for tf in tfs]
            @test m.M == 3 && m.N == N
            @test m.meta.nvar == sum(s.meta.nvar for (s, _) in singles)
            @test m.meta.ncon == sum(s.meta.ncon for (s, _) in singles)

            for (j, (_, zs)) in enumerate(singles)
                @test zf[:, j, :, :] ≈ zs rtol = 1e-9
            end
        end

        core = CollocationExaCore(nodes, K)
        @test core.M == 1
        core, z1 = add_var_collocation(core, 1:Nz)
        @test z1.dims == (1:Nz,) && z1.mesh == 1

        fcore = CollocationExaCore([nodes, nodes], K)
        fcore, zf = add_var_collocation(fcore, 1:Nz)
        fcore, zp = add_var_collocation(fcore, 1:Nz; mesh = 2)
        @test zf.dims == (1:Nz, 1:2) && zf.mesh === nothing
        @test zp.dims == (1:Nz,) && zp.mesh == 2
        @test_throws ArgumentError add_var_collocation(fcore, 1:Nz; mesh = 3)
    end

    @testset "a second block completes the same way" begin
        both(short_u) = () -> begin
            core = CollocationExaCore(nodes, K)
            @add_var_collocation(core, z, 1:Nz)
            @add_var_collocation(core, u, 1:Nz; include_boundary = false)
            if short_u
                @add_con_collocation(core, coll, z[v], -z[v] + u[v] for v in 1:Nz)
            else
                @add_con_collocation(core, coll, z[v], -z[v, i, k] + u[v, i, k] for v in 1:Nz)
            end
            @add_con_continuity(core, cont, z)
            ExaModels.ExaModel(core)
        end
        @test same_residual(both(true), both(false))
    end

    @testset "end and begin index as written" begin
        coeff(which) = () -> begin
            core = CollocationExaCore(nodes, K)
            @add_var_collocation(core, z)
            a = [1.0, 2.0, 10.0]
            if which === :literal
                @add_con_collocation(core, coll, z[], -10.0 * z)
            elseif which === :atend
                @add_con_collocation(core, coll, z[], -a[end] * z)
            else
                @add_con_collocation(core, coll, z[], -a[begin + 2] * z)
            end
            ExaModels.ExaModel(core)
        end

        @testset "$w" for w in (:atend, :atbegin)
            @test same_residual(coeff(w), coeff(:literal))
        end
    end
end

@testset "what the macro refuses" begin
    core = CollocationExaCore(range(0.0, 1.0; length = 5), 3)
    @add_var_collocation(core, z, 1:2)

    @test_throws Exception @macroexpand @add_con_collocation(
        core, coll, z[v], -t * z[v] for (v, i, k, t) in itr)
    @test_throws Exception @macroexpand @add_con_collocation(
        core, coll, z[v], -z[v] for (v, i) in itr)
    @test_throws Exception @macroexpand @add_con_collocation(
        core, coll, z[v], -z[v] for (i, k, v) in itr)

    @test_throws ArgumentError @add_con_collocation(core, coll, z[v, 1], -z[v, 1] for v in 1:2)
    @test_throws ArgumentError @add_con_collocation(core, coll, z, -z[v] for v in 1:2)
    @test_throws ArgumentError add_con_collocation(core, (-z[v, i, k] for (v, i, k, t) in
        [(v, i, k) for v in 1:2, i in 1:4, k in 1:3]))

    core, u = add_var_collocation(core, 1:1; include_boundary = false)
    @test_throws ArgumentError @add_con_collocation(core, coll, u[v], -u[v] for v in 1:1)

    @test_throws ArgumentError add_con_collocation(
        core, (z[v] => -z[v, 1, 1] for (v,) in [(v,) for v in 1:2]))

    @test_throws ArgumentError add_con_collocation(
        core, (z[v] => -z[v, i, k] for (v, i, k, t, extra) in
            [(v, i, k) for v in 1:2, i in 1:4, k in 1:3]))
    @test_throws ArgumentError add_con_collocation(
        core, (z[1] => -z[1, 1, 1] for r in [(v,) for v in 1:2]))
end

@testset "no leading dimensions" begin
    N, K = 5, 3

    @testset "$(nameof(typeof(b)))" for b in BASES
        core = CollocationExaCore(range(0.0, TF; length = N + 1), K; basis = b)
        @add_var_collocation(core, z)
        @test ExaModelsCollocation._nleading(z) == 0

        @add_con_collocation(core, coll, z[], -z)
        @add_con_continuity(core, cont, z)
        ExaModels.@add_con(core, ic, z[1, 0] - 1.0)

        @test core.ncon == N * K + (N - 1) + 1
        result = madnlp(ExaModels.ExaModel(core); print_level = MadNLP.ERROR, tol = 1e-12)
        @test result.status == MadNLP.SOLVE_SUCCEEDED

        _, ref = solve_decay(b, GaussRadau(), N, K)
        zsol = ExaModels.solution(result, z)
        @test size(zsol) == (N, K + 1)
        @test zsol ≈ ref[1, :, :] rtol = 1e-8
    end

    @testset "written out with an iterator" begin
        build(bare) = () -> begin
            core = CollocationExaCore(range(0.0, TF; length = N + 1), K)
            @add_var_collocation(core, z)
            if bare
                @add_con_collocation(core, coll, z[], -z)
            else
                itr = [(i, k) for i in 1:N, k in 1:K]
                @add_con_collocation(core, coll, z[], -z[i, k] for (i, k) in itr)
            end
            @add_con_continuity(core, cont, z)
            ExaModels.ExaModel(core)
        end
        @test same_residual(build(true), build(false))
    end

    @testset "a bare target names the block" begin
        build(bare) = () -> begin
            core = CollocationExaCore(range(0.0, TF; length = N + 1), K)
            @add_var_collocation(core, z)
            if bare
                @add_con_collocation(core, coll, z, -z)
            else
                @add_con_collocation(core, coll, z[], -z)
            end
            @add_con_continuity(core, cont, z)
            ExaModels.ExaModel(core)
        end
        @test same_residual(build(true), build(false))
    end

    @testset "crossed into rows that carry no slot" begin
        build(bare) = () -> begin
            core = CollocationExaCore(range(0.0, TF; length = N + 1), K)
            core, z = add_var_collocation(core)
            if bare
                @add_con_collocation(core, coll, z[], -z)
            else
                core, _ = add_con_collocation(
                    core, (z[] => -z[i, k] for (i, k, t) in [()]),
                )
            end
            core, _ = add_con_continuity(core, z)
            ExaModels.ExaModel(core)
        end
        @test same_residual(build(true), build(false))
    end
end

@testset "two leading dimensions" begin
    N, K, Nz, Nc = 5, 3, 2, 3
    core = CollocationExaCore(range(0.0, 1.0; length = N + 1), K)
    @add_var_collocation(core, z, 1:Nz, 1:Nc)

    itr = [(v, c) for v in 1:Nz, c in 1:Nc]
    @add_con_collocation(core, coll, z[v, c], -z[v, c] for (v, c) in itr)
    @add_con_continuity(core, cont, z)

    @test core.ncon == Nz * Nc * N * K + Nz * Nc * (N - 1)
    @test :cont in propertynames(core)

    @testset "crossed into two-slot rows" begin
        build(macroform) = () -> begin
            core = CollocationExaCore(range(0.0, 1.0; length = N + 1), K)
            core, z = add_var_collocation(core, 1:Nz, 1:Nc)
            rows = [(v, c) for v in 1:Nz, c in 1:Nc]
            if macroform
                @add_con_collocation(core, coll, z[v, c], -t * z[v, c] for (v, c) in rows)
            else
                core, _ = add_con_collocation(
                    core, (z[v, c] => -t * z[v, c, i, k] for (v, c, i, k, t) in rows),
                )
            end
            core, _ = add_con_continuity(core, z)
            ExaModels.ExaModel(core)
        end
        @test same_residual(build(true), build(false))
    end
end

@testset "anonymous helpers register nothing" begin
    K, Nz = 3, 2
    core = CollocationExaCore(range(0.0, 1.0; length = 5), K)

    z = @add_var_collocation(core, 1:Nz)
    coll = @add_con_collocation(core, z[v], -z[v] for v in 1:Nz)
    cont = @add_con_continuity(core, z)

    @test coll isa ExaModels.Constraint
    @test cont isa ExaModels.Constraint
    @test core.refs == (;)
end
