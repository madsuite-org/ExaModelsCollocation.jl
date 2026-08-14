# add_con_collocation: both residual forms against theory, and every spelling of the same one.
#
# The residual tests solve dz/dt = -z, z(0) = 1  =>  z(t) = exp(-t) through the helpers.
# StateForm (10.7) and DerivativeForm (10.8) are algebraically equivalent discretizations over
# the same variables, so on the same roots they must land on the same answer; and the terminal
# error must decay at the order the collocation family gives. Together those pin the weight
# indexing and the row mapping in a way a shape check cannot.

const TF = 5.0

# Convergence order of the terminal value, per family (Biegler Table 10.3)
order(::GaussLegendre, K) = 2K
order(::GaussRadau, K) = 2K - 1
order(::GaussLobatto, K) = 2K - 2

const MODES = push!(
    vec([(b, r) for b in BASES, r in (GaussRadau(), GaussLegendre())]),
    (DerivativeForm(), GaussLobatto()),
)

# z(TF) as the state polynomial at tau = 1: a variable when the family has a point there,
# the continuity combination otherwise (Biegler 10.14b / 10.15b).
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

# Build and solve the decay problem. Two components share one right-hand side, so v goes in
# the iterator and one call covers both.
function solve_decay(basis, roots, N, K; Nz = 2, adaptive = false)
    core = CollocationExaCore(range(0.0, TF; length = N + 1), K; basis, roots, adaptive)
    @add_var_collocation(core, z, 1:Nz)
    @add_con_collocation(core, coll, z[v], -z[v] for v in 1:Nz)

    # The same call under either basis: DerivativeForm takes the right-hand side from the
    # collocation call above rather than being handed it again.
    @add_con_continuity(core, cont, z)

    ExaModels.@add_con(core, ic, z[v, 1, 0] - val for (v, val) in [(1, 1.0), (2, 2.0)])

    model = ExaModels.ExaModel(core)
    result = madnlp(model; print_level = MadNLP.ERROR, tol = 1e-12)
    # solution() returns a plain 1-based array, so k = 0,...,K lands on 1,...,K+1
    zsol = ExaModels.solution(result, z)
    return result, zsol, terminal(core, zsol, N, K, Nz), model, z
end

# Two spellings of one residual must build the same rows, which a solve tolerance would hide
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

        # Right answer to within this family's discretization error, and linear in the
        # initial condition since the equation is
        @test zf[1] ≈ exp(-TF) rtol = 1e-3
        @test zf[2] ≈ 2 * zf[1] rtol = 1e-9
    end

    @testset "the two bases are the same discretization" begin
        # Same roots, same variables, algebraically equivalent residuals -> same solution.
        # This is what catches a weight matrix that is subtly wrong for its basis.
        for r in (GaussRadau(), GaussLegendre())
            _, zstate, sf = solve_decay(StateForm(), r, N, K)
            _, zderiv, df = solve_decay(DerivativeForm(), r, N, K)
            @test sf ≈ df rtol = 1e-8
            @test zstate ≈ zderiv rtol = 1e-8
        end
    end

    @testset "a row entry that varies with the collocation point" begin
        # dz/dt = -2t z, z(0) = 1 => exp(-t^2), with the coefficient carried in the row rather
        # than read as t, so it varies over the interval the weights sum across.
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
        # Halving h must shrink the terminal error by 2^p, p set by the family. A stencil
        # that is merely consistent but mis-weighted loses order here even when it converges.
        @testset "$(nameof(typeof(b))), $(nameof(typeof(r)))" for (b, r) in MODES
            errs = [abs(solve_decay(b, r, n, K)[3][1] - exp(-TF)) for n in (5, 10, 20)]
            @test issorted(errs; rev = true)
            rates = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
            @test all(≈(order(r, K); atol = 0.25), rates)
        end
    end
end

@testset "the spellings agree" begin
    # The macro supplies i, k and t and completes a block short of its mesh indices, so the
    # short spelling and the written-out one must be the same rows, not merely the same
    # answer. dz/dt = -2t z over two components, which reads the mesh, the slot and t at once.
    N, K, Nz = 5, 3, 2
    nodes = range(0.0, 1.0; length = N + 1)

    written(basis, adaptive) = () -> begin
        core = CollocationExaCore(nodes, K; basis, adaptive)
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

    # the slot indices are found by tracing the target, so they need not lead the row
    reordered(basis, adaptive) = () -> begin
        core = CollocationExaCore(nodes, K; basis, adaptive)
        @add_var_collocation(core, z, 1:Nz)
        itr = [(2.0, v) for v in 1:Nz]
        @add_con_collocation(core, coll, z[v], -a * t * z[v] for (a, v) in itr)
        @add_con_continuity(core, cont, z)
        ExaModels.ExaModel(core)
    end

    # and the function is the macro with the target moved inside and the mesh written out
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

    # and the same call again with the mesh left out of the iterator. The pattern binds three
    # names past a row where called binds one, which is what tells the two apart.
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
        # Same nodes, same equations: the parameter mesh must be the same discretization,
        # not merely a close one, and t must come off it rather than out of the row.
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

    @testset "a second block completes the same way" begin
        # The rewrite reaches every operand, not just the block being collocated, and it
        # cannot know their arities: u is completed at trace time like z is.
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

    # t is supplied whatever the mesh, so carrying it would be a second definition
    @test_throws Exception @macroexpand @add_con_collocation(
        core, coll, z[v], -t * z[v] for (v, i, k, t) in itr)
    # i and k name the mesh, so one of them alone is not a row the helper can read
    @test_throws Exception @macroexpand @add_con_collocation(
        core, coll, z[v], -z[v] for (v, i) in itr)
    # and given both, they are what the row ends in
    @test_throws Exception @macroexpand @add_con_collocation(
        core, coll, z[v], -z[v] for (i, k, v) in itr)

    # an index count that is neither a slot of the block nor a coefficient of it
    @test_throws ArgumentError @add_con_collocation(core, coll, z[v, 1], -z[v, 1] for v in 1:2)
    # and a generator that never names a slot at all
    @test_throws ArgumentError add_con_collocation(core, (-z[v, i, k] for (v, i, k, t) in
        [(v, i, k) for v in 1:2, i in 1:4, k in 1:3]))

    # a block with no k = 0 boundary node has nothing for the stencil to anchor on
    core, u = add_var_collocation(core, 1:1; include_boundary = false)
    @test_throws ArgumentError @add_con_collocation(core, coll, u[v], -u[v] for v in 1:1)

    # rows that do not reach the mesh at all
    @test_throws ArgumentError add_con_collocation(
        core, (z[v] => -z[v, 1, 1] for (v,) in [(v,) for v in 1:2]))

    # a pattern binding two names past its rows is neither spelling: one short of the mesh
    # crossed in, one long for rows that carry it
    @test_throws ArgumentError add_con_collocation(
        core, (z[v] => -z[v, i, k] for (v, i, k, t, extra) in
            [(v, i, k) for v in 1:2, i in 1:4, k in 1:3]))
    # and a row bound as a whole names nothing the residual can read
    @test_throws ArgumentError add_con_collocation(
        core, (z[1] => -z[1, 1, 1] for r in [(v,) for v in 1:2]))
end

@testset "no leading dimensions" begin
    # A block declared with no dimensions is a scalar state, z[i,k], so its rows carry no
    # slot at all and every stencil takes its shortest branch. With nothing left to vary
    # over, the right-hand side needs no iterator either. Pinned against the same problem as
    # a one-component block rather than against a tolerance: the two are the same
    # discretization of the same equation, so they must agree to solver tolerance.
    N, K = 5, 3

    @testset "$(nameof(typeof(b)))" for b in BASES
        core = CollocationExaCore(range(0.0, TF; length = N + 1), K; basis = b)
        @add_var_collocation(core, z)
        @test ExaModelsCollocation._nleading(z) == 0

        @add_con_collocation(core, coll, z[], -z)
        @add_con_continuity(core, cont, z)
        ExaModels.@add_con(core, ic, z[1, 0] - 1.0 for _ in 1:1)

        @test core.ncon == N * K + (N - 1) + 1
        result = madnlp(ExaModels.ExaModel(core); print_level = MadNLP.ERROR, tol = 1e-12)
        @test result.status == MadNLP.SOLVE_SUCCEEDED

        # component 1 of the one-dimensional block carries the same z(0) = 1
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

    @testset "crossed into rows that carry no slot" begin
        # The shortest branch of every stencil, reached through the function: one data row
        # holding nothing, so the crossed row is the mesh indices alone.
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
    # The stencils have a branch per leading-dimension count, and one call covers every slot.
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
