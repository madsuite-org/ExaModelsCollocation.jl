# The public helpers, one testset per function, and the two residual forms against theory.
#
# The residual tests solve dz/dt = -z, z(0) = 1  =>  z(t) = exp(-t) through the helpers.
# StateForm (10.7) and DerivativeForm (10.8) are algebraically equivalent discretizations over
# the same variables, so on the same roots they must land on the same answer; and the terminal
# error must decay at the order the collocation family gives. Together those pin the weight
# indexing and the row mapping in a way a shape check cannot.

using MadNLP

@testset "CollocationExaCore construction" begin
    nodes = range(0.0, 5.0; length = 21)
    K = 3

    @testset "mesh geometry" begin
        core = CollocationExaCore(nodes, K)
        @test core.N == 20
        @test core.K == 3
        @test length(core.mesh.h) == 20
        @test size(core.mesh.t) == (20, K)
        @test core.mesh.t[1, end] <= core.nodes[2]    # collocation times stay inside their interval
        @test all(core.mesh.h .≈ 0.25)
        @test length(core.weights.taus) == K
        @test core.adaptive == false

        # Derived properties read flat but are not stored twice
        @test core.nodes === core.mesh.nodes
        @test core.weights === core.mode.weights
        @test core.roots === core.mode.roots
        @test fieldnames(ExaModelsCollocation.CollocationTag) ==
            (:mode, :mesh, :blocks, :residuals)
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

        # a tag attaches to a core that already holds variables, keeping them
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
end

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

    @testset "the handle is a Variable that carries its layout" begin
        core = CollocationExaCore(nodes, K)
        core, z = add_var_collocation(core, 1:nz)

        # it indexes exactly like an ExaModels.Variable, which is why nothing else changed
        @test z isa ExaModels.AbstractVariable
        @test z isa CollocationVariable
        @test z[1, 1, 0] isa ExaModels.AbstractNode

        # the block layout rides on the handle, so there is no side table to look it up in
        @test ExaModelsCollocation._nleading(z) == 1
    end

    @testset "the name is an optional Val keyword, as in add_var" begin
        core = CollocationExaCore(nodes, K)

        # anonymous: laid out and tracked, but not registered under any name
        core, y = add_var_collocation(core, 1:nz, 1:Nc)
        @test core.refs == (;)
        @test y.dims == (1:nz, 1:Nc)
        @test y in core.blocks

        # named, exactly the way ExaModels.add_var takes it -- and registered upstream, so
        # the collocation handle is what core.z and model.z give back
        core, z = add_var_collocation(core, 1:nz, 1:Nc; name = Val(:z))
        @test core.z === z
        @test :z in propertynames(core)
        @test ExaModels.ExaModel(core).z === z
    end

    @testset "start and bounds pass through" begin
        core = CollocationExaCore(nodes, K)
        start = fill(0.5, nz, Nc, N, K + 1)
        core, z = add_var_collocation(core, 1:nz, 1:Nc; start = start, lvar = -2.0, uvar = 2.0)

        @test all(core.x0 .== 0.5)
        @test all(core.lvar .== -2.0)
        @test all(core.uvar .== 2.0)

        # and the ExaModels bound accessors reach a CollocationVariable
        m = ExaModels.ExaModel(core)
        @test all(ExaModels.get_start(m, z) .== 0.5)
        @test all(ExaModels.get_lvar(m, z) .== -2.0)
        ExaModels.set_start!(m, z, fill(0.25, z.length))
        @test all(ExaModels.get_start(m, z) .== 0.25)
        @test_throws DimensionMismatch ExaModels.set_start!(m, z, [1.0])
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
end

const TF = 5.0

# Convergence order of the terminal value, per family (Biegler Table 10.3)
order(::GaussLegendre, K) = 2K
order(::GaussRadau, K) = 2K - 1
order(::GaussLobatto, K) = 2K - 2

# Modes that pair: GaussLobatto collocates tau = 0, which StateForm cannot anchor
const MODES = push!(
    vec([(b, r) for b in (StateForm(), DerivativeForm()),
         r in (GaussRadau(), GaussLegendre())]),
    (DerivativeForm(), GaussLobatto()),
)

# z(TF) as the state polynomial at tau = 1: a variable when the family has a point there,
# the continuity combination otherwise (Biegler 10.14b / 10.15b).
function terminal(core, zsol, N, K, Nz)
    last(core.weights.taus) ≈ 1 && return zsol[:, N, K + 1]
    if core.basis isa StateForm
        return [sum(core.weights.b[j + 1] * zsol[v, N, j + 1] for j in 0:K) for v in 1:Nz]
    end
    return [
        zsol[v, N, 1] + core.mesh.h[N] * sum(core.weights.b[j] * -zsol[v, N, j + 1] for j in 1:K)
        for v in 1:Nz
    ]
end

# The rows a collocation generator runs over, for a block with one leading dimension: z's own
# index leads, the mesh entries trail. t rides in the row only on a numeric mesh -- a
# parameter block is a graph node, and one of those cannot ride in an iterator tuple.
mesh_rows(core, vs) = core.adaptive ?
    vec([(v, i, k) for v in vs, i in 1:core.N, k in 1:core.K]) :
    vec([(v, i, k, core.mesh.t[i, k]) for v in vs, i in 1:core.N, k in 1:core.K])

# Build and solve the decay problem. Two components share one right-hand side, so v goes in
# the iterator and one call covers both.
function solve_decay(basis, roots, N, K; Nz = 2, adaptive = false)
    core = CollocationExaCore(range(0.0, TF; length = N + 1), K; basis, roots, adaptive)
    @add_var_collocation(core, z, 1:Nz)

    itr = mesh_rows(core, 1:Nz)
    if adaptive
        # an adaptive mesh carries no t in the row; the rhs is autonomous either way
        @add_con_collocation(core, coll, z, -z[v, i, k] for (v, i, k) in itr)
    else
        @add_con_collocation(core, coll, z, -z[v, i, k] for (v, i, k, t) in itr)
    end

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

@testset "collocation residual" begin
    K, N = 3, 20

    @testset "$(nameof(typeof(b))), $(nameof(typeof(r)))" for (b, r) in MODES
        result, zsol, zf = solve_decay(b, r, N, K)

        @test result.status == MadNLP.SOLVE_SUCCEEDED
        @test zsol[1, 1, 1] ≈ 1.0 atol = 1e-8              # initial conditions honored
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

@testset "adaptive mesh" begin
    N, K = 10, 3

    @testset "reproduces the numeric mesh, $(nameof(typeof(b)))" for
            b in (StateForm(), DerivativeForm())

        # Same nodes, same equations: the parameter mesh must be the same discretization,
        # not merely a close one.
        _, zn, fn = solve_decay(b, GaussRadau(), N, K)
        _, za, fa, model, z = solve_decay(b, GaussRadau(), N, K; adaptive = true)
        @test fa ≈ fn rtol = 1e-12
        @test za ≈ zn rtol = 1e-12

        # h and t are parameters, and the row loses its t entry
        core = CollocationExaCore(range(0.0, TF; length = N + 1), K; basis = b, adaptive = true)
        @test core.adaptive
        @test core.npar == N * K + N
        @test length(first(mesh_rows(core, 1:2))) == 3              # (v, i, k)
        @test length(first(mesh_rows(
            CollocationExaCore(range(0.0, TF; length = N + 1), K; basis = b), 1:2))) == 4
    end

    @testset "set_nodes! moves the mesh without rebuilding" begin
        # A rhs that genuinely depends on t, so a mesh update that moved h but not t would
        # show up here: dz/dt = -2t z, z(0) = 1 => exp(-t^2).
        function build(adaptive; nodes = range(0.0, 1.0; length = 11), K = 3)
            core = CollocationExaCore(nodes, K; adaptive)
            @add_var_collocation(core, z, 1:1)
            itr = mesh_rows(core, 1:1)
            if adaptive
                tp = core.mesh.tpar
                @add_con_collocation(core, coll, z, -2 * tp[i, k] * z[v, i, k] for (v, i, k) in itr)
            else
                @add_con_collocation(core, coll, z, -2 * t * z[v, i, k] for (v, i, k, t) in itr)
            end
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
        @test m.mesh.h ≈ diff(graded)

        # the moved mesh must match one built on those nodes from scratch
        mr, zr = build(false; nodes = graded)
        @test terminal_of(m, z) ≈ terminal_of(mr, zr) rtol = 1e-10
    end

    @testset "set_nodes! rejects what it cannot do" begin
        core = CollocationExaCore(range(0.0, 1.0; length = 6), 3)
        @test_throws ArgumentError set_nodes!(core, collect(range(0.0, 2.0; length = 6)))

        acore = CollocationExaCore(range(0.0, 1.0; length = 6), 3; adaptive = true)
        # changing the interval count changes the variable count, so it needs a rebuild
        @test_throws DimensionMismatch set_nodes!(acore, collect(range(0.0, 1.0; length = 7)))
        @test_throws ArgumentError set_nodes!(acore, [0.0, 0.4, 0.2, 0.6, 0.8, 1.0])
    end
end

@testset "continuity ties every slot the collocation calls cover" begin
    # The slots come off the recorded collocation calls, for either basis, so those calls
    # come first and a slot left uncollocated is an error rather than a silent gap.
    for b in (StateForm(), DerivativeForm())
        core = CollocationExaCore(range(0.0, 1.0; length = 5), 3; basis = b)
        @add_var_collocation(core, z, 1:2)

        @test_throws ArgumentError add_con_continuity(core, z)     # nothing collocated yet

        itr = mesh_rows(core, 1:1)                                 # covers z[1] alone
        @add_con_collocation(core, coll, z, -z[v, i, k] for (v, i, k, t) in itr)
        @test_throws ArgumentError add_con_continuity(core, z)     # z[2] still uncovered

        itr2 = mesh_rows(core, 2:2)
        @add_con_collocation(core, coll2, z, -z[v, i, k] for (v, i, k, t) in itr2)
        @add_con_continuity(core, cont, z)
        @test :cont in propertynames(core)
        @test core.cont isa ExaModels.Constraint
    end

    @testset "no slot may be covered twice" begin
        core = CollocationExaCore(range(0.0, 1.0; length = 5), 3)
        @add_var_collocation(core, z, 1:1)
        itr = mesh_rows(core, 1:1)
        @add_con_collocation(core, c1, z, -z[v, i, k] for (v, i, k, t) in itr)
        @add_con_collocation(core, c2, z, -2 * z[v, i, k] for (v, i, k, t) in itr)
        @test_throws ArgumentError add_con_continuity(core, z)
    end

    @testset "a non-collocation variable is refused" begin
        core = CollocationExaCore(range(0.0, 1.0; length = 5), 3)
        core, w = ExaModels.add_var(core, 1:2)
        @test_throws ArgumentError add_con_continuity(core, w)

        # so is a block with no k = 0 boundary node
        core, u = add_var_collocation(core, 1:1; include_boundary = false)
        itr = mesh_rows(core, 1:1)
        @test_throws ArgumentError add_con_collocation(
            core, u, (-u[v, i, k] for (v, i, k, t) in itr))
    end

    @testset "rows too short for the block are refused" begin
        # z declares two leading dimensions, so a row must read (v, c, …, i, k, t)
        core = CollocationExaCore(range(0.0, 1.0; length = 5), 3)
        @add_var_collocation(core, z, 1:2, 1:2)
        short = mesh_rows(core, 1:2)                               # (v, i, k, t)
        @test_throws ArgumentError add_con_collocation(
            core, z, (-z[v, 1, i, k] for (v, i, k, t) in short))
    end
end

@testset "the macro is the function with core rebound" begin
    # The macro adds nothing to the function but the name binding and the core rebind, so on
    # the same rows the two must build identical residuals.
    N, K, Nz = 5, 3, 2
    nodes = range(0.0, 1.0; length = N + 1)

    @testset "$(nameof(typeof(b)))" for b in (StateForm(), DerivativeForm())
        c1 = CollocationExaCore(nodes, K; basis = b)
        @add_var_collocation(c1, z, 1:Nz)
        itr = mesh_rows(c1, 1:Nz)
        @add_con_collocation(c1, coll, z, -z[v, i, k] for (v, i, k, t) in itr)
        @add_con_continuity(c1, cont, z)

        c2 = CollocationExaCore(nodes, K; basis = b)
        c2, y = add_var_collocation(c2, 1:Nz)
        itr2 = mesh_rows(c2, 1:Nz)                        # rows lead with z's own index
        c2, _ = add_con_collocation(c2, y, (-y[v, i, k] for (v, i, k, t) in itr2))
        c2, _ = add_con_continuity(c2, y)

        m1, m2 = ExaModels.ExaModel(c1), ExaModels.ExaModel(c2)
        x = collect(range(0.1, 2.0; length = m1.meta.nvar))
        @test m1.meta.ncon == m2.meta.ncon
        @test ExaModels.NLPModels.cons(m1, x) ≈ ExaModels.NLPModels.cons(m2, x)
    end
end

@testset "anonymous helpers register nothing" begin
    K, Nz = 3, 2
    core = CollocationExaCore(range(0.0, 1.0; length = 5), K)

    z = @add_var_collocation(core, 1:Nz)
    itr = mesh_rows(core, 1:Nz)
    coll = @add_con_collocation(core, z, -z[v, i, k] for (v, i, k, t) in itr)
    cont = @add_con_continuity(core, z)

    @test z isa CollocationVariable
    @test coll isa ExaModels.Constraint
    @test cont isa ExaModels.Constraint
    @test core.refs == (;)
    @test z in core.blocks                       # tracked as a block all the same
    @test z.krange == 0:K
end

@testset "two leading dimensions" begin
    # The stencils have a branch per leading-dimension count, and one call covers every slot.
    N, K, Nz, Nc = 5, 3, 2, 3
    core = CollocationExaCore(range(0.0, 1.0; length = N + 1), K)
    @add_var_collocation(core, z, 1:Nz, 1:Nc)

    mesh_t = core.mesh.t
    itr = vec([(v, c, i, k, mesh_t[i, k]) for v in 1:Nz, c in 1:Nc, i in 1:N, k in 1:K])
    @test length(first(itr)) == 5                                 # (v, c, i, k, t)
    @add_con_collocation(core, coll, z, -z[v, c, i, k] for (v, c, i, k, t) in itr)
    @add_con_continuity(core, cont, z)

    @test core.ncon == Nz * Nc * N * K + Nz * Nc * (N - 1)        # collocation + junctions
    @test :cont in propertynames(core)
end
