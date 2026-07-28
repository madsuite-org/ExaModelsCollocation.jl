# The two residual forms, checked against each other and against theory.
#
# dz/dt = -z, z(0) = 1  =>  z(t) = exp(-t), solved through the helpers. StateForm (10.7) and
# DerivativeForm (10.8) are algebraically equivalent discretizations over the same variables,
# so on the same roots they must land on the same answer; and the terminal error must decay
# at the order the collocation family gives. Together those pin the weight indexing and the
# row mapping in a way a shape check cannot.

using MadNLP

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
function terminal(dae, zsol, N, K, Nz)
    last(dae.weights.taus) ≈ 1 && return zsol[:, N, K + 1]
    if dae.basis isa StateForm
        return [sum(dae.weights.b[j + 1] * zsol[v, N, j + 1] for j in 0:K) for v in 1:Nz]
    end
    return [
        zsol[v, N, 1] + dae.mesh.h[N] * sum(dae.weights.b[j] * -zsol[v, N, j + 1] for j in 1:K)
        for v in 1:Nz
    ]
end

# Build and solve the decay problem. Two components share one right-hand side, so v goes in
# the iterator and one call covers both.
function solve_decay(basis, roots, N, K; Nz = 2)
    dae = DAEta(range(0.0, TF; length = N + 1), K; basis = basis, roots = roots)
    core = ExaModels.ExaCore(; concrete = Val(true))
    @add_var_collocation(core, dae, z, 1:Nz)

    itr = collocation_itr(dae, 1:Nz)                  # autonomous, so t rides along unused
    @add_con_collocation(core, dae, coll, z[v], -z[v, i, k] for (v, i, k, t) in itr)

    # StateForm continuity is fixed by the mode; DerivativeForm integrates the right-hand
    # side across the element, so it takes the same slice and generator.
    if basis isa StateForm
        @add_con_continuity(core, dae, cont, z[v] for (v, i) in continuity_itr(dae, 1:Nz))
    else
        @add_con_continuity(core, dae, cont, z[v], -z[v, i, k] for (v, i, k, t) in itr)
    end

    ExaModels.@add_con(core, ic, z[v, 1, 0] - val for (v, val) in [(1, 1.0), (2, 2.0)])

    result = madnlp(ExaModels.ExaModel(core); print_level = MadNLP.ERROR, tol = 1e-12)
    # solution() returns a plain 1-based array, so k = 0,...,K lands on 1,...,K+1
    zsol = ExaModels.solution(result, z)
    return result, zsol, terminal(dae, zsol, N, K, Nz)
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

@testset "each basis must be called the way its mode says" begin
    dae = DAEta(range(0.0, 1.0; length = 5), 3)                # StateForm
    core = ExaModels.ExaCore(; concrete = Val(true))
    @add_var_collocation(core, dae, z, 1:1)
    itr = collocation_itr(dae, 1:1)

    # StateForm continuity is fixed by the mode and takes no right-hand side
    @test_throws ArgumentError @add_con_continuity(core, dae, bad, z[v],
        -z[v, i, k] for (v, i, k, t) in itr)

    ddae = DAEta(range(0.0, 1.0; length = 5), 3; basis = DerivativeForm())
    dcore = ExaModels.ExaCore(; concrete = Val(true))
    @add_var_collocation(dcore, ddae, y, 1:1)

    # DerivativeForm continuity integrates f, so it needs one
    @test_throws ArgumentError @add_con_continuity(dcore, ddae, bad2,
        y[v] for (v, i) in continuity_itr(ddae, 1:1))
end
