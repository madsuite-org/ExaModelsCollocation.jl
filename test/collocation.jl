# The mode math in src/collocation/: taus -> weights -> mesh.
#
# Every roots family and both bases are built here, and the weights are checked against the
# identities that define them rather than against stored numbers. A Lagrange weight vector
# applied to the nodal values of a polynomial p of degree <= length(nodes)-1 must reproduce
# exactly the operation it encodes, so monomials up to that degree pin every entry.

const C = ExaModelsCollocation

const ROOTS = (GaussRadau(), GaussLegendre(), GaussLobatto())
const BASES = (StateForm(), DerivativeForm())
const DEGREES = 1:5

# Stands in for a polynomial family that was never implemented
struct NotAPolynomial <: C.AbstractPolynomial end

@testset "collocation points" begin
    @testset "$(nameof(typeof(r))), K = $K" for r in ROOTS, K in DEGREES
        r isa GaussLobatto && K < 2 && continue      # Lobatto needs both endpoints

        taus = C._get_taus(r, K)
        @test length(taus) == K
        @test eltype(taus) <: AbstractFloat
        @test issorted(taus)
        @test allunique(taus)
        @test all(0 .<= taus .<= 1)                   # mapped from [-1,1] onto the interval

        # Which endpoints each family puts a point on
        @test (last(taus) ≈ 1) == (r isa GaussRadau || r isa GaussLobatto)
        @test (first(taus) ≈ 0) == (r isa GaussLobatto)
    end

    @testset "K = 1 special cases" begin
        @test C._get_taus(GaussRadau(), 1) ≈ [1.0]        # backward Euler
        @test C._get_taus(GaussLegendre(), 1) ≈ [0.5]     # implicit midpoint
        @test_throws ErrorException C._get_taus(GaussLobatto(), 1)
    end
end

@testset "Lagrange basis operators" begin
    # Nodal values of p over `nodes`, and the exact answers the weights must reproduce
    pval(p, nodes) = [p(x) for x in nodes]

    @testset "$(nameof(typeof(r))), K = $K" for r in ROOTS, K in DEGREES
        r isa GaussLobatto && K < 2 && continue

        taus = C._get_taus(r, K)
        anchored = C._anchor(taus)                    # [0, tau_1..tau_K]
        @test length(anchored) == K + 1
        @test anchored[1] == 0

        for nodes in (anchored, taus)
            n = length(nodes)
            allunique(nodes) || continue              # Lobatto: anchored repeats tau = 0

            D = C.delljk(nodes, nodes)                # d/dtau at every node
            b = C.ell1j(nodes)                        # evaluation at tau = 1
            O = C.Omegajk(nodes, nodes)               # integral 0 -> each node
            o = C.Omega1j(nodes)                      # integral 0 -> 1

            @test size(D) == (n, n)
            @test size(O) == (n, n)
            @test length(b) == n
            @test length(o) == n

            # Exact on every monomial the basis can represent, degree 0 through n-1
            for m in 0:(n - 1)
                p = x -> x^m
                dp = x -> m == 0 ? zero(x) : m * x^(m - 1)
                ip = x -> x^(m + 1) / (m + 1)
                v = pval(p, nodes)

                @test permutedims(D) * v ≈ pval(dp, nodes) atol = 1e-9
                @test b' * v ≈ p(1.0) atol = 1e-10
                @test permutedims(O) * v ≈ pval(ip, nodes) atol = 1e-10
                @test o' * v ≈ ip(1.0) atol = 1e-10
            end

            # Consequences of the above worth stating on their own:
            @test sum(D; dims = 1) ≈ zeros(1, n) atol = 1e-9    # d/dtau of a constant
            @test sum(b) ≈ 1                                    # partition of unity
            @test vec(sum(O; dims = 1)) ≈ nodes atol = 1e-10     # integral of 1 is tau
            @test sum(o) ≈ 1 atol = 1e-10

            # The basis is cardinal: ell_j(node_i) = delta_ij, so evaluating at tau = 1
            # picks out a single node whenever the family has one there
            last(nodes) ≈ 1 && @test b ≈ [i == n ? 1.0 : 0.0 for i in 1:n] atol = 1e-10
        end
    end

    @testset "delljk only differentiates at its own nodes" begin
        taus = C._get_taus(GaussRadau(), 3)
        @test_throws ArgumentError C.delljk(taus, [0.5])
    end
end

@testset "BasisWeights build paths" begin
    @testset "$(nameof(typeof(b))), $(nameof(typeof(r))), K = $K" for
            b in BASES, r in ROOTS, K in DEGREES

        r isa GaussLobatto && K < 2 && continue
        # Lobatto puts a point on tau = 0, which collides with StateForm's anchor
        r isa GaussLobatto && b isa StateForm && continue

        taus = C._get_taus(r, K)
        w = C._get_weights(Lagrange(), b, taus)

        @test w isa C.BasisWeights
        @test w.taus === taus
        # concretely typed fields, so the weights are not an inference barrier
        @test all(isconcretetype, fieldtypes(typeof(w)))

        # The shape each basis commits to. add_con_collocation reads A[j+1,k] for
        # j = 0,...,K, so StateForm must be (K+1) x K -- this is what pins that down.
        n = b isa StateForm ? K + 1 : K
        @test size(w.A) == (n, K)
        @test length(w.b) == n
        @test all(isfinite, w.A)
        @test all(isfinite, w.b)
    end

    @testset "mode resolution" begin
        nodes = range(0.0, 1.0; length = 5)

        # GaussLobatto has a point on tau = 0, so it cannot carry StateForm's anchor
        dae = @test_logs (:warn, r"GaussLobatto") DAEta(nodes, 3; roots = GaussLobatto())
        @test dae.basis isa DerivativeForm

        @test_throws ArgumentError DAEta(nodes, 3; polynomial = NotAPolynomial())
    end
end

@testset "mesh placement" begin
    @testset "uniform" begin
        taus = C._get_taus(GaussRadau(), 3)
        mesh = C._get_mesh(collect(range(0.0, 5.0; length = 21)), taus)

        @test length(mesh.nodes) == 21
        @test length(mesh.h) == 20                    # N is intervals, not boundaries
        @test all(mesh.h .≈ 0.25)
        @test size(mesh.t) == (20, 3)
        @test mesh.t[1, :] ≈ mesh.nodes[1] .+ mesh.h[1] .* taus
        @test mesh.t[end, end] ≈ 5.0                  # Radau's last point is the right edge
    end

    @testset "nonuniform" begin
        nodes = [0.0, 0.1, 1.0, 1.05, 3.0]
        taus = C._get_taus(GaussLegendre(), 2)
        mesh = C._get_mesh(nodes, taus)

        @test mesh.h ≈ diff(nodes)
        @test size(mesh.t) == (4, 2)
        for i in 1:4, j in 1:2
            @test mesh.t[i, j] ≈ nodes[i] + mesh.h[i] * taus[j]
            @test nodes[i] <= mesh.t[i, j] <= nodes[i + 1]   # stays inside its interval
        end
    end
end
