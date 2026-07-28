# Bruno et al. (2016), J Exp Bot 67(21):5993-6005 -- carotenoid cleavage by CCD enzymes,
# as posed in the PEtab Benchmark Collection (Bruno_JExpBot2016).
#
# A parameter-estimation problem rather than an optimal control one: six experimental
# conditions share one set of rate constants, and the objective is the Gaussian negative
# log-likelihood over 77 measurements. ExaModelsPEtab.jl builds this class of model from the
# PEtab tables automatically; here the SBML reactions and the condition table are transcribed
# by hand, so the same problem is assembled out of the helpers directly.
#
# Two things this exercises that Van der Pol does not:
#
#   * `start` on `add_var_collocation`. Unlike a control problem driven to a known target,
#     nothing here pins the state profile -- it is whatever the parameters imply -- so the
#     states have to be initialized from the parameters, by integrating the ODE at the
#     starting values and sampling the result on the mesh. That is what ExaModelsPEtab does
#     with a real integrator; a fixed-step RK4 is enough here. It is worth the trouble: on
#     this model it takes the solve from 33 interior-point iterations to 4.
#   * grouping. Seven species obey seven different equations, but only four of them are
#     structurally distinct expressions, so the model needs four `@add_con_collocation`
#     calls, not seven.

using MadNLP

@testset "Bruno et al. (2016) parameter estimation" begin
    # ---------------------------------------------------------------- model_Bruno.xml ----
    # Species, in the order of listOfSpecies.
    BCAR, BCRY, B10, BIO, OHB10, OHBIO, ZEA = 1:7
    Nz, Nc, K = 7, 6, 4

    # The six reactions are all first order, v_r = a_r * (its reactant):
    #
    #   v1  bcar  -> b10 + bio      a1 = kb1 * kb1_multiplier
    #   v2  b10   -> bio            a2 = kb2 * kb2_multiplier
    #   v3  bcry  -> b10 + ohbio    a3 = kc1 * kc1_multiplier
    #   v4  bcry  -> bio + ohb10    a4 = kc2 * kc2_multiplier
    #   v5  ohb10 -> ohbio          a5 = kc4 * kc4_multiplier
    #   v6  zea   -> ohb10 + ohbio  a6 = k5  * k5_multiplier
    #
    # so, with the compartment size fixed at 1,
    #
    #   d(bcar)/dt  = -v1                d(bio)/dt   = v1 + v2 + v4
    #   d(bcry)/dt  = -v3 - v4           d(ohb10)/dt = v4 - v5 + v6
    #   d(b10)/dt   = v1 - v2 + v3       d(ohbio)/dt = v3 + v5 + v6
    #                                    d(zea)/dt   = -v6
    #
    # Rate index r, in the column order of the condition table.
    K5, KB1, KB2, KC1, KC2, KC4 = 1:6

    # The same seven equations in plain Julia, used to build the state guess below.
    function bruno_rhs(z, a)
        bcar, bcry, b10, bio, ohb10, ohbio, zea = z
        v1, v2, v3 = a[KB1] * bcar, a[KB2] * b10, a[KC1] * bcry
        v4, v5, v6 = a[KC2] * bcry, a[KC4] * ohb10, a[K5] * zea
        return [-v1, -v3 - v4, v1 - v2 + v3, v1 + v2 + v4, v4 - v5 + v6, v3 + v5 + v6, -v6]
    end

    # ------------------------------------------------------------- parameters_Bruno.tsv ----
    # All thirteen are estimated on a log10 scale over [1e-5, 1e3], so the decision variable
    # is theta = log10(parameter) and the model reads the physical value as 10^theta. The
    # nominal column is the collection's published optimum; it is the starting point here.
    INIT_B10, INIT_BCAR1, INIT_BCAR2, INIT_BCRY, INIT_OHB10, INIT_ZEA = 1:6
    KP = [7, 8, 9, 10, 11, 12]   # rate r -> its parameter index
    SZEA = 13

    pnom = [
        4.22228991356159,   # init_b10_1
        4.44029572277168,   # init_bcar1
        3.30171980495163,   # init_bcar2
        5.21693865661801,   # init_bcry_1
        1.73724675926262,   # init_ohb10_1
        2.95812301577671,   # init_zea_1
        0.003086673219173,  # k5
        0.016628409992721,  # kb1
        0.005843831819271,  # kb2
        0.001678825242274,  # kc1
        0.006975989070463,  # kc2
        0.006082700899195,  # kc4
        0.516831766210391,  # szea
    ]
    Np = length(pnom)
    θnom = log10.(pnom)
    θLB, θUB = fill(log10(1e-5), Np), fill(log10(1e3), Np)

    # ------------------------------------------------- experimentalCondition_Bruno.tsv ----
    # Rows are the six conditions, columns the multipliers (k5, kb1, kb2, kc1, kc2, kc4).
    # A rate is switched off, taken as is, or scaled by the estimated factor szea.
    off, on, sc = :off, :on, :szea
    mult = [
        off off on  on  off on    # model1_data1
        on  on  on  off off off   # model1_data2
        sc  sc  sc  sc  sc  sc    # model1_data3
        on  off on  on  on  on    # model1_data4
        off off on  off off on    # model1_data5
        sc  off off off off sc    # model1_data6
    ]

    # Each condition starts one species from one estimated parameter; everything else is 0.
    ic_p = [(B10, 1, INIT_B10), (BCAR, 2, INIT_BCAR1), (BCAR, 3, INIT_BCAR2),
            (BCRY, 4, INIT_BCRY), (OHB10, 5, INIT_OHB10), (ZEA, 6, INIT_ZEA)]
    ic_0 = [(v, c) for v in 1:Nz, c in 1:Nc if !any(q -> q[1] == v && q[2] == c, ic_p)]

    # ------------------------------------------------------- measurementData_Bruno.tsv ----
    # One block per (observable, condition); every observable is a single species. Columns
    # are time, measurement, and noiseParameter -- sigma is given numerically for every row,
    # so it enters the objective as a constant instead of an auxiliary variable.
    meas = [
        (B10, 1, [  5.0  3.973385  0.406451
                   15.0  3.976118  0.406728
                   30.0  3.12959   0.320749
                   60.0  3.131418  0.320935
                   90.0  3.105543  0.31831
                  120.0  2.279702  0.234682
                  180.0  1.321613  0.138555]),
        (BCAR, 2, [ 15.0  3.83496   0.251006
                    30.0  2.68932   0.179664
                    45.0  2.12541   0.145321
                    60.0  1.55414   0.111703
                    90.0  0.90185   0.076761
                   120.0  0.62496   0.064455
                   180.0  0.30276   0.054089]),
        (B10, 2, [ 15.0  0.87947   0.110017
                   30.0  1.61775   0.194075
                   45.0  2.0221    0.240978
                   60.0  2.51711   0.298703
                   90.0  2.24951   0.267467
                  120.0  2.46923   0.29311
                  180.0  2.09599   0.249578]),
        (BCAR, 3, [ 15.0  2.792     0.161073
                    30.0  2.51      0.146065
                    45.0  2.101     0.124582
                    60.0  1.917     0.115071
                    90.0  1.598     0.098916
                   120.0  1.135     0.076702
                   180.0  0.824     0.063285]),
        (B10, 3, [ 15.0  0.573     0.066702
                   30.0  0.849     0.092359
                   45.0  0.968     0.10382
                   60.0  1.212     0.127709
                   90.0  1.411     0.147436
                  120.0  1.612     0.167498
                  180.0  2.168     0.223405]),
        (BCRY, 4, [ 15.0  4.804181  0.43054
                    30.0  3.874252  0.353484
                    45.0  3.687507  0.338201
                    60.0  2.905136  0.275261
                    90.0  2.471236  0.241465
                   120.0  2.243992  0.224241
                   180.0  0.922679  0.137702]),
        (B10, 4, [ 15.0  0.186897  0.043179
                   30.0  0.238582  0.046576
                   45.0  0.268357  0.048771
                   60.0  0.377915  0.057968
                   90.0  0.332696  0.053988
                  120.0  0.410503  0.060963
                  180.0  0.487949  0.068419]),
        (OHB10, 4, [ 15.0  0.750567  0.146042
                     30.0  0.957066  0.185626
                     45.0  1.058638  0.205133
                     60.0  1.40061   0.270906
                     90.0  1.342636  0.259748
                    120.0  1.652208  0.319353
                    180.0  2.005941  0.387509]),
        (OHB10, 5, [  5.0  2.187219  0.365852
                     15.0  1.490176  0.249545
                     30.0  1.243819  0.208483
                     60.0  1.100679  0.184647
                     90.0  1.142316  0.191578
                    120.0  0.794621  0.13378
                    180.0  0.696621  0.117543]),
        (OHB10, 6, [ 15.0  0.098     0.023108
                     30.0  0.1573    0.030931
                     45.0  0.165     0.032031
                     60.0  0.23      0.041749
                     90.0  0.374     0.064588
                    120.0  0.445     0.076127
                    180.0  0.46      0.090733]),
        (ZEA, 6, [ 15.0  2.761     0.172674
                   30.0  2.93      0.183243
                   45.0  2.842     0.17774
                   60.0  2.763     0.172799
                   90.0  2.581     0.161417
                  120.0  2.449     0.153161
                  180.0  2.128     0.133086]),
    ]
    Nm = sum(size(tbl, 1) for (_, _, tbl) in meas)

    # -------------------------------------------------------------------------- mesh ----
    # Every measurement time is an interval boundary, so the state at a measurement is a mesh
    # node rather than an interpolation. The measurement grid alone is far too coarse at the
    # tail -- it has to resolve the dynamics, not just the data -- so intervals are split
    # until none is longer than hmax.
    function refine(nodes, hmax)
        out = [float(first(nodes))]
        for (lo, hi) in zip(nodes[1:(end - 1)], nodes[2:end])
            n = max(1, ceil(Int, (hi - lo) / hmax))
            append!(out, lo .+ (hi - lo) .* (1:n) ./ n)
        end
        return out
    end

    t_meas = sort(unique(vcat(0.0, [tbl[:, 1] for (_, _, tbl) in meas]...)))
    nodes = refine(t_meas, 15.0)
    N = length(nodes) - 1

    # --------------------------------------------------------------- the state guess ----
    # Classic fourth-order Runge-Kutta, sampled at each time in `ts` (nondecreasing; a
    # repeated time, which Radau produces at every interval junction, is a no-op).
    function rk4_samples(f, z0, ts; substeps = 8)
        out = Vector{Vector{Float64}}(undef, length(ts))
        z, t = copy(z0), ts[1]
        out[1] = copy(z)
        for n in 2:length(ts)
            dt = (ts[n] - t) / substeps
            for _ in 1:substeps
                k1 = f(z)
                k2 = f(z .+ (dt / 2) .* k1)
                k3 = f(z .+ (dt / 2) .* k2)
                k4 = f(z .+ dt .* k3)
                z = z .+ (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
            end
            t = ts[n]                        # reset, so substepping cannot drift
            out[n] = copy(z)
        end
        return out
    end

    # Effective rate a[r,c] and initial state at a given parameter vector.
    rates(θ) = [mult[c, r] === off ? 0.0 :
                mult[c, r] === on ? 10.0^θ[KP[r]] :
                10.0^(θ[KP[r]] + θ[SZEA])
                for r in 1:6, c in 1:Nc]
    function state0(θ)
        z0 = zeros(Nz, Nc)
        for (v, c, m) in ic_p
            z0[v, c] = 10.0^θ[m]
        end
        return z0
    end

    # ------------------------------------------------------------------------ build ----
    function build(θ0 = θnom)
        dae = DAEta(nodes, K)
        core = ExaModels.ExaCore(; concrete = Val(true))

        # Sample times of the allocated block, in its own (i, k) order: k = 0 is the
        # interval boundary, k = 1,...,K the collocation points.
        ik = [(i, k) for i in 1:N for k in 0:K]
        ts = [k == 0 ? dae.nodes[i] : dae.mesh.t[i, k] for (i, k) in ik]

        a0, z0 = rates(θ0), state0(θ0)
        zstart = Array{Float64}(undef, Nz, Nc, N, K + 1)
        for c in 1:Nc
            prof = rk4_samples(z -> bruno_rhs(z, view(a0, :, c)), z0[:, c], ts)
            for (n, (i, k)) in enumerate(ik)
                zstart[:, c, i, k + 1] .= prof[n]
            end
        end

        # `start` is shaped to the allocated block, (dims..., 1:N, krange) -- not to the
        # per-timepoint shape that was declared.
        @add_var_collocation(core, dae, z, 1:Nz, 1:Nc; start = zstart)
        ExaModels.@add_var(core, p, 1:Np; lvar = θLB, uvar = θUB, start = θ0)

        # The condition table multiplies each rate constant by 0, 1, or szea. Folding that
        # product into one variable per (rate, condition) is what lets every species
        # equation below be one expression across all six conditions; PEtab keeps the two
        # factors apart, as cv[r,c] and 10^theta, and pays for it in every kernel.
        ExaModels.@add_var(core, a, 1:6, 1:Nc; start = a0)
        ExaModels.@add_con(core, rate_off, a[r, c]
            for (r, c) in [(r, c) for r in 1:6, c in 1:Nc if mult[c, r] === off])
        ExaModels.@add_con(core, rate_on, a[r, c] - exp(log(10.0) * p[m])
            for (r, c, m) in [(r, c, KP[r]) for r in 1:6, c in 1:Nc if mult[c, r] === on])
        ExaModels.@add_con(core, rate_scaled, a[r, c] - exp(log(10.0) * (p[m] + p[SZEA]))
            for (r, c, m) in [(r, c, KP[r]) for r in 1:6, c in 1:Nc if mult[c, r] === sc])

        # Seven species equations, four distinct expressions. Which species an equation is
        # for, and which rates and reactants it draws on, all merely vary -- so they ride in
        # the iterator, and the leading index of the state slice varies with them.
        mesh_t = dae.mesh.t
        sweep(rows) = vec([(row..., c, i, k, mesh_t[i, k])
                           for row in rows, c in 1:Nc, i in 1:N, k in 1:K])

        # bcar and zea are consumed and never produced
        @add_con_collocation(core, dae, coll_decay1, z[v, c],
            -a[r, c] * z[v, c, i, k]
            for (v, r, c, i, k, t) in sweep([(BCAR, KB1), (ZEA, K5)]))

        # bcry is consumed by two reactions at once
        @add_con_collocation(core, dae, coll_decay2, z[v, c],
            -(a[r1, c] + a[r2, c]) * z[v, c, i, k]
            for (v, r1, r2, c, i, k, t) in sweep([(BCRY, KC1, KC2)]))

        # b10 and ohb10: two sources, and consumed by a reaction of their own
        @add_con_collocation(core, dae, coll_flow, z[v, c],
            a[r1, c] * z[v1, c, i, k] + a[r2, c] * z[v2, c, i, k] - a[r3, c] * z[v, c, i, k]
            for (v, r1, v1, r2, v2, r3, c, i, k, t) in sweep([
                (B10, KB1, BCAR, KC1, BCRY, KB2), (OHB10, KC2, BCRY, K5, ZEA, KC4)]))

        # bio and ohbio are terminal products: three sources, no sink
        @add_con_collocation(core, dae, coll_sink, z[v, c],
            a[r1, c] * z[v1, c, i, k] + a[r2, c] * z[v2, c, i, k] + a[r3, c] * z[v3, c, i, k]
            for (v, r1, v1, r2, v2, r3, v3, c, i, k, t) in sweep([
                (BIO, KB1, BCAR, KB2, B10, KC2, BCRY),
                (OHBIO, KC1, BCRY, KC4, OHB10, K5, ZEA)]))

        @add_con_continuity(core, dae, cont,
            z[v, c] for (v, c) in continuity_itr(dae, 1:Nz, 1:Nc))

        ExaModels.@add_con(core, ic_zero, z[v, c, 1, 0] for (v, c) in ic_0)
        ExaModels.@add_con(core, ic_par, z[v, c, 1, 0] - exp(log(10.0) * p[m])
            for (v, c, m) in ic_p)

        # Gaussian negative log-likelihood. sigma is known per measurement, so each term is
        # 0.5((y - ymeas)/sigma)^2 + log(sigma) + 0.5log(2pi) with the tail a constant;
        # every observable is a single species, so y is that species' value at the time.
        #
        # Radau puts its last collocation point on tau = 1, so z[...,i,K] is the right end
        # of interval i and a measurement at t = nodes[m] is read off interval m-1. Nothing
        # to interpolate, and the last measurement is not a special case even though it
        # falls on the far end of the horizon. Legendre would need the tau = 1 weights
        # `dae.weights.b` here instead, since none of its points lands on the boundary.
        itr_obj = Tuple{Int, Int, Int, Float64, Float64, Float64}[]
        for (v, c, tbl) in meas, row in axes(tbl, 1)
            tm, ym, sd = tbl[row, 1], tbl[row, 2], tbl[row, 3]
            m = findfirst(n -> isapprox(n, tm; atol = 1e-9), dae.nodes)
            m > 1 || error("measurement at t = $tm is not on an interval's right end")
            push!(itr_obj, (v, c, m - 1, ym, sd, log(sd) + 0.5 * log(2π)))
        end
        ExaModels.@add_obj(core, 0.5 * ((z[v, c, i, K] - ym) / sd)^2 + cst
            for (v, c, i, ym, sd, cst) in itr_obj)

        return core, dae, z, p, a
    end

    # The negative log-likelihood PEtab.jl and ExaModelsPEtab.jl both report for this model
    # at the collection's nominal parameters, which are its optimum.
    NLL_REF = -46.68818145

    @testset "from the nominal parameters" begin
        core, dae, z, p, a = build()

        @test dae.N == N == 13
        @test size(dae.mesh.t) == (N, K)
        @test core.nvar == Nz * Nc * N * (K + 1) + Np + 6 * Nc
        @test Nm == 77

        result = madnlp(ExaModels.ExaModel(core); print_level = MadNLP.ERROR)
        @test result.status == MadNLP.SOLVE_SUCCEEDED

        # The discretization is what is being checked here: K = 4 Radau on this mesh has to
        # reproduce the likelihood an adaptive ODE solver gets on the same parameters.
        @test result.objective ≈ NLL_REF atol = 1e-6
        @test 10 .^ ExaModels.solution(result, p) ≈ pnom rtol = 1e-4

        zsol = ExaModels.solution(result, z)      # 1-based, so k = 0,...,K lands on 1,...,K+1
        θ = ExaModels.solution(result, p)

        # Initial conditions honored, on both branches
        @test zsol[BCAR, 2, 1, 1] ≈ 10.0^θ[INIT_BCAR1] rtol = 1e-8
        @test zsol[ZEA, 6, 1, 1] ≈ 10.0^θ[INIT_ZEA] rtol = 1e-8
        @test zsol[BIO, 1, 1, 1] ≈ 0.0 atol = 1e-8

        # Every species is a concentration, and the three that are only ever consumed decay
        # monotonically -- neither is imposed anywhere, so both are the discretization's own
        @test all(zsol .>= -1e-6)
        @test all(diff(vec(zsol[BCAR, 2, :, K + 1])) .<= 1e-8)
        @test all(diff(vec(zsol[ZEA, 6, :, K + 1])) .<= 1e-8)

        # Rates switched off in the condition table stay off, and condition 3 scales by szea
        asol = ExaModels.solution(result, a)
        @test asol[KB1, 1] ≈ 0.0 atol = 1e-8
        @test asol[KB1, 3] ≈ 10.0^θ[KP[KB1]] * 10.0^θ[SZEA] rtol = 1e-6

        # Radau's tau = 1 point is the interval boundary, so the continuity weights select it
        # and z[...,N,K] is the state at t = 180 with nothing to evaluate
        @test dae.weights.b ≈ [zeros(K); 1.0]
        @test zsol[BCAR, 2, N, K + 1] < zsol[BCAR, 2, 1, 1]
    end

    @testset "from a perturbed start" begin
        # Same build, but the whole starting point -- the parameters and the state profile
        # integrated from them -- is moved off the optimum, so the guess is doing real work
        # rather than handing the solver its own answer back. Same optimum, 8 iterations
        # instead of 72 from a flat state start.
        core, _, _, p, _ = build(θnom .+ 0.3)

        result = madnlp(ExaModels.ExaModel(core); print_level = MadNLP.ERROR)
        @test result.status == MadNLP.SOLVE_SUCCEEDED
        @test result.objective ≈ NLL_REF atol = 1e-6
        @test 10 .^ ExaModels.solution(result, p) ≈ pnom rtol = 1e-4
    end
end
