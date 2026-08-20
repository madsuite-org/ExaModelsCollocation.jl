# Bruno et al. (2016), J Exp Bot 67(21):5993-6005
# Bruno_JExptBot2016 from Benchmarking Initiative's PEtab Collection

using ExaModels
using ExaModelsCollocation
using MadNLP

# ----- Problem data from PEtab file -----

const BCAR, BCRY, B10, BIO, OHB10, OHBIO, ZEA = 1:7
const Nz, Nc, Ncv = 7, 6, 6

const INIT_B10, INIT_BCAR1, INIT_BCAR2, INIT_BCRY, INIT_OHB10, INIT_ZEA = 1:6
const KP = [7, 8, 9, 10, 11, 12]
const SZEA = 13

const pnom = [
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

const Np = length(pnom)
const θnom = log10.(pnom)
const θLB, θUB = fill(log10(1e-5), Np), fill(log10(1e3), Np)

const off, on, sc = :off, :on, :szea
const mult = [
    off off on  on  off on    # model1_data1
    on  on  on  off off off   # model1_data2
    sc  sc  sc  sc  sc  sc    # model1_data3
    on  off on  on  on  on    # model1_data4
    off off on  off off on    # model1_data5
    sc  off off off off sc    # model1_data6
]

const ic_p = [(B10, 1, INIT_B10), (BCAR, 2, INIT_BCAR1), (BCAR, 3, INIT_BCAR2),
              (BCRY, 4, INIT_BCRY), (OHB10, 5, INIT_OHB10), (ZEA, 6, INIT_ZEA)]
const ic_0 = [(v, c) for v in 1:Nz, c in 1:Nc if !any(q -> q[1] == v && q[2] == c, ic_p)]

const meas = [
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
const Nm = sum(size(tbl, 1) for (_, _, tbl) in meas)

# Known optimal solution
const NLL_REF = -46.68818145

# ----- right-hand side function -----

const K5, KB1, KB2, KC1, KC2, KC4 = 1:6

# right-hand side functions
function bruno_rhs(z, a)
    bcar, bcry, b10, bio, ohb10, ohbio, zea = z
    v1, v2, v3 = a[KB1] * bcar, a[KB2] * b10, a[KC1] * bcry
    v4, v5, v6 = a[KC2] * bcry, a[KC4] * ohb10, a[K5] * zea
    return [
        -v1, 
        -v3 - v4, 
        v1 - v2 + v3, 
        v1 + v2 + v4, 
        v4 - v5 + v6, 
        v3 + v5 + v6, 
        -v6
    ]
end

# ----- Mesh nodes -----

const TEND, NMESH, KDEG = 180.0, 36, 4

# ----- Obtain good initial guess for discretized states -----

function RK4(f, z0, ts; substeps = 8)
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
        t = ts[n]
        out[n] = copy(z)
    end
    return out
end

rates(θ) = [mult[c, cvidx] === off ? 0.0 :
            mult[c, cvidx] === on ? 10.0^θ[KP[cvidx]] :
            10.0^(θ[KP[cvidx]] + θ[SZEA])
            for cvidx in 1:Ncv, c in 1:Nc]

function state0(θ)
    z0 = zeros(Nz, Nc)
    for (v, c, m) in ic_p
        z0[v, c] = 10.0^θ[m]
    end
    return z0
end

# ----- Create ExaModel -----
function examodel_bruno(θ0 = θnom; K = KDEG, N = NMESH)
    nodes = range(0.0, TEND; length = N + 1)

    # Create CollocationExaCore
    core = CollocationExaCore(nodes, K)

    # Solve ODE system at nominal θ to obtain good initial guess
    h, taus = diff(core.nodes), core.weights.taus
    ik = [(i, k) for i in 1:N for k in 0:K]
    ts = [core.nodes[i] + (k == 0 ? 0.0 : h[i] * taus[k]) for (i, k) in ik]
    cv0, z0 = rates(θ0), state0(θ0)
    zstart = Array{Float64}(undef, Nz, Nc, N, K + 1)
    for c in 1:Nc
        prof = RK4(z -> bruno_rhs(z, view(cv0, :, c)), z0[:, c], ts)
        for (n, (i, k)) in enumerate(ik)
            zstart[:, c, i, k + 1] .= prof[n]
        end
    end

    # Create CollocationVariables
    @add_var_collocation(core, z, 1:Nz, 1:Nc; start = zstart) # z[v,c,i,k]

    # Create variables (unknown parameters to estimate)
    ExaModels.@add_var(core, p, 1:Np; lvar = θLB, uvar = θUB, start = θ0)

    # Create auxiliary variables and constraints for condition-dependent variables
    @add_var(core, cv, 1:Ncv, 1:Nc; start = cv0)
    @add_con(core, rate_off,
        cv[cvidx,c]
        for (cvidx,c) in [(cvidx,c) for cvidx in 1:Ncv, c in 1:Nc if mult[c,cvidx] === off]
    )
    @add_con(core, rate_on,
        cv[cvidx,c] - exp(log(10.0) * p[m])
        for (cvidx,c,m) in [(cvidx,c,KP[cvidx]) for cvidx in 1:Ncv, c in 1:Nc if mult[c,cvidx] === on]
    )
    @add_con(core, rate_scaled,
        cv[cvidx,c] - exp(log(10.0) * (p[m] + p[SZEA]))
        for (cvidx,c,m) in [(cvidx,c,KP[cvidx]) for cvidx in 1:Ncv, c in 1:Nc if mult[c,cvidx] === sc]
    )

    # Create collocation constraints
    @add_con_collocation(core, coll_decay1, z[v,c],
        -cv[cvidx,c] * z[v,c]
        for (v,cvidx) in [(BCAR, KB1), (ZEA, K5)], c in 1:Nc
    )

    # bcry: two sinks
    @add_con_collocation(core, coll_decay2, z[v,c],
        -(cv[cvidx1,c] + cv[cvidx2,c]) * z[v,c]
        for (v,cvidx1,cvidx2) in [(BCRY, KC1, KC2)], c in 1:Nc
    )

    # b10, ohb10: two sources and a sink
    @add_con_collocation(core, coll_flow, z[v,c],
        cv[cvidx1,c] * z[v1,c] + cv[cvidx2,c] * z[v2,c] - cv[cvidx3,c] * z[v,c]
        for (v,cvidx1,v1,cvidx2,v2,cvidx3) in [
            (B10,   KB1, BCAR, KC1, BCRY, KB2),
            (OHB10, K5,  ZEA,  KC2, BCRY, KC4)
        ], c in 1:Nc
    )

    # bio, ohbio: three sources, no sink
    @add_con_collocation(core, coll_sink, z[v,c],
        cv[cvidx1,c] * z[v1,c] + cv[cvidx2,c] * z[v2,c] + cv[cvidx3,c] * z[v3,c]
        for (v,cvidx1,v1,cvidx2,v2,cvidx3,v3) in [
            (BIO,   KB1, BCAR, KB2, B10,   KC2, BCRY),
            (OHBIO, K5,  ZEA,  KC4, OHB10, KC1, BCRY)
        ], c in 1:Nc
    )

    # Create continuity constraints
    @add_con_continuity(core, cont, z)

    # Create initial condition constraints
    @add_con(core, ic_zero,
        z[v,c,1,0]
        for (v,c) in ic_0
    )
    @add_con(core, ic_par,
        z[v,c,1,0] - exp(log(10.0) * p[m])
        for (v,c,m) in ic_p
    )

    # Create objective function
    itr_obj = Tuple{Int, Int, Int, Float64, Float64, Float64}[]
    for (v, c, tbl) in meas, row in axes(tbl, 1)
        tm, ym, sd = tbl[row, 1], tbl[row, 2], tbl[row, 3]
        i, r = divrem(Int(tm) * N, Int(TEND))
        r == 0 || error("measurement at t = $tm is not on an interval's right end")
        push!(itr_obj, (v, c, i, ym, sd, log(sd) + 0.5 * log(2π)))
    end
    @add_obj(core,
        0.5 * ((z[v,c,i,K] - ym) / sd)^2 + cst
        for (v,c,i,ym,sd,cst) in itr_obj
    )

    return ExaModel(core)
end

# ----- Solve -----

# Create CollocationExaModel
model = examodel_bruno()

# Solve
result = madnlp(model)

θsol = solution(result, model.p)
println("status    = $(result.status)")
println("nll       = $(result.objective), reference = $NLL_REF")
println("|nll-ref| = $(abs(result.objective - NLL_REF))")
println("max |θ - θnom| = $(maximum(abs, θsol - θnom))")
