# Fujita et al. (2010), Sci Signal 3(132):ra56, as posed in the PEtab Benchmark Collection
# (Fujita_SciSignal2010, test_data/input_pulse). Parameter estimation: nine species, six EGF
# doses, one set of rate constants, and a Gaussian negative log-likelihood over 144 measurements.
#
# EGF is a rectangular pulse. The dose is held for TPULSE and then washed out, and the edge is
# itself a measurement time, so it lands on a node and no interval straddles it. EGF is data
# rather than a variable, so it rides in the iterator as the coefficient of the association term.

ENV["GKSwstype"] = "100"

using ExaModels
using ExaModelsCollocation
using MadNLP
using Plots

# ----- Problem data from PEtab file -----

# Species, in the order of listOfSpecies
const EGFR, PEGFR, PEGFR_AKT, AKT, PAKT, S6, PAKT_S6, PS6, EGF_EGFR = 1:9
const Nz, Nc = 9, 6

# Collocation degree and subdivisions per measurement interval
const KDEG, NSUB = 4, 4

# Eleven reactions, with EGF a known input rather than a species:
#
#   v1  EGFR + EGF <-> EGF_EGFR      v7   pAkt      -> Akt
#   v2  pEGFR + Akt <-> pEGFR_Akt    v8   pS6       -> S6
#   v3  pEGFR_Akt -> pEGFR + pAkt    v9   EGF_EGFR  -> pEGFR
#   v4  pEGFR     -> 0               v10  EGFR      -> 0
#   v5  pAkt + S6 <-> pAkt_S6        v11  0         -> EGFR
#   v6  pAkt_S6   -> pAkt + pS6
#
# v10 and v11 are one turnover pair, EGFR_turnover * (EGFR_SS - EGFR).
const EGFR_SS = 68190.0

# ---------------------------------------------------- parameters_Fujita.tsv ----
# All nineteen estimated on a log10 scale over [1e-8, 1e8]: p = log10(parameter), and the
# model reads the physical value as 10^p. The nominal column is the published optimum.
const TURNOVER, INIT_AKT, INIT_EGFR, INIT_S6 = 1:4
const K1F, K1R, K2F, K2R, K3, K4, K5F, K5R, K6, K7, K8, K9 = 5:16
const SC_PAKT, SC_PEGFR, SC_PS6 = 17:19

const pnom = [
    0.001479614562719,     # EGFR_turnover
    0.026930923029049,     # init_AKT
    37536153.8059105,      # init_EGFR
    188.063148677911,      # init_S6
    0.003690766129111,     # reaction_1_k1
    0.002301175486005,     # reaction_1_k2
    0.000936500808211,     # reaction_2_k1
    60965.2066642586,      # reaction_2_k2
    0.433225051651771,     # reaction_3_k1
    0.030155177423024,     # reaction_4_k1
    3.27310803801897e-06,  # reaction_5_k1
    0.000398546299782,     # reaction_5_k2
    5.46319692934546e-06,  # reaction_6_k1
    0.011803208311735,     # reaction_7_k1
    0.000944761775113,     # reaction_8_k1
    0.028510798479438,     # reaction_9_k1
    41.3771031603841,      # scaling_pAkt_tot
    5.64785460492811e-08,  # scaling_pEGFR_tot
    78521.9513232784,      # scaling_pS6_tot
]
const Np = length(pnom)
const θnom = log10.(pnom)
const θLB, θUB = fill(log10(1e-8), Np), fill(log10(1e8), Np)

# One call per structurally distinct right-hand side. The Akt module and the S6 module are
# isomorphic, so three of the six cover both members of a pair and the rows are index sets.

# A free kinase: (free, partner, complex, recycled from, kf, kr, kd)
const FREE = [
    (AKT, PEGFR, PEGFR_AKT, PAKT, K2F, K2R, K7),
    (S6,  PAKT,  PAKT_S6,   PS6,  K5F, K5R, K8),
]

# Its complex: (complex, reactant, reactant, kf, kr, kc)
const CPLX = [
    (PEGFR_AKT, AKT, PEGFR, K2F, K2R, K3),
    (PAKT_S6,   S6,  PAKT,  K5F, K5R, K6),
]

# Its activated form: (activated, upstream, ka, partner, complex, kf, kr, kc, kd)
const ACTV = [
    (PEGFR, EGF_EGFR,  K9, AKT, PEGFR_AKT, K2F, K2R, K3, K4),
    (PAKT,  PEGFR_AKT, K3, S6,  PAKT_S6,   K5F, K5R, K6, K7),
]

# Species present at t = 0, each from its own parameter; everything else is 0
const ic_p = [(EGFR, INIT_EGFR), (AKT, INIT_AKT), (S6, INIT_S6)]
const ic_0 = [v for v in 1:Nz if !any(q -> q[1] == v, ic_p)]

# ------------------------------------------ experimentalCondition_Fujita.tsv ----
# The six conditions are EGF doses [ng/mL], each applied as a pulse over [0, TPULSE]
const TEND, TPULSE = 3600.0, 60.0
const DOSE = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

# ------------------------------------- measurementData_pulse_Fujita.tsv ----
# One entry per (observable, condition). The two-species observables are scaled sums, so each
# carries the pair it reads. Columns are time, measurement, and sigma.
const meas = [
    (SC_PEGFR, (PEGFR, PEGFR_AKT), 1, [   0.0   0.0000000e+00   0.01
                                         60.0   2.6607499e-02   0.01
                                        120.0   1.8747276e-02   0.01
                                        300.0   4.4578443e-03   0.01
                                        600.0   2.8718951e-03   0.01
                                        900.0   2.8718951e-03   0.01
                                       1800.0   2.8718951e-03   0.01
                                       3600.0   6.8758800e-04   0.01]),
    (SC_PEGFR, (PEGFR, PEGFR_AKT), 2, [   0.0   0.0000000e+00   0.01
                                         60.0   7.2142779e-02   0.02
                                        120.0   2.8296383e-02   0.01
                                        300.0   1.2036420e-02   0.01
                                        600.0   6.3916011e-03   0.01
                                        900.0   6.5770372e-03   0.01
                                       1800.0   2.4638037e-03   0.01
                                       3600.0   1.1737089e-03   0.01]),
    (SC_PEGFR, (PEGFR, PEGFR_AKT), 3, [   0.0   0.0000000e+00   0.01
                                         60.0   2.5434890e-01   0.05
                                        120.0   8.6042962e-02   0.01
                                        300.0   1.0307423e-02   0.01
                                        600.0   2.8718951e-03   0.01
                                        900.0   2.8718951e-03   0.01
                                       1800.0   2.8718951e-03   0.01
                                       3600.0   2.3383729e-03   0.01]),
    (SC_PEGFR, (PEGFR, PEGFR_AKT), 4, [   0.0   0.0000000e+00   0.01
                                         60.0   5.8356409e-01    0.1
                                        120.0   2.1277657e-01   0.04
                                        300.0   2.4674493e-02   0.01
                                        600.0   1.3849656e-02   0.01
                                        900.0   7.8426463e-03   0.01
                                       1800.0   7.6294933e-03   0.01
                                       3600.0   1.6372512e-03   0.01]),
    (SC_PEGFR, (PEGFR, PEGFR_AKT), 5, [   0.0   0.0000000e+00   0.01
                                         60.0   1.0629958e+00  0.125
                                        120.0   3.9510878e-01   0.06
                                        300.0   4.9381839e-02   0.05
                                        600.0   4.2019287e-02   0.03
                                        900.0   2.7795719e-02   0.01
                                       1800.0   1.0327331e-02   0.01
                                       3600.0   2.9840891e-03   0.01]),
    (SC_PEGFR, (PEGFR, PEGFR_AKT), 6, [   0.0   0.0000000e+00   0.01
                                         60.0   1.0188246e+00   0.07
                                        120.0   4.5013403e-01   0.04
                                        300.0   1.0357264e-01   0.03
                                        600.0   5.6001180e-02   0.01
                                        900.0   2.0638918e-02   0.01
                                       1800.0   2.3500184e-02   0.01
                                       3600.0   4.7958476e-03   0.01]),
    (SC_PAKT,  (PAKT, PAKT_S6), 1, [   0.0   0.0000000e+00   0.01
                                      60.0   1.8292054e-02   0.02
                                     120.0   3.8759236e-02   0.01
                                     300.0   3.3397574e-02   0.01
                                     600.0   3.4415432e-03   0.01
                                     900.0   3.4415432e-03   0.01
                                    1800.0   3.4415432e-03   0.01
                                    3600.0   5.2939586e-03   0.01]),
    (SC_PAKT,  (PAKT, PAKT_S6), 2, [   0.0   0.0000000e+00   0.01
                                      60.0   4.8407699e-02   0.01
                                     120.0   2.1885126e-01   0.04
                                     300.0   1.1153496e-01   0.02
                                     600.0   1.1040503e-02   0.01
                                     900.0   1.0648881e-02   0.01
                                    1800.0   5.5566403e-03   0.01
                                    3600.0   4.8664931e-03   0.01]),
    (SC_PAKT,  (PAKT, PAKT_S6), 3, [   0.0   0.0000000e+00   0.01
                                      60.0   3.2908701e-01   0.03
                                     120.0   4.5038187e-01    0.1
                                     300.0   1.8438721e-01   0.03
                                     600.0   1.8350976e-02   0.03
                                     900.0   6.8830863e-03   0.02
                                    1800.0   6.8830863e-03   0.01
                                    3600.0   9.2644587e-04   0.01]),
    (SC_PAKT,  (PAKT, PAKT_S6), 4, [   0.0   0.0000000e+00   0.01
                                      60.0   8.1405426e-01    0.1
                                     120.0   8.4795502e-01   0.05
                                     300.0   3.1184711e-01   0.05
                                     600.0   3.6473296e-02   0.03
                                     900.0   1.6065632e-02   0.03
                                    1800.0   1.8031099e-02   0.01
                                    3600.0   1.3950348e-02   0.01]),
    (SC_PAKT,  (PAKT, PAKT_S6), 5, [   0.0   0.0000000e+00   0.01
                                      60.0   8.9307345e-01    0.1
                                     120.0   9.9645811e-01    0.1
                                     300.0   2.9820608e-01   0.07
                                     600.0   3.9357645e-02   0.02
                                     900.0   2.4093155e-02   0.01
                                    1800.0   1.8031099e-02   0.01
                                    3600.0   1.0588425e-02   0.01]),
    (SC_PAKT,  (PAKT, PAKT_S6), 6, [   0.0   0.0000000e+00   0.01
                                      60.0   1.0061314e+00   0.05
                                     120.0   1.1127800e+00   0.04
                                     300.0   3.0429659e-01   0.05
                                     600.0   4.0281846e-02   0.01
                                     900.0   1.4749471e-02   0.01
                                    1800.0   1.4749471e-02   0.01
                                    3600.0   9.1743119e-03   0.01]),
    (SC_PS6,   (PS6,), 1, [   0.0   0.0000000e+00   0.01
                             60.0   8.5344898e-03   0.01
                            120.0   6.6955415e-03   0.01
                            300.0   5.4726298e-03   0.01
                            600.0   9.9161154e-03   0.01
                            900.0   9.0407288e-03   0.01
                           1800.0   2.9881533e-03   0.01
                           3600.0   6.7162533e-04   0.01]),
    (SC_PS6,   (PS6,), 2, [   0.0   0.0000000e+00   0.01
                             60.0   4.9995523e-06   0.01
                            120.0   2.7725194e-05   0.01
                            300.0   3.4766455e-04   0.01
                            600.0   1.1209877e-02   0.01
                            900.0   3.2618783e-02   0.01
                           1800.0   6.0753424e-03   0.01
                           3600.0  -7.4947758e-04   0.01]),
    (SC_PS6,   (PS6,), 3, [   0.0   0.0000000e+00   0.01
                             60.0   6.1492506e-03   0.01
                            120.0   6.2248329e-03   0.01
                            300.0   8.2858830e-03   0.02
                            600.0   4.0732776e-02   0.01
                            900.0   1.3314356e-01   0.01
                           1800.0   1.9442348e-02   0.01
                           3600.0   5.9841288e-03   0.01]),
    (SC_PS6,   (PS6,), 4, [   0.0   0.0000000e+00   0.01
                             60.0   8.7061243e-03   0.01
                            120.0   8.5923618e-03   0.01
                            300.0   1.3731842e-02   0.03
                            600.0   1.0145278e-01   0.02
                            900.0   2.2009104e-01   0.02
                           1800.0   4.5782144e-02   0.01
                           3600.0   6.5694391e-03   0.01]),
    (SC_PS6,   (PS6,), 5, [   0.0   0.0000000e+00   0.01
                             60.0   1.3799008e-02   0.01
                            120.0   1.3799032e-02   0.01
                            300.0   1.3708360e-02   0.01
                            600.0   1.4189141e-01   0.04
                            900.0   2.8473327e-01   0.02
                           1800.0   1.1341547e-01   0.03
                           3600.0   1.6032687e-02   0.02]),
    (SC_PS6,   (PS6,), 6, [   0.0   0.0000000e+00   0.01
                             60.0   6.6615897e-03   0.01
                            120.0   6.6615897e-03   0.01
                            300.0   1.3734175e-02   0.01
                            600.0   1.6416058e-01   0.04
                            900.0   3.2879703e-01  0.015
                           1800.0   1.8462120e-01   0.03
                           3600.0   5.3497993e-02   0.02]),
]
const Nm = sum(size(tbl, 1) for (_, _, _, tbl) in meas)

# ----- Create ExaModel -----

# The physical value of a log10-scaled parameter, in-lined into every kernel
ph(p, m) = exp(log(10.0) * p[m])

function state0(θ)
    z0 = zeros(Nz)
    for (v, m) in ic_p
        z0[v] = 10.0^θ[m]
    end
    return z0
end

# The negative log-likelihood on the pulse data at the nominal θ, from Rodas5P at atol = rtol =
# 1e-12. The nominal column is the optimum for the step data, so here it is a start point and not
# the answer.
#
# PEtab.jl reports 25813.1357 for the same problem, which is this model with EGF never switched
# off: integrating the pulse conditions as steps reproduces 25813.1359. SBMLImporter does not
# deactivate the `time <= EGF_end` gate, so its EGF stays at the dose through t = 3600.
const NLL_REF = 4028.148273723562

function examodel_fujita(θ0 = θnom; K = KDEG, nsub = NSUB, pin = false, zstart = nothing)
    # Mesh: every measurement time is a node, each measurement interval split nsub ways.
    # TPULSE is itself a measurement time, so the pulse edge is a node by construction.
    tmeas = sort(unique(vcat([tbl[:, 1] for (_, _, _, tbl) in meas]...)))
    nodes, imeas = [tmeas[1]], Dict(tmeas[1] => 0)
    for j in 2:length(tmeas)
        for s in 1:nsub
            push!(nodes, tmeas[j-1] + s * (tmeas[j] - tmeas[j-1]) / nsub)
        end
        nodes[end] = tmeas[j]                     # the node is the measurement time itself
        imeas[tmeas[j]] = length(nodes) - 1
    end
    N, ion = length(nodes) - 1, imeas[TPULSE]     # ion = last interval under the pulse

    # Create CollocationExaCore
    core = CollocationExaCore(nodes, K)

    # Create CollocationVariables, started flat at the initial condition unless handed a
    # trajectory. Solving the pinned problem first and starting the estimation from its states
    # is what carries this model, the flat start reaching only a locally infeasible point.
    z0 = state0(θ0)
    zs = zstart === nothing ? [z0[v] for v in 1:Nz, _ in 1:Nc, _ in 1:N, _ in 0:K] : zstart
    @add_var_collocation(core, z, 1:Nz, 1:Nc; start = zs)       # z[v,c,i,k]

    # Create variables (unknown parameters to estimate)
    # pin = true fixes θ and leaves a pure simulation, which is what checks the discretization
    ExaModels.@add_var(core, p, 1:Np;
        lvar = pin ? θ0 : θLB, uvar = pin ? θ0 : θUB, start = θ0)

    # Create iterator. EGF is data and constant across an interval, the pulse edge being a
    # node, so the two receptor equations are written out over (i,k) with the dose in the row
    itr_egf = [(i <= ion ? DOSE[c] : 0.0, c, i, k) for c in 1:Nc, i in 1:N, k in 1:K]

    # Create collocation constraints
    @add_con_collocation(core, coll_free, z[y,c],
        -ph(p, kf) * z[x,c] * z[y,c] + ph(p, kr) * z[xy,c] + ph(p, kd) * z[w,c]
        for (y,x,xy,w,kf,kr,kd) in FREE, c in 1:Nc
    )
    @add_con_collocation(core, coll_cplx, z[xy,c],
        ph(p, kf) * z[a,c] * z[b,c] - (ph(p, kr) + ph(p, kc)) * z[xy,c]
        for (xy,a,b,kf,kr,kc) in CPLX, c in 1:Nc
    )
    @add_con_collocation(core, coll_actv, z[y,c],
        ph(p, ka) * z[up,c] - ph(p, kf) * z[a,c] * z[y,c] +
        (ph(p, kr) + ph(p, kc)) * z[xy,c] - ph(p, kd) * z[y,c]
        for (y,up,ka,a,xy,kf,kr,kc,kd) in ACTV, c in 1:Nc
    )
    @add_con_collocation(core, coll_ps6, z[PS6,c],
        ph(p, K6) * z[PAKT_S6,c] - ph(p, K8) * z[PS6,c]
        for c in 1:Nc
    )
    @add_con_collocation(core, coll_bound, z[EGF_EGFR,c],
        u * ph(p, K1F) * z[EGFR,c] - (ph(p, K1R) + ph(p, K9)) * z[EGF_EGFR,c]
        for (u,c,i,k) in itr_egf
    )
    @add_con_collocation(core, coll_recep, z[EGFR,c],
        -u * ph(p, K1F) * z[EGFR,c] + ph(p, K1R) * z[EGF_EGFR,c] +
        ph(p, TURNOVER) * (EGFR_SS - z[EGFR,c])
        for (u,c,i,k) in itr_egf
    )

    # Create continuity constraints
    @add_con_continuity(core, cont, z)

    # Create initial condition constraints
    ExaModels.@add_con(core, ic_par, z[v,c,1,0] - ph(p, m) for (v,m) in ic_p, c in 1:Nc)
    ExaModels.@add_con(core, ic_zero, z[v,c,1,0] for v in ic_0, c in 1:Nc)

    # Create objective function. A t = 0 sample reads (i,kk) = (1,0), every other one the
    # right end of its interval
    obj2 = Tuple{Int,Int,Int,Int,Int,Int,Float64,Float64,Float64}[]
    obj1 = Tuple{Int,Int,Int,Int,Int,Float64,Float64,Float64}[]
    for (ms, vs, c, tbl) in meas, row in axes(tbl, 1)
        tm, ym, sd = tbl[row, 1], tbl[row, 2], tbl[row, 3]
        i, kk = imeas[tm] == 0 ? (1, 0) : (imeas[tm], K)
        cst = log(sd) + 0.5 * log(2π)
        length(vs) == 2 ? push!(obj2, (ms, vs[1], vs[2], c, i, kk, ym, sd, cst)) :
                          push!(obj1, (ms, vs[1], c, i, kk, ym, sd, cst))
    end
    ExaModels.@add_obj(core,
        0.5 * ((ph(p, ms) * (z[v1,c,i,kk] + z[v2,c,i,kk]) - ym) / sd)^2 + cst
        for (ms,v1,v2,c,i,kk,ym,sd,cst) in obj2
    )
    ExaModels.@add_obj(core,
        0.5 * ((ph(p, ms) * z[v,c,i,kk] - ym) / sd)^2 + cst
        for (ms,v,c,i,kk,ym,sd,cst) in obj1
    )

    return ExaModel(core)
end

# ----- Solve -----

# Simulate at the nominal θ, which pins the discretization against NLL_REF
sim = examodel_fujita(; pin = true)
rsim = madnlp(sim; tol = 1e-8, max_iter = 3000)
println("simulate at nominal θ")
println("  status = $(rsim.status)")
println("  nllh   = $(rsim.objective), against $NLL_REF")

# Estimate, started from that trajectory
model = examodel_fujita(; zstart = solution(rsim, sim.z))
result = madnlp(model; tol = 1e-8, max_iter = 3000)
θsol = solution(result, model.p)
println("estimate")
println("  status = $(result.status)")
println("  nllh   = $(result.objective), from $(rsim.objective) at the nominal θ")

# ----- Plot -----

const OBS = [(SC_PEGFR, (PEGFR, PEGFR_AKT), "pEGFR_tot"),
             (SC_PAKT,  (PAKT, PAKT_S6),    "pAkt_tot"),
             (SC_PS6,   (PS6,),             "pS6_tot")]

tgrid = collect(range(0.0, TEND; length = 601))
zfit = interpolate(model, result, model.z, tgrid)
rsol = 10.0 .^ θsol

# The gate as the model reads it: one value per interval, held over [0, TPULSE] and 0 after
pu = plot(
    tgrid, [t <= TPULSE ? DOSE[c] : 0.0 for t in tgrid, c in 1:Nc];
    linewidth = 2, color = (1:Nc)', xlims = (0.0, TEND), legend = :topright,
    label = reshape(["$d ng/mL" for d in DOSE], 1, :),
    ylabel = "EGF [ng/mL]", title = "gate profile",
)

panels = map(OBS) do (ms, vs, name)
    yfit = [rsol[ms] * sum(zt[v, c] for v in vs) for zt in zfit, c in 1:Nc]
    q = plot(
        tgrid, yfit;
        linewidth = 2, color = (1:Nc)', xlims = (0.0, TEND), legend = false,
        xlabel = (name == "pS6_tot" ? "t [s]" : ""), ylabel = name,
    )
    for (msq, _, c, tbl) in meas
        msq == ms || continue
        scatter!(q, tbl[:, 1], tbl[:, 2];
                 yerror = tbl[:, 3], markersize = 3, color = c, label = false)
    end
    q
end

for q in (pu, panels...)
    vline!(q, [TPULSE]; color = :black, linestyle = :dash, linewidth = 1, label = false)
end

png = joinpath(@__DIR__, "fujita.png")
savefig(plot(pu, panels...;
             layout = grid(4, 1; heights = [0.22, 0.26, 0.26, 0.26]), size = (900, 1000)), png)
println("wrote $png")
