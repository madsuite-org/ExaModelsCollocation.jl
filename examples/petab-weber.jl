# Weber et al. (2015), BMC Systems Biology 9:9
# Weber_BMC2015 from the Benchmarking Initiative's PEtab Collection

ENV["GKSwstype"] = "100"

using ExaModels
using ExaModelsCollocation
using MadNLP
using Plots

# ----- Problem data from PEtab file -----

const PKD, PKDDAGa, PI4K3B, PI4K3Ba, CERTERa, CERT, CERTTGNa = 1:7
const Nz, Nc = 7, 2

const DATA2, DATA3 = 1, 2

const KDEG, NSUB = 4, 4

const ZSS0 = [466534.7994, 123.8608, 1577540.5394, 332054.5041,
              31948388.5902, 160797.7364, 42082828.6681]

const A11, A12, A21, A22, A31, A32, A33 = 1:7
const M11, M22, M31, M33 = 8:11
const P11, P12, P13, P21, P22, P31, P32, P33 = 12:19
const PU3, PU4, PU5, PU6 = 20:23
const S12, S21, S31 = 24:26
const SC_CERTpRN24, SC_PI4K3BpRN24, SC_PKDpN0, SC_PKDpN24, SC_PKDpN25 = 27:31
const SD_PKDpN0, SD_PKDpN24, SD_PKDpN25, SD_PI4K3BpRN24, SD_CERTpRN24 = 32:36

const pnom = [
    0.182532326367113,  # a11
    26.6713328753608,   # a12
    1.87704479834148,   # a21
    0.0001,             # a22
    0.135335059339132,  # a31
    0.0001,             # a32
    0.0001,             # a33
    1.0e10,             # m11
    2153.51392335252,   # m22
    9970035709.05734,   # m31
    41871228.8317411,   # m33
    4.24383990916234,   # p11
    0.005935939292274,  # p12
    0.002460257477278,  # p13
    5.6290367522644,    # p21
    21.7856860653472,   # p22
    2537.97413617078,   # p31
    16.7677390849425,   # p32
    21658.8400024158,   # p33
    1.0e8,              # pu3
    33133606.9433133,   # pu4
    33.6959765525242,   # pu5
    113.366007469962,   # pu6
    88461.2150536401,   # s12
    2961147.46928213,   # s21
    4327961.43037034,   # s31
    121.028409854968,   # scale_yCERTpRN24
    3.39366857781642,   # scale_yPI4K3BpRN24
    0.007176447966379,  # scale_yPKDpN0
    0.003370141139919,  # scale_yPKDpN24
    0.000237349365354,  # scale_yPKDpN25
    0.974100018572643,  # std_yPKDpN0
    1.08030443428794,   # std_yPKDpN24
    0.117466359121065,  # std_yPKDpN25
    0.414223339385595,  # std_yPI4K3BpRN24
    0.066162154188523,  # std_yCERTpRN24
]
const Np = length(pnom)
const θnom = log10.(pnom)
const θLB = log10.([fill(1e-4, 7); fill(1e2, 4); fill(1e-5, 8); fill(1e-1, 4);
                    fill(1e2, 3); fill(1e-5, 5); fill(1e-2, 5)])
const θUB = log10.([fill(1e2, 7); fill(1e10, 4); fill(1e5, 8); fill(1e8, 4);
                    fill(1e7, 3); fill(1e3, 5); fill(1e2, 5)])

const SD_PKDt, SD_PI4K3Bt, SD_CERTt = 1910527.835, 887676.131, 244727220.419

const MEAS_SUM2 = [
    (PKD, PKDDAGa, SD_PKDt, DATA3, [0.0 466570.0; 0.0 466570.0; 0.0 466570.0]),
    (PI4K3B, PI4K3Ba, SD_PI4K3Bt, DATA2, [24.0 8.00343e7; 24.0 8.00343e7; 24.0 8.00343e7]),
    (PI4K3B, PI4K3Ba, SD_PI4K3Bt, DATA3, [0.0 1.91053e6; 0.0 1.91053e6; 0.0 1.91053e6]),
]

const MEAS_SUM3 = [
    (DATA2, [0.0 509861.0; 0.0 509861.0; 0.0 509861.0]),
    (DATA3, [24.0 4.65126e8; 24.0 4.65126e8; 24.0 4.65126e8]),
]

const MEAS_SCALED = [
    (SC_PKDpN0, SD_PKDpN0, DATA2, [0.0 1.0; 3.0 1.61952; 6.0 1.62347;
                                   0.0 1.0; 3.0 1.04145; 6.0 1.19777;
                                   0.0 1.0; 3.0 1.11665; 6.0 1.16507]),
    (SC_PKDpN0, SD_PKDpN0, DATA3, [0.0 1.0; 3.0 2.06292; 6.0 1.69749; 24.0 6.19963;
                                   24.25 4.51439; 24.5 5.67509; 25.0 5.74541; 26.0 4.61973; 27.0 3.92466;
                                   0.0 1.0; 3.0 1.7144; 6.0 1.72091; 24.0 3.26188;
                                   24.25 2.79688; 24.5 2.64929; 25.0 2.49079; 26.0 1.63371; 27.0 2.9095;
                                   0.0 1.0; 3.0 1.46472; 6.0 1.98961; 24.0 3.46662;
                                   24.25 1.42776; 24.5 1.95708; 25.0 2.2429; 26.0 3.19321; 27.0 1.27776;
                                   0.0 1.0; 3.0 1.16913; 6.0 1.7192]),
    (SC_PKDpN24, SD_PKDpN24, DATA2, [24.0 1.0; 27.0 10.8539; 30.0 7.90753;
                                     24.0 1.0; 27.0 7.67389; 30.0 5.26782;
                                     24.0 1.0; 27.0 8.61855; 30.0 8.15543]),
    (SC_PKDpN25, SD_PKDpN25, DATA2, [24.0 0.0967194; 24.08 0.76386; 24.17 0.875442;
                                     24.25 0.875946; 24.5 1.31372; 25.0 1.0;
                                     24.0 0.170906; 24.08 0.910004; 24.17 1.07457;
                                     24.25 1.27322; 24.5 1.30927; 25.0 1.0;
                                     24.0 0.0957289; 24.08 0.784837; 24.17 1.16906;
                                     24.25 1.25508; 24.5 1.24719; 25.0 1.0;
                                     24.08 0.822255; 24.17 0.825694; 24.25 1.38294;
                                     24.5 1.0487; 25.0 1.0]),
]

const MEAS_RATIO2 = [
    (SC_PI4K3BpRN24, SD_PI4K3BpRN24, DATA2, [24.0 1.0; 24.08 1.58426; 24.17 2.4846;
                                             24.25 2.72911; 24.5 1.82027; 25.0 2.99535; 27.0 2.27924; 30.0 2.16054;
                                             24.0 1.0; 24.08 2.83372; 24.17 1.76735;
                                             24.25 1.59287; 24.5 2.61621; 25.0 1.78642; 27.0 2.50706; 30.0 2.34829;
                                             24.0 1.0; 24.17 1.84135; 24.25 1.75292;
                                             24.5 2.67541; 25.0 1.8798; 27.0 2.55858; 30.0 2.6126;
                                             24.0 1.0; 24.25 2.91676; 25.0 2.77524;
                                             24.0 1.0; 24.25 2.73995; 25.0 2.67654]),
]

const MEAS_RATIO3 = [
    (SC_CERTpRN24, SD_CERTpRN24, DATA3, [24.0 1.0; 24.25 0.962125; 24.5 0.949774;
                                         25.0 0.895086; 26.0 0.886877; 27.0 0.917778;
                                         24.0 1.0; 24.25 0.748912; 24.5 0.846352;
                                         25.0 0.854435; 26.0 0.709759; 27.0 0.889224;
                                         24.0 1.0; 24.25 0.979999; 24.5 0.788311;
                                         25.0 0.808387; 26.0 0.867731; 27.0 0.829811;
                                         24.0 1.0; 24.5 0.971393]),
]

# Known optimal solution
const NLL_REF = 296.206

# ----- right-hand side function -----

# Transform log varibale to linear space
ph(p, m) = exp(log(10.0) * p[m])

# right hand side functions
function weber_rhs(z, p, u3, u4, u5, u6)
    pkd, pdag, pi3b, pi3ba, certe, cert, certt = z[1], z[2], z[3], z[4], z[5], z[6], z[7]
    mm  = ph(p, P31) * certe * pi3ba / (pi3ba + ph(p, M31))
    R1  = ph(p, P11) * pkd * mm / (ph(p, M11) + mm)
    R2  = ph(p, P12) * pkd * (ph(p, PU5) * u5 + 1)
    R3  = ph(p, P13) * pdag * (ph(p, PU6) * u6 + 1)
    R9  = ph(p, P22) * pi3b * pdag / (pdag + ph(p, M22))
    R16 = ph(p, P33) * certt * pdag / (pdag + ph(p, M33))
    return [
        -R1 - R2 + R3 + ph(p, S12) - ph(p, A11) * pkd,                       # PKD
         R1 + R2 - R3 - ph(p, A12) * pdag,                                   # PKDDAGa
         ph(p, P21) * pi3ba - R9 + ph(p, S21) + ph(p, PU3) * u3 - ph(p, A21) * pi3b,  # PI4K3B
        -ph(p, P21) * pi3ba + R9 - ph(p, A22) * pi3ba,                       # PI4K3Ba
        -mm + ph(p, P32) * cert + ph(p, S31) + ph(p, PU4) * u4 - ph(p, A31) * certe,  # CERTERa
        -ph(p, P32) * cert + R16 - ph(p, A32) * cert,                        # CERT
         mm - R16 - ph(p, A33) * certt,                                      # CERTTGNa
    ]
end

rhs_v(v, zc, p, u3, u4, u5, u6) = weber_rhs(zc, p, u3, u4, u5, u6)[v]

# ----- Obtain good initial guess for discretized states -----

# Explicit steps sized so the stiff PKDDAGa decay stays stable
function RK4(f, z0, ts; dtmax = 0.02)
    out = Vector{Vector{Float64}}(undef, length(ts))
    z, t = copy(z0), ts[1]
    out[1] = copy(z)
    for n in 2:length(ts)
        m = max(1, ceil(Int, (ts[n] - t) / dtmax))
        dt = (ts[n] - t) / m
        for _ in 1:m
            k1 = f(z, t)
            k2 = f(z .+ (dt / 2) .* k1, t + dt / 2)
            k3 = f(z .+ (dt / 2) .* k2, t + dt / 2)
            k4 = f(z .+ dt .* k3, t + dt)
            z = z .+ (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
            t += dt
        end
        t = ts[n]
        out[n] = copy(z)
    end
    return out
end

# ----- Create ExaModel -----

# Known control profile
u_input(c, t) = t <= 24.0 ? (0.0, 0.0) : c == DATA2 ? (1.0, 0.0) : (0.0, 10.0)

function examodel_weber(θ0 = θnom; K = KDEG, nsub = NSUB, backend = nothing)
    # Create mesh nodes
    tabs = vcat(MEAS_SUM2, MEAS_SUM3, MEAS_SCALED, MEAS_RATIO2, MEAS_RATIO3)
    tmeas = sort(unique(vcat([e[end][:, 1] for e in tabs]...)))
    nodes, imeas = [tmeas[1]], Dict(tmeas[1] => 0)
    for j in 2:length(tmeas)
        for s in 1:nsub
            push!(nodes, tmeas[j-1] + s * (tmeas[j] - tmeas[j-1]) / nsub)
        end
        nodes[end] = tmeas[j]
        imeas[tmeas[j]] = length(nodes) - 1
    end
    N = length(nodes) - 1

    # Create CollocationExaCore
    core = CollocationExaCore(nodes, K; backend)

    # Solve ODE system at nominal θ to obtain good initial guess
    h, taus = diff(core.nodes), core.weights.taus
    ik = [(i, k) for i in 1:N for k in 0:K]
    ts = [core.nodes[i] + (k == 0 ? 0.0 : h[i] * taus[k]) for (i, k) in ik]
    zstart = Array{Float64}(undef, Nz, Nc, N, K + 1)
    for c in 1:Nc
        u3, u4 = c == DATA2 ? (1.0, 0.0) : (0.0, 1.0)
        prof = RK4((z, t) -> weber_rhs(z, θ0, u3, u4, u_input(c, t)...), ZSS0, ts)
        for (n, (i, k)) in enumerate(ik)
            zstart[:,c,i,k+1] .= prof[n]
        end
    end

    # Create CollocationVariables
    @add_var_collocation(core, z, 1:Nz, 1:Nc; start = zstart)

    # Create variables (unknown parameters to estimate)
    @add_var(core, p, 1:Np; lvar = θLB, uvar = θUB, start = θ0)

    # Create pre-equilibration steady state constraints
    @add_var(core, zss, 1:Nz; start = ZSS0)
    for v in 1:Nz
        core, _ = add_con(core, weber_rhs(zss, p, 0.0, 0.0, 0.0, 0.0)[v])
    end

    # Create collocation constraints
    itr = [
        (
            (c == DATA2 ? (1.0, 0.0) : (0.0, 1.0))...,
            u_input(c, core.mesh.t[i,1])..., 
            c, i, k
        )
        for c in 1:Nc, i in 1:N, k in 1:K
    ]
    for v in 1:Nz
        @add_con_collocation(core, z[v,c],
            rhs_v(v, ntuple(w -> z[w,c], Nz), p, u3, u4, u5, u6)
            for (u3, u4, u5, u6, c, i, k) in itr
        )
    end

    # Create continuity constraints
    @add_con_continuity(core, cont, z)

    # Create initial condition constraints
    @add_con(core, ic, z[v,c,1,0] - zss[v] for v in 1:Nz, c in 1:Nc)

    # Create objective function
    node_of(tm) = imeas[tm] == 0 ? (1, 0) : (imeas[tm], KDEG)
    lc = 0.5 * log(2π)

    _rows(meas) = [(meta..., node_of(tbl[r,1])..., tbl[r,2])
                   for (meta..., tbl) in meas for r in axes(tbl, 1)]

    @add_obj(core,
        0.5 * ((z[v1,c,i,kk] + z[v2,c,i,kk] - ym) / sd)^2 + log(sd) + lc
        for (v1, v2, sd, c, i, kk, ym) in _rows(MEAS_SUM2)
    )
    @add_obj(core,
        0.5 * ((z[CERT,c,i,kk] + z[CERTERa,c,i,kk] + z[CERTTGNa,c,i,kk] - ym) / SD_CERTt)^2 + log(SD_CERTt) + lc
        for (c, i, kk, ym) in _rows(MEAS_SUM3)
    )
    @add_obj(core,
        0.5 * ((ph(p, sc) * z[PKDDAGa,c,i,kk] - ym) / ph(p, sd))^2 + log(10.0) * p[sd] + lc
        for (sc, sd, c, i, kk, ym) in _rows(MEAS_SCALED)
    )
    @add_obj(core,
        0.5 * ((ph(p, sc) * z[PI4K3Ba,c,i,kk] / (z[PI4K3B,c,i,kk] + z[PI4K3Ba,c,i,kk]) - ym) / ph(p, sd))^2 + log(10.0) * p[sd] + lc
        for (sc, sd, c, i, kk, ym) in _rows(MEAS_RATIO2)
    )
    @add_obj(core,
        0.5 * ((ph(p, sc) * z[CERT,c,i,kk] / (z[CERT,c,i,kk] + z[CERTERa,c,i,kk] + z[CERTTGNa,c,i,kk]) - ym) / ph(p, sd))^2 + log(10.0) * p[sd] + lc
        for (sc, sd, c, i, kk, ym) in _rows(MEAS_RATIO3)
    )

    return ExaModel(core)
end

# ----- Solve -----

using MadNLPGPU, CUDA, CUDSS

# Create CollocationExaModel
model = examodel_weber(backend = CUDA.CUDABackend())

# Solve
@time result = madnlp(model; tol = 1e-6, max_iter = 1000)

θsol = solution(result, model.p)
println("status    = $(result.status)")
println("nllh      = $(result.objective), reference = $NLL_REF")
println("|nll-ref| = $(abs(result.objective - NLL_REF))")

# ----- Plot: model observables vs measured data -----

tgrid = collect(range(0.0, 30.0; length = 601))
zsol = Array(solution(result, model.z))
psol = Array(solution(result, model.p))
zfit = interpolate(model, zsol, model.z, tgrid)

s_PKDpN0      = ph(psol, SC_PKDpN0)
s_PKDpN24     = ph(psol, SC_PKDpN24)
s_PKDpN25     = ph(psol, SC_PKDpN25)
s_PI4K3BpRN24 = ph(psol, SC_PI4K3BpRN24)
s_CERTpRN24   = ph(psol, SC_CERTpRN24)

pkdt(Z, c)  = Z[PKD,c] + Z[PKDDAGa,c]
pi3bt(Z, c) = Z[PI4K3B,c] + Z[PI4K3Ba,c]
certt(Z, c) = Z[CERT,c] + Z[CERTERa,c] + Z[CERTTGNa,c]

specs = [
    ("total PKD",     DATA3, :log10,   Z -> pkdt(Z, DATA3),  MEAS_SUM2[1][5]),
    ("total PI4K3B",  DATA2, :log10,   Z -> pi3bt(Z, DATA2), MEAS_SUM2[2][5]),
    ("total PI4K3B",  DATA3, :log10,   Z -> pi3bt(Z, DATA3), MEAS_SUM2[3][5]),
    ("total CERT",    DATA2, :log10,   Z -> certt(Z, DATA2), MEAS_SUM3[1][2]),
    ("total CERT",    DATA3, :log10,   Z -> certt(Z, DATA3), MEAS_SUM3[2][2]),
    ("PKDpN0",        DATA2, :identity, Z -> s_PKDpN0  * Z[PKDDAGa,DATA2], MEAS_SCALED[1][4]),
    ("PKDpN0",        DATA3, :identity, Z -> s_PKDpN0  * Z[PKDDAGa,DATA3], MEAS_SCALED[2][4]),
    ("PKDpN24",       DATA2, :identity, Z -> s_PKDpN24 * Z[PKDDAGa,DATA2], MEAS_SCALED[3][4]),
    ("PKDpN25",       DATA2, :identity, Z -> s_PKDpN25 * Z[PKDDAGa,DATA2], MEAS_SCALED[4][4]),
    ("PI4K3BpRN24",   DATA2, :identity, Z -> s_PI4K3BpRN24 * Z[PI4K3Ba,DATA2] / pi3bt(Z, DATA2), MEAS_RATIO2[1][4]),
    ("CERTpRN24",     DATA3, :identity, Z -> s_CERTpRN24 * Z[CERT,DATA3] / certt(Z, DATA3), MEAS_RATIO3[1][4]),
]

groups = [
    (DATA2, :log10,    "data2: observables (log)"),
    (DATA3, :log10,    "data3: observables (log)"),
    (DATA2, :identity, "data2: observables (linear)"),
    (DATA3, :identity, "data3: observables (linear)"),
]

panels = map(groups) do (gc, gys, gttl)
    members = [s for s in specs if s[2] == gc && s[3] == gys]
    q = plot(;
        yscale = gys, legend = :topleft, xlims = (0.0, 30.0),
        xlabel = "t [min]", ylabel = "observable", title = gttl,
    )
    allvals = Float64[]
    for (idx, (nm, c, ys, g, tbl)) in enumerate(members)
        ycurve = [g(zt) for zt in zfit]
        plot!(q, tgrid, ycurve; linewidth = 2, color = idx, label = nm)
        scatter!(q, tbl[:,1], tbl[:,2];
            markersize = 4, markerstrokewidth = 0, color = idx, label = false)
        append!(allvals, ycurve)
        append!(allvals, tbl[:,2])
    end
    lo, hi = minimum(allvals), maximum(allvals)
    yl = gys === :log10 ? (lo / 1.5, hi * 3.0) : (min(0.0, lo), hi + 0.15 * (hi - lo))
    vspan!(q, [24.0, 30.0]; color = :black, alpha = 0.06, label = false)
    plot!(q; ylims = yl)
    q
end

for (c, dlab, stim) in [(DATA2, "data2", "PdBu"), (DATA3, "data3", "NB142-70")]
    Z = [zt[v,c] for zt in zfit, v in 1:Nz]
    smin, smax = minimum(Z), maximum(Z)
    ctl = [t > 24.0 ? 1.0 : 0.0 for t in tgrid]
    qs = plot(
        tgrid, Z;
        yscale = :log10, linewidth = 2, color = (1:Nz)', legend = false,
        xlims = (0.0, 30.0), ylims = (smin / 2, smax * 3),
        xticks = [0, 10, 20, 24, 30],
        xlabel = "t [min]", ylabel = "concentration", title = "$dlab: states + control ($stim)",
    )
    vspan!(qs, [24.0, 30.0]; color = :black, alpha = 0.07, label = false)
    plot!(twinx(qs), tgrid, ctl;
        color = :black, linewidth = 3, label = "u(t)",
        legend = (0.1, 0.35),
        xlims = (0.0, 30.0), ylims = (-0.05, 1.05))
    push!(panels, qs)
end

png = joinpath(@__DIR__, "petab-weber.png")
savefig(plot(panels...; layout = (3, 2), size = (1200, 1500),
             left_margin = 5Plots.mm, bottom_margin = 5Plots.mm), png)
println("wrote $png")
