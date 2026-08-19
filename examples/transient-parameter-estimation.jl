# Transient parameter estimation under a known control profile
#
#   min_p  sum_m 0.5 ((z_v(t_m) - y_vm)/sd)^2
#   s.t.   cA' = -k(u) cA
#          cB' =  k(u) cA
#          k(T) = exp(p1 - p2 (Tref/T - 1)),  p1 = log k(Tref),  p2 = Ea/(R Tref)
#          (cA, cB)(0) = (cA0, 0)
#
# A -> B in a batch reactor under a known temperature program: a classic pulse experiment, a base
# temperature with one rectangular pulse. u is data rather than a variable, so the iterator is
# written out over (i, k) and carries the profile of interval i in the row, with the sign that
# tells the two states apart. The mesh is the measurement times rather than a uniform grid.

ENV["GKSwstype"] = "100"

using ExaModels
using ExaModelsCollocation
using MadNLP
using Plots

# ----- Problem data -----

const CA, CB = 1, 2
const CA0, TEND, TREF, SD = 1.0, 60.0, 310.0, 0.02

const TBASE, TPULSE, TON, TOFF = 300.0, 325.0, 10.0, 20.0

# p = (log k(Tref), Ea/(R Tref)), the values the data was simulated at
const PTRUE = [log(0.05), 6000.0 / TREF]

# Simulated data: time, cA, cB, each with Gaussian noise of standard deviation SD
const MEAS = [
     4.0   0.8863   0.0990
     8.0   0.8399   0.1790
    12.0   0.6136   0.3825
    16.0   0.3641   0.6512
    20.0   0.2093   0.7624
    24.0   0.1980   0.8109
    28.0   0.1963   0.8060
    32.0   0.1454   0.8342
    36.0   0.1463   0.8523
    40.0   0.1736   0.8430
    44.0   0.1008   0.8930
    48.0   0.0894   0.8960
    52.0   0.0876   0.9194
    56.0   0.1073   0.9511
    60.0   0.0724   0.9192
]

# ----- Create ExaModel -----

# Control profile u(t)
u_profile(t) = TON <= t <= TOFF ? TPULSE : TBASE

function examodel_transient_pe(p0 = [log(0.02), 12.0]; K = 4)
    # Mesh: the horizon ends, the measurement times, and the two pulse edges
    nodes = sort(union([0.0, TEND], MEAS[:, 1], [TON, TOFF]))
    N = length(nodes) - 1

    # Create CollocationExaCore
    core = CollocationExaCore(nodes, K)

    # Create CollocationVariable
    @add_var_collocation(core, z, 1:2; start = CA0 / 2) # z1, z2 = (cA, cB)

    # Create variables (unknown kinetic parameters)
    @add_var(core, p, 1:2; 
        lvar = [log(1e-4), 0.0],
        uvar = [log(10.0), 60.0],
        start = p0
    )

    # Create iterator
    itr = [
        (v, sgn, u_profile(core.mesh.t[i, 1]), i, k)
        for (v, sgn) in [(CA, -1.0), (CB, 1.0)], i in 1:N, k in 1:K
    ]

    # Create collocation constraints
    @add_con_collocation(core, coll, z[v],
        sgn * exp(p[1] - p[2] * (TREF / u - 1)) * z[CA]
        for (v, sgn, u, i, k) in itr
    )

    # Create continuity constraints
    @add_con_continuity(core, cont, z)

    # Create initial condition constraints
    @add_con(core, ic, z[v, 1, 0] - val for (v, val) in [(CA, CA0), (CB, 0.0)])

    # Create objective function. Every sample time is a node, so it is an interval's right end
    itr_obj = [(v, findfirst(==(MEAS[m, 1]), nodes) - 1, MEAS[m, v + 1])
               for m in axes(MEAS, 1), v in 1:2]
    @add_obj(core, 0.5 * ((z[v, i, K] - ym) / SD)^2 for (v, i, ym) in itr_obj)

    return ExaModel(core)
end

# ----- Solve -----

# Create CollocationExaModel
model = examodel_transient_pe()

# Solve
result = madnlp(model; tol = 1e-8)

psol = solution(result, model.p)
println("status    = $(result.status)")
println("objective = $(result.objective)")
println("k(Tref)   = $(exp(psol[1])), simulated at $(exp(PTRUE[1]))")
println("Ea/R      = $(psol[2] * TREF), simulated at $(PTRUE[2] * TREF)")

# ----- Plot -----

tgrid = collect(range(0.0, TEND; length = 601))
zfit = permutedims(reduce(hcat, interpolate(model, result, model.z, tgrid)))

pu = plot(
    tgrid, u_profile.(tgrid);
    color = :black, linewidth = 2, legend = false, xlims = (0.0, TEND),
    ylabel = "T [K]", title = "known control profile",
)
pz = plot(
    tgrid, zfit;
    linewidth = 2, label = ["cA" "cB"], legend = :right, xlims = (0.0, TEND),
    xlabel = "t [min]", ylabel = "concentration", title = "fit against simulated data",
)
scatter!(pz, MEAS[:, 1], MEAS[:, 2:3];
    yerror = SD, markersize = 4, color = [1 2], label = ["cA data" "cB data"])

png = joinpath(@__DIR__, "transient-parameter-estimation.png")
savefig(plot(pu, pz; layout = grid(2, 1; heights = [0.3, 0.7]), size = (900, 640)), png)
println("wrote $png")
