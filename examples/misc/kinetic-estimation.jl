# Transient Kinetic Parameter Estimation
#
#   min  sum_m sum_v 0.5 ((c_v(t_m) - y_vm)/sd)^2
#   s.t. cA' = -k1(T) cA
#        cB' =  k1(T) cA - k2(T) cB
#        cC' =  k2(T) cB
#        kr(T) = exp(pr1 - pr2 (Tref/T - 1)),  r = 1, 2
#        T(t) = Tpulse for Ton <= t <= Toff, Tbase otherwise
#        cA(0) = cA0,  cB(0) = 0,  cC(0) = 0
#
#   cA, cB, cC : concentrations of species A, B, C
#   T          : temperature, the known control profile
#   k1, k2     : Arrhenius rate constants of A -> B and B -> C
#   pr1        : log kr(Tref), estimated
#   pr2        : Ear/(R Tref), estimated
#   Tref       : reference temperature
#   Ear        : activation energy of reaction r
#   R          : gas constant
#   y_vm       : measured concentration of species v at t_m
#   sd         : measurement standard deviation
#   cA0        : initial concentration of A
#   Tbase      : base temperature
#   Tpulse     : pulse temperature
#   Ton, Toff  : pulse window
#   tend       : final time

ENV["GKSwstype"] = "100"

using BenchmarkTools, Plots

using ExaModels, ExaModelsCollocation
using MadNLP, MadNLPHSL
using MadNLPGPU, CUDA, CUDSS

# ----- Problem data -----

const CA, CB, CC = 1, 2, 3
const CA0, TEND, TREF, SD = 1.0, 60.0, 310.0, 0.02

const TBASE, TPULSE, TON, TOFF = 300.0, 340.0, 10.0, 20.0

const PTRUE = [log(0.02), 6300.0 / TREF, log(0.31), 2000.0 / TREF]
const P0 = [log(0.01), 12.0, log(0.5), 4.0]

# Simulated data: time, cA, cB, cC, each with Gaussian noise of standard deviation SD
const MEAS = [
     4.0   0.9796   0.0432   0.0150
     8.0   0.9024   0.0040   0.0267
    12.0   0.7284   0.1221   0.1712
    16.0   0.4386   0.1268   0.4516
    20.0   0.2596   0.0395   0.6279
    24.0   0.2319   0.0134   0.6858
    28.0   0.3046   0.0222   0.7211
    32.0   0.2710  -0.0234   0.7414
    36.0   0.2461  -0.0105   0.7331
    40.0   0.2041   0.0283   0.7534
    44.0   0.2270  -0.0293   0.7738
    48.0   0.2262   0.0134   0.8023
    52.0   0.2137   0.0165   0.7711
    56.0   0.1902   0.0061   0.8254
    60.0   0.2001   0.0385   0.8358
]

# ----- Create ExaModel -----

# Control profile u(t)
u_profile(t) = TON <= t <= TOFF ? TPULSE : TBASE

function kinetic_estimation_model(; K = 4, backend = nothing)
    # Define mesh nodes at the measurement times and control switches
    nodes = sort(union([0.0, TEND], MEAS[:, 1], [TON, TOFF]))
    N = length(nodes) - 1

    # Create CollocationExaCore
    core = CollocationExaCore(nodes, K; backend = backend)

    # Create CollocationVariable
    @add_var_collocation(core, z, 1:3; start = CA0 / 3) # z1, z2, z3 = (cA, cB, cC)

    # Create variables (unknown kinetic parameters)
    @add_var(core, p, 1:4;
        lvar = [log(1e-4), 0.0, log(1e-4), 0.0],
        uvar = [log(10.0), 60.0, log(10.0), 60.0],
        start = P0
    )

    # Create iterator
    itr = [
        (u_profile(core.mesh.t[i, 1]), i, k)
        for i in 1:N, k in 1:K
    ]

    # Create collocation constraints
    @add_con_collocation(core, cA, z[CA],
        -exp(p[1] - p[2] * (TREF / u - 1)) * z[CA]
        for (u, i, k) in itr
    )
    @add_con_collocation(core, cB, z[CB],
        exp(p[1] - p[2] * (TREF / u - 1)) * z[CA] - exp(p[3] - p[4] * (TREF / u - 1)) * z[CB]
        for (u, i, k) in itr
    )
    @add_con_collocation(core, cC, z[CC],
        exp(p[3] - p[4] * (TREF / u - 1)) * z[CB]
        for (u, i, k) in itr
    )

    # Create continuity constraints
    @add_con_continuity(core, cont, z)

    # Create initial condition constraints
    @add_con(core, ic,
        z[v, 1, 0] - val
        for (v, val) in [(CA, CA0), (CB, 0.0), (CC, 0.0)]
    )

    # Create objective function
    itr_obj = [
        (v, findfirst(==(MEAS[m, 1]), nodes) - 1, MEAS[m, v + 1])
        for m in axes(MEAS, 1), v in 1:3
    ]
    @add_obj(core,
        0.5 * ((z[v, i, K] - ym) / SD)^2
        for (v, i, ym) in itr_obj
    )

    return ExaModel(core)
end

# ----- Solve -----

# Solve with CPU
model_cpu = kinetic_estimation_model()
result_cpu = @btime madnlp(model_cpu; tol = 1e-6, print_level = MadNLP.ERROR,
    kkt_system = MadNLP.SparseCondensedKKTSystem,
    equality_treatment = MadNLP.RelaxEquality,
    fixed_variable_treatment = MadNLP.RelaxBound,
    linear_solver = Ma57Solver,
)

# Solve with GPU
model_gpu = kinetic_estimation_model(backend = CUDA.CUDABackend())
result_gpu = @btime madnlp(model_gpu; tol = 1e-6, print_level = MadNLP.ERROR,
    kkt_system = MadNLP.SparseCondensedKKTSystem,
    equality_treatment = MadNLP.RelaxEquality,
    fixed_variable_treatment = MadNLP.RelaxBound,
)

# ----- Display results -----

function report(label, model, result)
    psol = Array(solution(result, model.p))
    println("$label status   = $(result.status)")
    println("$label obj      = $(result.objective)")
    println("$label k1(Tref) = $(exp(psol[1])), simulated at $(exp(PTRUE[1]))")
    println("$label Ea1/R    = $(psol[2] * TREF), simulated at $(PTRUE[2] * TREF)")
    println("$label k2(Tref) = $(exp(psol[3])), simulated at $(exp(PTRUE[3]))")
    println("$label Ea2/R    = $(psol[4] * TREF), simulated at $(PTRUE[4] * TREF)")
end
report("cpu", model_cpu, result_cpu)
report("gpu", model_gpu, result_gpu)

# ----- Plot -----

tgrid = collect(range(0.0, TEND; length = 601))
zfit = permutedims(reduce(hcat, interpolate(model_cpu, result_cpu, model_cpu.z, tgrid)))

clo, chi = -0.07, 1.05
f0 = (0.0 - clo) / (chi - clo)
thi = TPULSE + 5.0
tlo = (TBASE - f0 * thi) / (1 - f0)

pz = plot(
    tgrid, zfit;
    linewidth = 2, label = ["cA" "cB" "cC"], legend = :right, xlims = (0.0, TEND),
    ylims = (clo, chi),
    xlabel = "t [min]", ylabel = "concentration", title = "fit against simulated data",
    left_margin = 5Plots.mm, right_margin = 15Plots.mm, bottom_margin = 5Plots.mm,
)
scatter!(pz, MEAS[:, 1], MEAS[:, 2:4];
    yerror = SD, markersize = 4, color = [1 2 3], label = ["cA data" "cB data" "cC data"])

pu = twinx(pz)
plot!(pu, tgrid, u_profile.(tgrid);
    color = :black, linewidth = 2, linestyle = :dash, legend = false,
    xlims = (0.0, TEND), ylims = (tlo, thi), ylabel = "T [K]")

png = joinpath(@__DIR__, splitext(basename(@__FILE__))[1] * ".png")
savefig(plot(pz; size = (900, 480)), png)
println("wrote $png")
