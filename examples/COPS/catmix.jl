# COPS 3.0 Problem 14: Catalyst Mixing
#
# E. D. Dolan, J. J. More, and T. S. Munson, Benchmarking optimization software with COPS 3.0,
# Tech. Report ANL/MCS-TM-273, Argonne National Laboratory, 2004.
# D. J. Gunn and W. J. Thomas, Mass transport and chemical reaction in multifunctional catalyst
# systems, Chem. Eng. Sci., 20 (1965), pp. 89-100.
#
#   max  1 - x1(tf) - x2(tf)
#   s.t. x1' = u (10 x2 - x1)
#        x2' = u (x1 - 10 x2) - (1 - u) x2
#        x1(0) = x10,  x2(0) = x20
#        0 <= u <= 1
#
#   x1   : mole fraction of species A
#   x2   : mole fraction of species B
#   u    : fraction of catalyst 1, the control
#   tf   : final time
#   x10  : initial mole fraction of A
#   x20  : initial mole fraction of B

using BenchmarkTools

using ExaModels, ExaModelsCollocation
using MadNLP, MadNLPHSL
using MadNLPGPU, CUDA, CUDSS

const x10, x20 = 1.0, 0.0
const tf = 1.0

function catmix_model(; N = 10000, K = 4, backend = nothing)
    # Create CollocationExaCore
    core = CollocationExaCore(range(0.0, tf; length = N + 1), K; backend = backend)

    # Define start guess
    z0 = Array{Float64}(undef, 2, N, K + 1)
    z0[1, :, :] .= x10
    z0[2, :, :] .= x20

    # Create CollocationVariables
    @add_var_collocation(core, z, 1:2; start = z0) # z[v,i,k] = (x1, x2)
    @add_var_collocation(core, u; include_boundary = false, lvar = 0.0, uvar = 1.0, start = 0.0)

    # Create collocation constraints
    @add_con_collocation(core, coll_x1, z[1], # x1'
        u * (10 * z[2] - z[1])
    )
    @add_con_collocation(core, coll_x2, z[2], # x2'
        u * (z[1] - 10 * z[2]) - (1 - u) * z[2]
    )

    # Create continuity constraints
    @add_con_continuity(core, cont, z)

    # Create initial condition constraints
    @add_con(core, ic, z[v, 1, 0] - val for (v, val) in [(1, x10), (2, x20)])

    # Create objective function
    @add_obj(core, -1.0 + z[1, N, K] + z[2, N, K])

    return ExaModel(core)
end

# ----- Solve -----

# Solve with CPU
model_cpu = catmix_model()
result_cpu = @btime madnlp(model_cpu; tol = 1e-8, print_level = MadNLP.ERROR,
    kkt_system = MadNLP.SparseCondensedKKTSystem,
    equality_treatment = MadNLP.RelaxEquality,
    fixed_variable_treatment = MadNLP.RelaxBound,
    linear_solver = Ma57Solver,
)

# Solve with GPU
model_gpu = catmix_model(backend = CUDA.CUDABackend())
result_gpu = @btime madnlp(model_gpu; tol = 1e-8, print_level = MadNLP.ERROR,
    kkt_system = MadNLP.SparseCondensedKKTSystem,
    equality_treatment = MadNLP.RelaxEquality,
    fixed_variable_treatment = MadNLP.RelaxBound,
)

# ----- Display results -----

function report(label, model, result)
    zsol = Array(solution(result, model.z))
    println("$label status = $(result.status)")
    println("$label obj    = $(result.objective)")
    println("$label x1(tf) = $(zsol[1, end, end])")
    println("$label x2(tf) = $(zsol[2, end, end])")
end
report("cpu", model_cpu, result_cpu)
report("gpu", model_gpu, result_gpu)
