# Van der Pol Oscillator Optimal Control
#
#   min  z3(tf)
#   s.t. z1' = z2
#        z2' = theta z2 (1 - z1^2) - z1 + u + p2 cos(t)
#        z3' = z1^2 + z2^2 + u^2
#        z1(0) = 0,  z2(0) = p1,  z3(0) = 0
#        -1 <= p2 <= 1
#
#   z1    : position
#   z2    : velocity
#   z3    : accumulated cost
#   u     : forcing, the control
#   t     : time
#   tf    : final time
#   theta : damping coefficient, an ExaModels parameter
#   p1    : initial velocity, fixed by its bounds
#   p2    : amplitude of the cos(t) forcing, free

using BenchmarkTools

using ExaModels, ExaModelsCollocation
using MadNLP, MadNLPHSL
using MadNLPGPU, CUDA, CUDSS

const tf = 5.0
const p1 = 1.0

function vanderpol_model(; N = 20, K = 3, adaptive = false, backend = nothing)
    # Create CollocationExaCore
    core = CollocationExaCore(range(0.0, tf; length = N + 1), K; adaptive = adaptive, backend = backend)

    # Create CollocationVariables
    @add_var_collocation(core, z, 1:3) # z[v,i,k] = (z1, z2, z3)
    @add_var_collocation(core, u; include_boundary = false)

    # Create ExaModels variables and parameters
    @add_var(core, p, 1:2; lvar = [p1, -1.0], uvar = [p1, 1.0], start = [1.0, 0.0])
    @add_par(core, theta, [1.0])

    # Create collocation constraints
    @add_con_collocation(core, coll1, z[1], # z1'
        z[2]
    )
    @add_con_collocation(core, coll2, z[2], # z2'
        theta[1] * z[2] * (1 - z[1]^2) - z[1] + u + p[2] * cos(t)
    )
    @add_con_collocation(core, coll3, z[3], # z3'
        z[1]^2 + z[2]^2 + u^2
    )

    # Create continuity constraints
    @add_con_continuity(core, cont, z)

    # Create initial condition constraints
    @add_con(core, ic_zero, z[v, 1, 0] for v in [1, 3])
    @add_con(core, ic_par, z[2, 1, 0] - p[1])

    # Create objective function
    @add_obj(core, z[3, N, K])

    return ExaModel(core)
end

# ----- Solve -----

# Solve with CPU
model_cpu = vanderpol_model()
result_cpu = @btime madnlp(model_cpu; tol = 1e-6, print_level = MadNLP.ERROR,
    kkt_system = MadNLP.SparseCondensedKKTSystem,
    equality_treatment = MadNLP.RelaxEquality,
    fixed_variable_treatment = MadNLP.RelaxBound,
    linear_solver = Ma57Solver,
)

# Solve with GPU
model_gpu = vanderpol_model(backend = CUDA.CUDABackend())
result_gpu = @btime madnlp(model_gpu; tol = 1e-6, print_level = MadNLP.ERROR,
    kkt_system = MadNLP.SparseCondensedKKTSystem,
    equality_treatment = MadNLP.RelaxEquality,
    fixed_variable_treatment = MadNLP.RelaxBound,
)

# ----- Display results -----

function report(label, model, result)
    zsol = Array(solution(result, model.z))
    println("$label status = $(result.status)")
    println("$label obj    = $(result.objective)")
    println("$label z(tf)  = $(zsol[:, end, end])")
    println("$label p      = $(Array(solution(result, model.p)))")
end
report("cpu", model_cpu, result_cpu)
report("gpu", model_gpu, result_gpu)
