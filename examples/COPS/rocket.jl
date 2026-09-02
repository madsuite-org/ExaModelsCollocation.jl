# COPS 3.0 Problem 10: Goddard Rocket
#
# E. D. Dolan, J. J. More, and T. S. Munson, Benchmarking optimization software with COPS 3.0,
# Tech. Report ANL/MCS-TM-273, Argonne National Laboratory, 2004.
# A. E. Bryson, Dynamic Optimization, Addison-Wesley, 1999.
#
#   max  h(tf)
#   s.t. h' = v
#        v' = (T - D(h,v))/m - g(h)
#        m' = -T/c
#        D(h,v) = Dc v^2 exp(-hc (h - h0)/h0)
#        g(h)   = g0 (h0/h)^2
#        h(0) = h0,  v(0) = v0,  m(0) = m0,  m(tf) = mf
#        h >= h0,  v >= 0,  mf <= m <= m0,  0 <= T <= Tmax
#
#   h    : altitude
#   v    : velocity
#   m    : mass
#   T    : thrust, the control
#   tf   : final time, free
#   D    : drag
#   g    : gravity
#   c    : exhaust velocity
#   Dc   : drag constant
#   hc   : drag scale height
#   vc   : drag velocity
#   g0   : gravity at the surface
#   h0   : initial altitude
#   v0   : initial velocity
#   m0   : initial mass
#   mf   : final mass
#   Tmax : maximum thrust

using BenchmarkTools

using ExaModels, ExaModelsCollocation
using MadNLP, MadNLPHSL
using MadNLPGPU, CUDA, CUDSS

const h0, v0, m0 = 1.0, 0.0, 1.0
const g0, hc, vc = 1.0, 500.0, 620.0
const cv = 0.5 * sqrt(g0 * h0)
const Dc = 0.5 * vc * (m0 / g0)
const Tmax = 3.5 * g0 * m0
const mf = 0.6 * m0
const tf0 = 1.0

function rocket_model(; N = 10000, K = 4, backend = nothing)
    # Create CollocationExaCore with unknown horizon
    core = CollocationExaCore(range(0.0, tf0; length = N + 1), K; 
        unknown_horizon = true,
        backend = backend
    )

    # Define bounds and start guess
    lz, uz = zeros(3, N, K + 1), fill(Inf, 3, N, K + 1)
    lz[1, :, :] .= h0
    lz[3, :, :] .= mf
    uz[3, :, :] .= m0
    ts = [k == 0 ? core.nodes[i] : core.mesh.t[i,k] for i in 1:N, k in 0:K]
    s = ts ./ tf0
    z0 = Array{Float64}(undef, 3, N, K + 1)
    z0[1, :, :] .= h0
    z0[2, :, :] .= s .* (1 .- s)
    z0[3, :, :] .= m0 .+ (mf - m0) .* s

    # Create CollocationVariables
    @add_var_collocation(core, z, 1:3; lvar = lz, uvar = uz, start = z0) # z[v,i,k] = (h, v, m)
    @add_var_collocation(core, u; include_boundary = false, lvar = 0.0, uvar = Tmax, start = Tmax / 2)

    # Create collocation constraints
    @add_con_collocation(core, coll_h, z[1], # h'
        z[2]
    )
    @add_con_collocation(core, coll_v, z[2], # v'
        (u - Dc * z[2]^2 * exp(-hc * (z[1] - h0) / h0)) / z[3] - g0 * (h0 / z[1])^2
    )
    @add_con_collocation(core, coll_m, z[3], # m'
        -u / cv
    )

    # Create continuity constraints
    @add_con_continuity(core, cont, z)

    # Create initial and terminal condition constraints
    @add_con(core, ic, z[v, 1, 0] - val for (v, val) in [(1, h0), (2, v0), (3, m0)])
    @add_con(core, tc, z[3, N, K] - mf)

    # Create objective function
    @add_obj(core, -z[1, N, K])

    return ExaModel(core)
end

# ----- Solve -----

# Solve with CPU
model_cpu = rocket_model()
result_cpu = @btime madnlp(model_cpu; tol = 1e-8, print_level = MadNLP.ERROR,
    kkt_system = MadNLP.SparseCondensedKKTSystem,
    equality_treatment = MadNLP.RelaxEquality,
    fixed_variable_treatment = MadNLP.RelaxBound,
    linear_solver = Ma57Solver,
)

# Solve with GPU
model_gpu = rocket_model(backend = CUDA.CUDABackend())
result_gpu = @btime madnlp(model_gpu; tol = 1e-8, print_level = MadNLP.ERROR,
    kkt_system = MadNLP.SparseCondensedKKTSystem,
    equality_treatment = MadNLP.RelaxEquality,
    fixed_variable_treatment = MadNLP.RelaxBound,
)

# ----- Display results -----

function report(label, model, result)
    zsol = Array(solution(result, model.z))
    println("$label status = $(result.status)")
    println("$label h(tf)  = $(zsol[1, end, end])")
    println("$label tf     = $(only(Array(solution(result, model.tscale))))")
    println("$label m(tf)  = $(zsol[3, end, end])")
end
report("cpu", model_cpu, result_cpu)
report("gpu", model_gpu, result_gpu)