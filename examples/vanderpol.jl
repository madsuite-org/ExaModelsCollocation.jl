# Van der Pol optimal control
#
#   min z3(tf)  s.t.  z1' = z2
#                     z2' = th1 z2 (1 - z1^2) - z1 + u + p2 cos(t)
#                     z3' = z1^2 + z2^2 + u^2
#                     z(t0) = (0, p1, 0)

using ExaModels
using ExaModelsCollocation
using MadNLP

function examodel_van_der_pol(; tf = 5.0, N = 20, K = 3, p1 = 1.0, adaptive = false)
    # Create CollocationExaCore
    core = CollocationExaCore(range(0.0, tf; length = N + 1), K; adaptive)

    # Create CollocationVariables
    @add_var_collocation(core, z, 1:3)                       # z[v,i,k] w/ k = 0,...,K
    @add_var_collocation(core, u; include_boundary = false)  # k = 1,...,K

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

# Create CollocationExaModel
model = examodel_van_der_pol()

# Solve
result = madnlp(model)

zsol = solution(result, model.z)
println("status    = $(result.status)")
println("objective = $(result.objective)")
println("z(tf)     = $(zsol[:, end, end])")
println("p         = $(solution(result, model.p))")
