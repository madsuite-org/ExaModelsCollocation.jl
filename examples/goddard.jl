# Goddard rocket, maximum ascent (Bryson & Ho; COPS 3.0 normalization)
#
#   max h(tf)  s.t.  h' = v
#                    v' = (T - D(h,v))/m - g(h)
#                    m' = -T/c
#                    D(h,v) = Dc v^2 exp(-hc (h - h0)/h0),  g(h) = g0 (h0/h)^2
#                    (h,v,m)(0) = (h0, 0, m0),  m(tf) = mf,  0 <= T <= Tmax

using ExaModels
using ExaModelsCollocation
using MadNLP

const h0, v0, m0 = 1.0, 0.0, 1.0
const g0, hc, vc = 1.0, 500.0, 620.0
const cv = 0.5 * sqrt(g0 * h0)
const Dc = 0.5 * vc * (m0 / g0)
const Tmax = 3.5 * g0 * m0
const mf = 0.6 * m0

function examodel_goddard(; tf0 = 0.2, N = 60, K = 3)
    # Create CollocationExaCore with unknown horizon
    core = CollocationExaCore(range(0.0, tf0; length = N + 1), K; unknown_horizon = true)

    # Define bounds and start guess
    lz, uz = zeros(3, N, K + 1), fill(Inf, 3, N, K + 1)
    lz[1, :, :] .= h0
    lz[3, :, :] .= mf
    uz[3, :, :] .= m0
    ts = [k == 0 ? core.nodes[i] : core.mesh.t[i,k] for i in 1:N, k in 0:K]
    z0 = Array{Float64}(undef, 3, N, K + 1)
    z0[1, :, :] .= h0
    z0[2, :, :] .= 0.0
    z0[3, :, :] .= m0 .+ (mf - m0) .* ts ./ tf0

    # Create CollocationVariables
    @add_var_collocation(core, z, 1:3; lvar = lz, uvar = uz, start = z0) # z[v,i,k] = (h, v, m)
    @add_var_collocation(core, u; include_boundary = false, lvar = 0.0, uvar = Tmax, start = Tmax)

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
    @add_con(core, tc, z[3, N, K] - mf for _ in 1:1)

    # Create objective function
    @add_obj(core, -z[1, N, K] for _ in 1:1)

    return ExaModel(core)
end

# ----- Solve -----

# Create CollocationExaModel
model = examodel_goddard()

# Solve
result = madnlp(model; tol = 1e-8)

zsol = solution(result, model.z)
println("status    = $(result.status)")
println("h(tf)     = $(zsol[1, end, end])")
println("tf        = $(only(solution(result, model.tscale)))")
println("m(tf)     = $(zsol[3, end, end])")
