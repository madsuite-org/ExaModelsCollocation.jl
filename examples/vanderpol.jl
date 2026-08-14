# Van der Pol optimal control, one collocation block per state.
#
#   min z3(tf)  s.t.  z1' = z2
#                     z2' = th1 z2 (1 - z1^2) - z1 + u + p2 cos(t)
#                     z3' = z1^2 + z2^2 + u^2
#                     z(t0) = (0, p1, 0)
#
# Each state is a block with no declared dimensions, so it is written bare and indexes as
# z1[i,k] where a full index is wanted. Nothing in these right-hand sides varies with anything
# but the mesh, so none of them needs an iterator: the helper crosses in i = 1,…,N and
# k = 1,…,K itself, and binds t at each of those points.
#
# The mesh is adaptive, so set_nodes! can move it between solves. That changes nothing above:
# t reads the same either way.

module VanDerPol

using ExaModels
using ExaModelsCollocation

# p1 is pinned by default so the optimum is not the rest solution
function build(; tf = 5.0, N = 20, K = 3, p1 = 1.0, adaptive = true)
    # Create CollocationExaCore
    core = CollocationExaCore(range(0.0, tf; length = N + 1), K; adaptive)

    # Create CollocationVariables
    @add_var_collocation(core, z1)                           # z1[i,k]
    @add_var_collocation(core, z2)                           # z2[i,k]
    @add_var_collocation(core, z3)                           # z3[i,k]
    @add_var_collocation(core, u; include_boundary = false)  # k = 1,...,K

    # Create standard ExaModels variables and parameters
    @add_var(core, p, 1:2; lvar = [p1, -1.0], uvar = [p1, 1.0], start = [1.0, 0.0])
    @add_par(core, theta, [1.0])

    # Create collocation constraints, one per state, each naming the slot it is for
    @add_con_collocation(core, coll1, z1[], z2)
    @add_con_collocation(core, coll2, z2[],
        theta[1] * z2 * (1 - z1^2) - z1 + u + p[2] * cos(t)
    )
    @add_con_collocation(core, coll3, z3[], z1^2 + z2^2 + u^2)

    # Create continuity constraints
    # NOTE: one per CollocationVariable. Declared as z[1:3,i,k] instead, this would be one call
    @add_con_continuity(core, cont1, z1)
    @add_con_continuity(core, cont2, z2)
    @add_con_continuity(core, cont3, z3)

    # Create initial condition constraints
    @add_con(core, ic1, z1[1, 0] for _ in 1:1)
    @add_con(core, ic2, z2[1, 0] - p[1] for _ in 1:1)
    @add_con(core, ic3, z3[1, 0] for _ in 1:1)

    # Create objective function
    @add_obj(core, z3[N, K] for _ in 1:1)

    return ExaModel(core)
end

end  # module VanDerPol
