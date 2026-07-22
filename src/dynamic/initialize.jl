# Deriving initial guesses for all decision variables
function _get_init_full(
        meta::NamedTuple,
        callbacks::DAECallbacks,
        weights::BasisWeights
    )
    # Unpack structs
    (; tspan, init, nodes, degree, polynomial, basis, roots, adaptive) = meta
    (; f, z0, g, c, hE, u) = callbacks
    (; taus) = weights

    # Normalize scalar tspan to a (t0, tf) tuple
    tspan = tspan isa Real ? (zero(tspan), tspan) : tspan

    # Identify path
    (; z, y, u, p, theta) = init
        # TODO init determines build paths

    # Identify path
    
    nodes == nothing ? path = 1 : tstops = union(nodes, tspan)

    # ---------- Path 1.  ----------
    # 

    # ---------- Path 2. (OrdinaryDiffEq.jl) ----------
    # 

    # ---------- Path 3. (OrdinaryDiffEq.jl) hi ----------
    # 

    # TODO if meta.nodes not provided, adopt the time steps taken by the forward solve
    # TODO if nodes provided, enforce tstops = nodes, only choose N = length(nodes)
    # TODO only instantiate OrdinaryDiffEq.jl if the path envokes it
    return init_full
end

# Path 1. 
function _initialize(::)
end

