# Exported functions for public API
"""
    add_dae(core, f, z0, tspan, init; kwargs...)

Transcribes differential-algebraic equations as algebraic ExaModels constraints.

inputs:
- `core`: an `ExaModels.ExaCore`
- `f`: differential equation `f(z, y, u, p, theta, t)`
- `z0`: initial condition `z0(y, u, p, theta)`
- `tspan`: time horizon `(t0, tf)`
- `init.p/init.u`: initial guess for the estimated parameter and control profile
- `init.theta`: mutable parameter value
- `init.z`/`init.y`: initial guess for differential and algebraic states

kwargs:
- `g`, `c`, `hE`: algebraic, path, and terminal constraints
- `u`: fixed control profile `u(t)`; given fixes the control, omitted makes it a decision variable
- `bounds`: variable bounds
- `nodes`: vector of interval boundary points
- `degree`: number of interpolating points per interval (degree of interpolating polynomial)
- `polynomial`: interpolating polynomial, only [`Lagrange()`](@ref) (default) supported
- `basis`: differential-state basis, [`StateForm()`](@ref) (default) or [`DerivativeForm()`](@ref)
- `roots`: collocation points, [`GaussRadau()`](@ref) (default), [`GaussLegendre()`](@ref),
  or [`GaussLobatto()`](@ref)

See `docs/api_design.md` for more details.
"""
function add_dae(
        core::ExaCore,
        f::Function,
        z0::Function,
        tspan::Union{Real, Tuple{Real,Real}},
        init::NamedTuple;
        g::Union{Nothing, Function} = nothing,
        c::Union{Nothing, Function} = nothing,
        hE::Union{Nothing, Function} = nothing,
        u::Union{Nothing, AbstractVector, Function} = nothing,
        bounds::NamedTuple = (;),
        nodes::Union{Nothing, AbstractVector} = nothing,
        degree::Integer = 4,
        polynomial::AbstractPolynomial = ExaModelsDAE.Lagrange(),
        basis::AbstractBasis = ExaModelsDAE.StateForm(),
        roots::AbstractRoots = ExaModelsDAE.GaussRadau(),
        adaptive::Bool = false
    )
    # Warnings for unsupported features
    # polynomial.jl
    polynomial isa ExaModelsDAE.Lagrange || error("Only Lagrange interpolation polynomials are supported currently.")

    # taus.jl: obtain K+1 interpolation points, taus = {tau0 = 0, ..., tauK}
    taus = _get_taus(roots, degree)

    # basis.jl: get collocation and continuity weights A (ajk), b (bj) as constants
    weights = _get_weights(basis, polynomial, taus)

    # initialize.jl: OrdinaryDiffEq.jl to adpatively foward solve for mesh
    # ...

    # mesh.jl: create tij, hi info (for future AMR support)
    core, mesh = _create_mesh(core, tspan, init, nodes, taus)

    # ...
    dae = DAEta(f, z0, g, c, hE, u, taus, mesh)
    # DAEta
        # Callback Functions
            # f, z0, g, c, hE, u
        # Constants
            # taus, weights
        # Dimensions
            # N, K, Nz, Np, ...
        # Variable/parameter handles
            # ...
    
    # parameters.jl: theta[:] as ExaModels parameters (+ t[i,j], h[i] for future AMR support)
    # tij, hi are constants left as constants if adaptive = false
    core = _create_parameters(core)

    # variables.jl: z[v,i,k,c], y[v,i,k,c], u[v,i,k,c], p[:], (c=[Nc] for future multi-condition support)
    core, vars = _create_variables(core)

    # collocation.jl: main collocation equations
    core = _create_collocation(core)

    # continuity.jl: cross-interval continuity
    core = _create_continuity(core)

    # initialcons.jl: z(t0) = z0(y,u,p,theta)
    core = _create_initialcons(core)

    # TODO algebraic.jl: g(z,y,u,p,theta,t) = 0
    core = _create_algebraic(core)

    # TODO pathcons.jl: c(z,y,u,p,theta,t) \le 0
    core = _create_pathcons(core)

    # TODO terminalcons.jl: hE(z,y,u,p,theta) = 0
    core = _create_termincalcons(core)

    # Return ExaCore and DAEta
    return core, dae
end