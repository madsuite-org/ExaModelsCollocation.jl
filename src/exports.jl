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
- `basis`: differential-state basis, [`StateForm()`](@ref) (default) or [`DerivativeForm()`](@ref)
- `polynomial`: interpolating polynomial, only [`Lagrange()`](@ref) (default) supported
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
        basis::AbstractBasis = ExaModelsDAE.StateForm(),
        polynomial::AbstractPolynomial = ExaModelsDAE.Lagrange(),
        roots::AbstractRoots = ExaModelsDAE.GaussRadau(),
        adaptive::Bool = false
    )::Tuple{ExaCore, DAEta}
    # Warnings for unsupported features
    # polynomial.jl
    polynomial isa ExaModelsDAE.Lagrange || error("Only Lagrange interpolation polynomials are supported currently.")

    # roots.jl: obtain K+1 interpolation points, taus = {tau0 = 0, ..., tauK}
    taus = _get_roots(roots, degree)

    # get A (ajk), b (bj) as constants; needs tau, so must follow _get_roots
    # basis.jl: get collocation and continuity weights
    weights = _get_weights(basis, polynomial, taus)

    # OrdinaryDiffEq.jl to adpatively foward solve for mesh


    # nodes.jl: create tij, hi info (for future AMR support)
    core, mesh = _create_mesh(core, tspan, init, nodes, taus)
    
    # parameters.jl: tau[j/k], t[i,j], h[i], theta[:] as ExaModels parameters (for future AMR support)
    # tauj/k, tij, hi are constants left as constants if adaptive = false
    core = _create_parameters(core)

    # variables.jl: z[v,i,k,c], y[v,i,k,c], u[v,i,k,c], p[:], (c=[Nc] for future multi-condition support)
    core, vars = _create_variables(core)

    # collocation.jl: main collocation equations
    core = _create_collocation(core)

    # continuity.jl: cross-interval continuity
    core = _create_continuity(core)

    # initialcons.jl: z(t0) = z0(y,u,p,theta)
    core = _create_initialcons(core)

    # algebraic.jl: g(z,y,u,p,theta,t) = 0
    core = _create_algebraic(core)

    # pathcons.jl: c(z,y,u,p,theta,t) \le 0
    core = _create_pathcons(core)

    # terminalcons.jl: hE(z,y,u,p,theta) = 0
    core = _create_termincalcons(core)

    # Display DAEta information
    display(dae)

    # Return ExaCore and DAEta
    return core, dae
end