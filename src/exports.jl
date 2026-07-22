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
- `u`: fixed control profile `u(t)`
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
    # ---------- Errors for unsupported features ----------
    # polynomial.jl
    polynomial isa ExaModelsDAE.Lagrange || error("Only Lagrange interpolation polynomials are supported currently.")

    # ---------- Create DAEta things ----------
    # DAEta field 1. meta: NamedTuple of user inputs
    meta = (; tspan, init, bounds, nodes, degree, polynomial, basis, roots, adaptive)
    
    # DAEta field 2. callbacks: DAECallbacks of user problem funxtions
    callbacks = DAECallbacks(f, z0, g, c, hE, u)

    # DAEta field 3. weights: BasisWeights based on polynomial and basis
    # taus.jl: K+1 interpolation points, taus = {tau0 = 0, ..., tauK}
    taus = _get_taus(roots, degree)
    # basis.jl: collocation and continuity weights A (ajk), b (bj) and taus
    weights = _get_weights(polynomial, basis, taus)

    # DAEta field 4. mesh: CollocationMesh
    # initialize.jl: OrdinaryDiffEq.jl forward solve for the mesh
    init_full = _get_init_full(meta, callbacks, weights)
    # mesh.jl: tij, hi info
    mesh = _get_mesh(meta, init_full)

    # DAEta field 5. dims: DAEDims
    dims = _get_dims(meta, mesh)

    # DAEta initialize
    dae = DAEta(meta, callbacks, weights, mesh, dims, (;), (;))

    # ---------- Create ExaModels things ----------
    # parameters.jl: theta[:] as ExaModels parameters (+ t[i,j], h[i] for future AMR support)
    core, dae = _create_parameters(core, dae)

    # variables.jl: z[v,i,k,c], y[v,i,k,c], u[v,i,k,c], p[:] (c=[Nc] for future multi-condition support)
    core, dae = _create_variables(core, dae)

    # collocation.jl: main collocation equations
    core, dae = _create_collocation(core, dae)

    # continuity.jl: cross-interval continuity
    core, dae = _create_continuity(core, dae)

    # TODO initial.jl: z(t0) = z0(y,u,p,theta)
    core, dae = _create_initial(core, dae)

    # TODO algebraic.jl: g(z,y,u,p,theta,t) = 0
    core, dae = _create_algebraic(core, dae)

    # TODO path.jl: c(z,y,u,p,theta,t) <= 0
    core, dae = _create_path(core, dae)

    # TODO terminal.jl: hE(z,y,u,p,theta) = 0
    core, dae = _create_terminal(core, dae)

    return core, dae
end
