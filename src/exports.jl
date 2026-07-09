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
- `ufixed`: fixed control profile
- `bounds`: variable bounds
- `nodes`: vector of interval boundary points
- `degree`: number of interpolating points per interval (degree of interpolating polynomial)
- `basis`: differential-state basis, [`StateForm()`](@ref) (default) or [`DerivativeForm()`](@ref)
- `polynomial`: interpolating polynomial, only [`Lagrange()`](@ref) (default) supported currently
- `roots`: collocation points, [`GaussRadau()`](@ref) (default), [`GaussLegendre()`](@ref),
  or [`GaussLobatto()`](@ref)

See `docs/api_design.md` for more details.
"""
function add_dae(
      core::ExaCore,
      f,
      z0,
      tspan,
      init;
      g = nothing,
      c = nothing,
      hE = nothing,
      ufixed = nothing,
      bounds = (;),
      nodes = nothing,
      degree = 4,
      basis = ExaModelsDAE.StateForm(),
      polynomial = ExaModelsDAE.Lagrange(),
      roots = ExaModelsDAE.GaussRadau()
  )
  # Warnings for unsupported features
  polynomial isa ExaModelsDAE.Lagrange() || error("Only Lagrange interpolation polynomials are supported.")

  

  #
  return core, dae
end