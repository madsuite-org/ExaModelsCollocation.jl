# Discretization metadata
"""
    DAECallbacks{Tf,Tz0,Tg,Tc,ThE,Tu}

The user problem functions for transcription.

# Fields
- `f`  : right-hand side function `dz/dt = f(z,y,u,p,theta,t)`
- `z0` : initial condition `z0(y,u,p,theta)`
- `g`  : algebraic constraint `g(z,y,u,p,theta,t) = 0`
- `c`  : bounds/path inequality `c(z,y,u,p,theta,t) <= 0`
- `hE` : terminal equality `hE(z,y,u,p,theta) = 0`
- `u`  : fixed control profile `u(t)`
"""
struct DAECallbacks{Tf,Tz0,Tg,Tc,ThE,Tu}
    f::Tf
    z0::Tz0
    g::Tg
    c::Tc
    hE::ThE
    u::Tu
end

"""
    DAEDims

Problem dimensions of the discretized DAE system.

# Fields
- `N`  : number of intervals
- `K`  : degree of interpolating polynomial (K+1 interpolation points per interval)
- `nz` : differential state variables
- `ny` : algebraic state variables
- `nu` : control variables
- `np` : free decision parameters
- `ntheta` : mutable fixed parameters
"""
struct DAEDims
    N::Int
    K::Int
    nz::Int
    ny::Int
    nu::Int
    np::Int
    ntheta::Int
end

"""
    DAEta

DAE metadata returned by [`add_dae`](@ref).
The user can write constraints and objectives separately using DAEta.

# Fields
- `meta`      : NamedTuple of user inputs
                `tspan`,
                `init`,
                `bounds`,
                `nodes`,
                `degree`,
                `polynomial`,
                `basis`,
                `roots`,
                `adaptive`
- `callbacks` : [`DAECallbacks`](@ref) of the user problem functions `f, z0, g, c, hE, u`
- `weights`   : [`BasisWeights`](@ref) `A`, `b`, `taus`
- `mesh`      : [`CollocationMesh`](@ref): `t[i,j]`, `h[i]`
- `dims`      : [`DAEDims`](@ref) problem dimensions0
- `vars`      : NamedTuple of ExaModels variable/parameter handles:
                `z` differential collocation states,
                `y` algebraic collocation states,
                `u` controls,
                `p` free decision parameters,
                `theta` mutable fixed parameters,
                `zf` terminal states
- `cons`      : NamedTuple of constraint handles:
                `collocation`,
                `continuity`,
                `initial`,
                `algebraic`,
                `path`,
                `terminal`
"""
struct DAEta{MT,CB<:DAECallbacks,T,MESH,V,C}
    meta::MT
    callbacks::CB
    weights::BasisWeights{T}
    mesh::MESH
    dims::DAEDims
    vars::V
    cons::C
end
# NOTE: `u` is an examodels parameter if fixed, examodels variable if decision variable
# NOTE: zf automatically routes to z[endpoint] without actually creating a zf variable (for Radau and Lobatto roots)

function DAEta(dae::DAEta; kwargs...)
    kw = NamedTuple(kwargs)
    DAEta((get(kw, name, getfield(dae, name)) for name in fieldnames(DAEta))...)
end

# TODO rewrite
# Alias handle for the terminal state when the endpoint is a collocation node
# (Radau, Lobatto). Forwards index access to the final node z[v, N, K, c] so that
# vars.zf exposes the same zf[v] / zf[v, c] interface as the dedicated GaussLegendre
# variable, and callers need not branch on roots.
struct TerminalState{Z}
    z::Z
    N::Int
    K::Int
end
Base.getindex(zf::TerminalState, v, c = 1) = zf.z[v, zf.N, zf.K, c]

function Base.show(io::IO, dae::DAEta)
    d = dae.dims
    m = dae.meta
    print(
        io,
        """
        DAEta

          states     nz = $(d.nz),  ny = $(d.ny),  controls nu = $(d.nu)
          parameters np = $(d.np),  ntheta = $(d.ntheta)
          mesh       N = $(d.N) intervals, K = $(d.K) degree
          horizon    tspan = $(m.tspan)
          method     $(m.basis), $(m.polynomial), $(m.roots)
        """,
    )
end

function Base.getproperty(dae::DAEta, name::Symbol)
    if hasfield(DAEta, name)
        getfield(dae, name)
    elseif hasfield(typeof(getfield(dae, :vars)), name)
        getfield(getfield(dae, :vars), name)
    elseif hasfield(typeof(getfield(dae, :dims)), name)
        getfield(getfield(dae, :dims), name)
    else
        getfield(dae, name)
    end
end

# Surface the forwarded handle/dimension names to tab-completion and introspection.
Base.propertynames(dae::DAEta) = (
    fieldnames(DAEta)..., propertynames(getfield(dae, :vars))..., fieldnames(DAEDims)...
)
