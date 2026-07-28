# The collocation model container.
#
# DAEta separates the two things a collocation discretization needs, and holds nothing twice:
#   mode : reference-element data in tau-space  (roots, basis, polynomial, weights)
#   mesh   : physical placement in t-space        (nodes, h, t)
# plus the handles registered by the add_*_collocation helpers. Each helper mutates it in
# place and returns it alongside the ExaCore, mirroring how ExaModels threads its core.
#
# N, K, nodes, roots, basis, polynomial, and weights are all derived and surface as
# properties, so they read flat from the outside without being stored twice.

"""
    VarBlock

Layout of one variable block created by [`add_var_collocation`](@ref), retrieved with
[`block`](@ref).

# Fields
- `var`    : the `ExaModels.Variable` handle the block was allocated as
- `dims`   : the user-declared dimensions
- `krange` : collocation index range, `0:K` with the element-left boundary node, else `1:K`
"""
struct VarBlock{V, D}
    var::V
    dims::D
    krange::UnitRange{Int}
end

"""
    CollocationMode

Reference-element discretization, independent of where the elements sit along `t`.

# Fields
- `roots`, `basis`, `polynomial` : the mode
- `weights` : [`BasisWeights`](@ref) `A`, `b`, `taus`
"""
struct CollocationMode{R <: AbstractRoots, B <: AbstractBasis, P <: AbstractPolynomial, W <: BasisWeights}
    roots::R
    basis::B
    polynomial::P
    weights::W
end

"""
    DAEta(nodes, K; roots = GaussRadau(), basis = StateForm(), polynomial = Lagrange())
    DAEta()

Collocation metadata threaded through the `add_*_collocation` helpers.

`nodes` are the `N+1` element boundaries (so `N = length(nodes) - 1`) and `K` is the degree
of the interpolating polynomial. Both are fixed once, here, and every helper reads them off
the `DAEta` rather than taking them again.

The no-argument form builds an empty container; attach a mesh with [`set_mesh!`](@ref)
before calling a helper that needs one.

# Fields
- `mode` : [`CollocationMode`](@ref) — `roots`, `basis`, `polynomial`, `weights`
- `mesh`   : [`CollocationMesh`](@ref) — `nodes`, `h`, `t`
- `vars`, `cons` : `NamedTuple`s of the handles given a name, so `dae.vars.z`
- `blocks` : `Vector` of every [`VarBlock`](@ref), named or not; look one up by handle with
  [`block`](@ref)

This mirrors how `ExaCore` keeps every variable in `core.var` and only the named ones in
`core.refs`. Named handles are forwarded to the container itself, so `dae.z` works.

Derived properties: `dae.N`, `dae.K`, `dae.nodes`, and the `mode` fields `dae.roots`,
`dae.basis`, `dae.polynomial`, `dae.weights`.
"""
mutable struct DAEta
    mode::Union{Nothing, CollocationMode}
    mesh::Union{Nothing, CollocationMesh}
    vars::NamedTuple
    cons::NamedTuple
    blocks::Vector{VarBlock}
end

DAEta() = DAEta(nothing, nothing, (;), (;), VarBlock[])

function DAEta(
        nodes::AbstractVector,
        K::Integer;
        roots::AbstractRoots = GaussRadau(),
        basis::AbstractBasis = StateForm(),
        polynomial::AbstractPolynomial = Lagrange(),
    )
    return set_mesh!(DAEta(), nodes, K; roots, basis, polynomial)
end

"""
    set_mesh!(dae, nodes, K; roots, basis, polynomial)

Attach the mode and mesh to `dae`, in place. Called by the [`DAEta`](@ref) two-argument
constructor; use it directly to fill in a `DAEta()`.
"""
function set_mesh!(
        dae::DAEta,
        nodes::AbstractVector,
        K::Integer;
        roots::AbstractRoots = GaussRadau(),
        basis::AbstractBasis = StateForm(),
        polynomial::AbstractPolynomial = Lagrange(),
    )
    length(nodes) >= 2 ||
        throw(ArgumentError("nodes needs at least 2 element boundaries, got $(length(nodes))"))
    K >= 1 || throw(ArgumentError("K must be at least 1, got $K"))

    # polynomial.jl: only Lagrange interpolation is implemented
    polynomial isa Lagrange ||
        throw(ArgumentError("Only Lagrange interpolation polynomials are supported currently."))

    # taus.jl / basis.jl: GaussLobatto puts a collocation point on tau = 0, which collides
    # with the tau0 = 0 anchor StateForm prepends -- repeated nodes give NaN barycentric
    # weights. DerivativeForm needs no anchor, so switch to it.
    if roots isa GaussLobatto && basis isa StateForm
        @warn "GaussLobatto: automatically switching to DerivativeForm since not possible with StateForm."
        basis = DerivativeForm()
    end

    bnds = collect(float.(nodes))
    issorted(bnds) || throw(ArgumentError("nodes must be nondecreasing along t"))

    # taus.jl: the K collocation points; basis.jl: collocation/continuity weights A, b
    taus = _get_taus(roots, K)
    weights = _get_weights(polynomial, basis, taus)

    setfield!(dae, :mode, CollocationMode(roots, basis, polynomial, weights))
    setfield!(dae, :mesh, _get_mesh(bnds, taus))
    return dae
end

# ---------- internal accessors ----------
_mesh(dae::DAEta) = getfield(dae, :mesh)
_mode(dae::DAEta) = getfield(dae, :mode)
_weights(dae::DAEta) = _mode(dae).weights
_degree(dae::DAEta) = length(_weights(dae).taus)

# Which residual form the add_con_* helpers build (Biegler 10.7/10.14a vs 10.8/10.15a)
_isstateform(dae::DAEta) = _mode(dae).basis isa StateForm

# Number of elements, N = length(nodes) - 1
function _nelements(dae::DAEta)
    mesh = _mesh(dae)
    mesh === nothing ? 0 : length(mesh.h)
end

# Register a handle under `name` in one of the NamedTuple fields
function _register!(dae::DAEta, field::Symbol, name::Symbol, value)
    setfield!(dae, field, merge(getfield(dae, field), NamedTuple{(name,)}((value,))))
    return dae
end

# Guard for helpers that cannot run on an empty DAEta
function _require_mesh(dae::DAEta, who::Symbol)
    _mesh(dae) === nothing && error(
        "$who requires a mesh: build the container with `DAEta(nodes, K)` " *
        "or attach one with `set_mesh!(dae, nodes, K)`."
    )
    return nothing
end

"""
    block(dae, var) -> VarBlock

The [`VarBlock`](@ref) layout of a handle returned by [`add_var_collocation`](@ref). Looked
up by handle identity, the way ExaModels passes handles around, so it works whether or not
the block was given a name.

```julia
julia> block(dae, u).krange
1:3
```
"""
function block(dae::DAEta, var)
    for blk in getfield(dae, :blocks)
        blk.var === var && return blk
    end
    throw(ArgumentError("that variable was not created by add_var_collocation on this DAEta"))
end

# Layout of the block a variable handle belongs to, with the checks the add_con_* helpers need.
function _block_layout(dae::DAEta, z, who::Symbol)
    _require_mesh(dae, who)
    blk = try
        block(dae, z)
    catch
        throw(ArgumentError(
            "$who: that variable was not created by add_var_collocation on this DAEta"
        ))
    end

    K = _degree(dae)
    blk.krange == 0:K || throw(ArgumentError(
        "$who: that block was created with include_boundary = false, so it has no k = 0 " *
        "boundary node; the collocation stencil needs one."
    ))

    nlead = length(blk.dims)
    1 <= nlead <= 2 || throw(ArgumentError(
        "$who: that block declares $nlead leading dimensions; the add_con_* helpers handle " *
        "one or two."
    ))
    return nlead, K
end

# ---------- property forwarding ----------
# Derived quantities and registered handles surface flat: dae.N, dae.K, dae.nodes, dae.z, ...
const _SCHEME_FIELDS = (:roots, :basis, :polynomial, :weights)

function Base.getproperty(dae::DAEta, name::Symbol)
    hasfield(DAEta, name) && return getfield(dae, name)

    name === :N && return _nelements(dae)
    if name === :K || name === :nodes || name in _SCHEME_FIELDS
        _require_mesh(dae, :getproperty)
        name === :K && return _degree(dae)
        name === :nodes && return _mesh(dae).nodes
        return getfield(_mode(dae), name)
    end

    vars = getfield(dae, :vars)
    haskey(vars, name) && return getfield(vars, name)
    cons = getfield(dae, :cons)
    haskey(cons, name) && return getfield(cons, name)

    throw(ArgumentError("DAEta has no field or registered handle `$name`"))
end

# Display name of a block: whatever it was registered under, or "_" if it was anonymous
function _block_name(dae::DAEta, blk::VarBlock)
    vars = getfield(dae, :vars)
    for nm in keys(vars)
        getfield(vars, nm) === blk.var && return nm
    end
    return :_
end

Base.propertynames(dae::DAEta) = (
    fieldnames(DAEta)...,
    :N, :K, :nodes, _SCHEME_FIELDS...,
    keys(getfield(dae, :vars))...,
    keys(getfield(dae, :cons))...,
)

function Base.show(io::IO, dae::DAEta)
    if _mesh(dae) === nothing
        print(io, "DAEta (no mesh)")
        return
    end
    cons = keys(getfield(dae, :cons))
    mode = _mode(dae)
    nodes = _mesh(dae).nodes
    vars = join(
        ["$(_block_name(dae, b))[$(join(b.dims, ", ")), i, k=$(b.krange)]"
         for b in getfield(dae, :blocks)],
        ", ",
    )
    print(
        io,
        """
        DAEta

          mesh    N = $(_nelements(dae)) elements, K = $(_degree(dae)) degree
          horizon [$(first(nodes)), $(last(nodes))]
          mode  $(mode.basis), $(mode.polynomial), $(mode.roots)
          vars    $(isempty(vars) ? "-" : vars)
          cons    $(isempty(cons) ? "-" : join(cons, ", "))
        """,
    )
end
