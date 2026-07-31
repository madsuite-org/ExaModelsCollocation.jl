# The discretization rides in the core's `tag`, so a collocation core IS an ExaCore: every
# ExaModels.add_* works on it unforwarded, and ExaModel(core) carries the tag over, which is
# what keeps the mesh reachable after a solve. Same construction as ExaModels' own two-stage
# extension (two_stage.jl).
#
#   mode : tau-space  (roots, basis, polynomial, weights)
#   mesh : t-space    (nodes, h, t, parameter handles)
#
# blocks and residuals grow in place, as two_stage.jl grows var_scen, so the tag's type is
# fixed once. ExaCore does the same with x0 and lvar.

"""
    Residual

The right-hand side one [`add_con_collocation`](@ref) call was given, recorded so
[`add_con_continuity`](@ref) knows which slots were collocated and, under `DerivativeForm`,
integrates the same `f`.

# Fields
- `var`   : the variable it collocates
- `rows`  : the iterator rows it was built over, `(z's indices…, …, i, k, t)`
- `f`     : the right-hand side, evaluated on one of those rows
- `slots` : the distinct slots of `var` those rows cover
"""
struct Residual{V, R, F, S}
    var::V
    rows::R
    f::F
    slots::S
end

"""
    CollocationMode

Reference-interval discretization, independent of where the intervals sit along `t`.

# Fields
- `roots`, `basis`, `polynomial` : the mode
- `weights` : `A`, `b`, `taus`
"""
struct CollocationMode{R <: AbstractRoots, B <: AbstractBasis, P <: AbstractPolynomial, W <: BasisWeights}
    roots::R
    basis::B
    polynomial::P
    weights::W
end

"""
    CollocationTag

The discretization a [`CollocationExaCore`](@ref) carries in its `tag`. Build one with
[`Collocation`](@ref).

# Fields
- `mode` : [`CollocationMode`](@ref)
- `mesh` : [`CollocationMesh`](@ref)
- `blocks` : every [`CollocationVariable`](@ref) allocated on the core
- `residuals` : every [`Residual`](@ref) recorded by [`add_con_collocation`](@ref)
"""
struct CollocationTag{MO <: CollocationMode, ME <: CollocationMesh} <: ExaModels.AbstractExaModelTag
    mode::MO
    mesh::ME
    blocks::Vector{Any}
    residuals::Vector{Residual}
end

"""
    Collocation(nodes, K; roots = GaussRadau(), basis = StateForm(), polynomial = Lagrange())

The discretization, as an `ExaCore` tag. `nodes` are the `N+1` interval boundaries and `K`
the degree of the interpolating polynomial.

[`CollocationExaCore`](@ref) is the usual way in and takes these same keywords. Pass this
directly to build the core by hand, or to attach a mesh to one that already holds variables:

```julia
julia> core = ExaCore(concrete = Val(true); tag = Collocation(nodes, 3));

julia> core = ExaCore(core; tag = Collocation(nodes, 3));   # keeps what core already holds
```
"""
function Collocation(
        nodes::AbstractVector,
        K::Integer;
        roots::AbstractRoots = GaussRadau(),
        basis::AbstractBasis = StateForm(),
        polynomial::AbstractPolynomial = Lagrange(),
    )
    length(nodes) >= 2 ||
        throw(ArgumentError("nodes needs at least 2 interval boundaries, got $(length(nodes))"))
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
    mode = CollocationMode(roots, basis, polynomial, _get_weights(polynomial, basis, taus))
    return CollocationTag(mode, _get_mesh(bnds, taus), [], Residual[])
end

"""
    CollocationExaCore{T,VT,B}

An `ExaCore` whose `tag` is a [`CollocationTag`](@ref). Every `ExaModels.add_*` works on it
unchanged, so a model mixes plain ExaModels calls with collocation ones on one object.

    CollocationExaCore([array_eltype::Type,] nodes, K; roots = GaussRadau(),
                       basis = StateForm(), polynomial = Lagrange(), adaptive = false,
                       backend = nothing, minimize = true, name = :Generic)

`nodes` are the `N+1` interval boundaries and `K` the degree of the interpolating polynomial.

## Keyword Arguments
- `roots` : collocation family, `GaussRadau()`, `GaussLegendre()`, or `GaussLobatto()`
- `basis` : differential-state representation, `StateForm()` or `DerivativeForm()`
- `polynomial` : interpolating polynomial, `Lagrange()` only, so it is left at the default
- `adaptive` : allocate `h` and `t` as parameter blocks, so the mesh moves between solves
  without rebuilding. Costs `N·K + N` parameters and traces `h` symbolically.
- `backend`, `minimize`, `name` : passed on to `ExaCore`

`roots`, `basis` and `polynomial` are [`Collocation`](@ref)'s, and this is sugar over
`ExaCore(T; tag = Collocation(nodes, K; …))`. `adaptive` is not: it allocates parameter
blocks, so it needs a core and is only available here.

## Properties
`core.mode`, `core.mesh`, `core.blocks`, `core.residuals`, and derived `core.N`, `core.K`,
`core.nodes`, `core.adaptive`, `core.roots`, `core.basis`, `core.polynomial`, `core.weights`.
All of them read the same way off the [`CollocationExaModel`](@ref) the core builds.

## Example
```julia
julia> core = CollocationExaCore(range(0.0, 5.0; length = 21), 3);

julia> core.N, core.K
(20, 3)
```
"""
const CollocationExaCore{T, VT, B} = ExaCore{T, VT, B, <:CollocationTag}

"""
    CollocationExaModel{T,VT,E,V,P,O,C,R}

The `ExaModel` a [`CollocationExaCore`](@ref) builds. It carries the same tag, so `model.mesh`
and `model.z` read exactly as they do on the core.
"""
const CollocationExaModel{T, VT, E, V, P, O, C, R} = ExaModels.ExaModel{T, VT, E, V, P, O, C, <:CollocationTag, R}

CollocationExaCore(nodes::AbstractVector, K::Integer; backend = nothing, kwargs...) =
    CollocationExaCore(ExaModels.default_T(backend), nodes, K; backend, kwargs...)

function CollocationExaCore(
        ::Type{T},
        nodes::AbstractVector,
        K::Integer;
        roots::AbstractRoots = GaussRadau(),
        basis::AbstractBasis = StateForm(),
        polynomial::AbstractPolynomial = Lagrange(),
        adaptive::Bool = false,
        kwargs...,
    ) where {T <: AbstractFloat}
    # A LegacyExaCore would match neither the alias nor its getproperty methods.
    haskey(kwargs, :concrete) && throw(ArgumentError(
        "CollocationExaCore: `concrete` is always Val(true); the mutable LegacyExaCore " *
        "cannot carry a collocation tag."
    ))

    tag = Collocation(nodes, K; roots, basis, polynomial)
    core = ExaCore(T; tag = tag, concrete = Val(true), kwargs...)
    return adaptive ? _make_adaptive(core) : core
end

# h and t as parameter blocks, which is what lets a residual index them with a traced index
# and lets set_nodes! move the mesh without rebuilding. The numeric arrays stay as they are.
function _make_adaptive(core)
    tag, mesh = _tag(core), _mesh(core)
    core, hp = ExaModels.add_par(core, 1:length(mesh.h); value = mesh.h)
    core, tp = ExaModels.add_par(core, 1:size(mesh.t, 1), 1:size(mesh.t, 2); value = mesh.t)
    return ExaCore(
        core;
        tag = CollocationTag(
            tag.mode, _with_parameters(mesh, hp, tp), tag.blocks, tag.residuals,
        ),
    )
end

# ---------- internal accessors ----------
_tag(c) = getfield(c, :tag)
_mesh(c) = _tag(c).mesh
_mode(c) = _tag(c).mode
_weights(c) = _mode(c).weights
_degree(c) = length(_weights(c).taus)

# Number of intervals, N = length(nodes) - 1
_nintervals(c) = length(_mesh(c).h)

# Which residual form the add_con_* helpers build (Biegler 10.7/10.14a vs 10.8/10.15a)
_isstateform(c) = _mode(c).basis isa StateForm

# Whether h and t are parameter handles rather than numbers. The add_con_* helpers branch on
# this the way they branch on _isstateform: it changes where the interval width comes from,
# not which equation is built.
_isadaptive(c) = _mesh(c).hpar !== nothing

# Mesh entries a caller's iterator row ends in: (i, k, t) numerically, (i, k) adaptively,
# since t is then a parameter the right-hand side indexes rather than data in the row.
_nmesh(c) = _isadaptive(c) ? 2 : 3

"""
    set_nodes!(core_or_model, nodes)

Moves the mesh to new interval boundaries, on a core or model built with `adaptive = true`.
Recomputes `h` and `t` and writes both through, so the model is re-solved without rebuilding.

`nodes` must have the same length it was built with: redistributing a fixed number of
intervals needs no rebuild, adding one changes the variable count and does.
"""
function set_nodes!(c, nodes::AbstractVector)
    mesh = _mesh(c)
    mesh.hpar === nothing && throw(ArgumentError(
        "set_nodes!: this mesh is not adaptive; build it with `adaptive = true`"
    ))
    length(nodes) == length(mesh.nodes) || throw(DimensionMismatch(
        "set_nodes!: expected $(length(mesh.nodes)) boundaries, got $(length(nodes)); " *
        "changing the number of intervals needs a rebuild"
    ))
    issorted(nodes) || throw(ArgumentError("set_nodes!: nodes must be nondecreasing along t"))

    taus = _weights(c).taus
    copyto!(mesh.nodes, nodes)
    mesh.h .= diff(mesh.nodes)
    for i in axes(mesh.t, 1), j in axes(mesh.t, 2)
        mesh.t[i, j] = mesh.nodes[i] + mesh.h[i] * taus[j]
    end

    _set_mesh_parameter!(c, mesh.hpar, mesh.h)
    _set_mesh_parameter!(c, mesh.tpar, mesh.t)
    return nothing
end

_set_mesh_parameter!(c::CollocationExaCore, p, values) =
    ExaModels.set_parameter!(c, p, values)
_set_mesh_parameter!(m::CollocationExaModel, p, values) =
    ExaModels.set_value!(m, p, values)

_addblock(c::CollocationExaCore, z::CollocationVariable) = (push!(_tag(c).blocks, z); c)
_addresidual(c::CollocationExaCore, res::Residual) = (push!(_tag(c).residuals, res); c)

# The residuals recorded for a variable, in the order they were added
_residuals(c::CollocationExaCore, var) = [r for r in _tag(c).residuals if r.var === var]

# Swap the handle a helper wrapped back into the core, so `core.z` and `model.z` give the
# CollocationVariable rather than the bare Variable add_var registered.
function _rehandle(c::CollocationExaCore, old, new, name)
    var = map(v -> v === old ? new : v, getfield(c, :var))
    refs = name === nothing ? getfield(c, :refs) : (; getfield(c, :refs)..., _name_of(name) => new)
    return ExaCore(c; var = var, refs = refs)
end

_name_of(::Val{N}) where {N} = N

# The layout a constraint helper needs off a state block, with the checks it has to make.
# The block carries its own layout, so this only validates -- there is nothing to look up.
function _block_layout(c::CollocationExaCore, z, who::Symbol)
    z isa CollocationVariable || throw(ArgumentError(
        "$who: `z` must be a variable created by add_var_collocation, got a $(typeof(z))"
    ))

    K = _degree(c)
    z.krange == 0:K || throw(ArgumentError(
        "$who: that block was created with include_boundary = false, so it has no k = 0 " *
        "boundary node; the collocation stencil needs one."
    ))

    nlead = _nleading(z)
    1 <= nlead <= 2 || throw(ArgumentError(
        "$who: that block declares $nlead leading dimensions; the add_con_* helpers handle " *
        "one or two."
    ))
    return nlead, K
end

# ---------- property forwarding ----------
# The tag's contents and the quantities derived from them read flat off either the core or
# the model, which both carry the same tag.
const _MODE_FIELDS = (:roots, :basis, :polynomial, :weights)
const _TAG_FIELDS = (:mode, :mesh, :blocks, :residuals)

# The alias drops ExaCore's own `VT <: AbstractVector{T}` bound, which leaves these ambiguous
# with ExaModels' getproperty over `E <: Union{ExaCore, ExaModel}`. Restating it in the where
# clause is what two_stage.jl does throughout.
Base.getproperty(c::CollocationExaCore{T, VT, B}, name::Symbol) where {T, VT <: AbstractVector{T}, B} =
    _getprop(c, name)
Base.getproperty(m::CollocationExaModel{T, VT}, name::Symbol) where {T, VT <: AbstractVector{T}} =
    _getprop(m, name)

function _getprop(c, name::Symbol)
    hasfield(typeof(c), name) && return getfield(c, name)

    tag = _tag(c)
    name in _TAG_FIELDS && return getfield(tag, name)
    name === :N && return length(tag.mesh.h)
    name === :K && return length(tag.mode.weights.taus)
    name === :nodes && return tag.mesh.nodes
    name === :adaptive && return tag.mesh.hpar !== nothing
    name in _MODE_FIELDS && return getfield(tag.mode, name)

    refs = getfield(c, :refs)
    hasfield(typeof(refs), name) && return getfield(refs, name)
    return getfield(c, name)
end

Base.propertynames(c::CollocationExaCore{T, VT, B}) where {T, VT <: AbstractVector{T}, B} =
    _propnames(c)
Base.propertynames(m::CollocationExaModel{T, VT}) where {T, VT <: AbstractVector{T}} =
    _propnames(m)
_propnames(c) = (
    fieldnames(typeof(c))...,
    _TAG_FIELDS..., :N, :K, :nodes, :adaptive, _MODE_FIELDS...,
    keys(getfield(c, :refs))...,
)

Base.show(io::IO, c::CollocationExaCore{T, VT, B}) where {T, VT <: AbstractVector{T}, B} =
    _show_collocation(io, c, "A CollocationExaCore")
Base.show(io::IO, m::CollocationExaModel{T, VT}) where {T, VT <: AbstractVector{T}} =
    _show_collocation(io, m, "A CollocationExaModel")

function _show_collocation(io::IO, c, header)
    tag = _tag(c)
    nodes = tag.mesh.nodes
    vars = join(
        ["$(b.name)[$(join(b.dims, ", ")), i, k=$(b.krange)]" for b in tag.blocks], ", ",
    )
    print(
        io,
        """
        $header

          mesh    N = $(length(tag.mesh.h)) intervals, K = $(length(tag.mode.weights.taus)) degree$(tag.mesh.hpar === nothing ? "" : ", adaptive")
          horizon [$(first(nodes)), $(last(nodes))]
          mode    $(tag.mode.basis), $(tag.mode.polynomial), $(tag.mode.roots)
          vars    $(isempty(vars) ? "-" : vars)
        """,
    )
end
