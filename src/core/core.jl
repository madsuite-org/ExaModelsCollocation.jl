# CollocationExaCore things
"""
    Residual

Contains [`add_con_collocation`](@ref) call details.

# Fields
- `var` : the `CollocationVariable` that was collocated
- `f` : right-hand side function expression
- `fiter` : the full iterator used by `f`
- `fwhich` : the `CollocationVariable` indicies that were collocated
- `fwhere` : where `dims...`, `i`, `k` are located in `fiter`
"""
struct Residual{V, F, I, W, L}
    var::V
    f::F
    fiter::I
    fwhich::W
    fwhere::L
end

"""
    CollocationMode

Contains details on what method of collocation is used.

# Fields
- `roots`      : collocation family, `GaussRadau()`, `GaussLendgre()`, or `GaussLobatto()`
- `basis`      : differential-state representation, `StateForm()` or `DerivativeForm`
- `polynomial` : interpolating polynomial, `Lagrange()`
- `weights`    : `A`, `b`, `taus`
"""
struct CollocationMode{R <: AbstractRoots, B <: AbstractBasis, P <: AbstractPolynomial, W <: BasisWeights}
    roots::R
    basis::B
    polynomial::P
    weights::W
end

"""
    CollocationTag

The collocation metadata a [`CollocationExaCore`](@ref) carries in its `tag`. 
Built with [`Collocation`](@ref).

# Fields
- `mode`  : [`CollocationMode`](@ref)
- `mesh`  : [`CollocationMesh`](@ref)
- `block` : every [`CollocationVariable`](@ref) dimensions
- `resid` : every [`Residual`](@ref) recorded by [`add_con_collocation`](@ref)
"""
struct CollocationTag{MO <: CollocationMode, ME <: CollocationMesh} <: ExaModels.AbstractExaModelTag
    mode::MO
    mesh::ME
    block::Vector{Any}
    resid::Vector{Residual}
end

"""
    Collocation(nodes, K; kwargs...)

Contains collocation metadata as an `ExaCore` tag.

# Example
```julia
julia> core = ExaCore(tag = Collocation(nodes, 3))

julia> core = ExaCore(core; tag = Collocation(nodes, 3))
```
"""
function Collocation(
        nodes::AbstractVector,
        K::Integer;
        roots::AbstractRoots = GaussRadau(),
        basis::AbstractBasis = StateForm(),
        polynomial::AbstractPolynomial = Lagrange(),
    )
    K >= 1 || throw(ArgumentError("K must be at least 1, got $K"))

    bounds = _nodes_input(nodes)
    nb = size(bounds, ndims(bounds))
    nb >= 2 || throw(ArgumentError("nodes requires at least 2 interval boundaries, got $nb"))
    all(>(0), _interval_widths(bounds)) ||
        throw(ArgumentError("nodes must be strictly increasing along t"))

    # polynomial.jl: only Lagrange polynomial is implemented
    polynomial isa Lagrange ||
        throw(ArgumentError("Only Lagrange interpolation polynomials supported currently"))

    # taus.jl: Lobatto must include both endpoints so at least K = 2 required
    roots isa GaussLobatto && K < 2 &&
        throw(ArgumentError("GaussLobatto requires K ≥ 2, got $K"))

    # basis.jl: StateForm cant collocate at tau = 0, but Lobatto has tau1 = 0
    roots isa GaussLobatto && basis isa StateForm &&
        throw(ArgumentError("GaussLobatto requires basis = DerivativeForm(), got $basis"))

    # taus.jl: the K collocation points; basis.jl: collocation/continuity weights A, b
    taus = _get_taus(roots, K)
    mode = CollocationMode(roots, basis, polynomial, _get_weights(polynomial, basis, taus))
    return CollocationTag(mode, _get_mesh(bounds, taus), [], Residual[])
end

"""
    CollocationExaCore{T,VT,B}

Type alias for an `ExaCore` whose `tag` is a [`CollocationTag`](@ref).

    CollocationExaCore(
        [array_eltype::Type,] nodes, K; 
        roots = GaussRadau(),
        basis = StateForm(), 
        polynomial = Lagrange(), 
        adaptive = false,
        unknown_horizon = false,
        kwargs...
    )

Creates an intermediate data object `CollocationExaCore`, which contains collocation metadata
used by collocation helper functions.

# Arguments
- `nodes` : vector of interval boundary placements for `N+1` boundaries for `N` intervals in the mesh, or a vector of `M` meshes
- `K`     : degree of interpolating polynomial

# Keyword Arguments
- `roots`           : collocation family, `GaussRadau()`, `GaussLegendre()`, or `GaussLobatto()`
- `basis`           : differential-state representation, `StateForm()` or `DerivativeForm()`
- `polynomial`      : interpolating polynomial, `Lagrange()`
- `adaptive`        : interval widths are mutable `ExaModels` parameters
- `unknown_horizon` : time horizon is a decision variable (Experimental)
- remaining kwargs passed on to `ExaCore`: `backend`, `minimize`, `name`

# Fields
- `mode`  : `roots`, `basis`, `polynomial`, `weights`
- `mesh`  : `nodes`, `h` interval lengths, `t` time
- `block` : `CollocationVariable` dimensions
- `resid` : `CollocationVariable` right-hand side functions
- `N`, `K`, `M`, `nodes`, `adaptive`, `unknown_horizon`

# Example
```julia
julia> nodes = range(0.0, 5.0; length = 21) # 21 interval boundary placements

julia> core = CollocationExaCore(nodes, 3) # N=20, K=3

julia> core = ExaCore(tag = Collocation(nodes, 3)) # also works

julia> core = ExaCore(core; tag = Collocation(nodes, 3)) # also works
```
"""
const CollocationExaCore{T, VT, B} = ExaCore{T, VT, B, <:CollocationTag}

"""
    CollocationExaModel{T,VT,E,V,P,O,C,R}

Type alias for an `ExaModel` built from a [`CollocationExaCore`](@ref).
"""
const CollocationExaModel{T, VT, E, V, P, O, C, R} = ExaModels.ExaModel{T, VT, E, V, P, O, C, <:CollocationTag, R}

CollocationExaCore(nodes::AbstractVector, K::Integer; backend = nothing, kwargs...) =
    CollocationExaCore(ExaModels.default_T(backend), nodes, K; backend, kwargs...)

# Construct CollocationExaCore
function CollocationExaCore(
        ::Type{T},
        nodes::AbstractVector,
        K::Integer;
        roots::AbstractRoots = GaussRadau(),
        basis::AbstractBasis = StateForm(),
        polynomial::AbstractPolynomial = Lagrange(),
        adaptive::Bool = false,
        unknown_horizon::Bool = false,
        kwargs...,
    ) where {T <: AbstractFloat}

    # Both move h and t, so a mesh under a horizon variable has nothing left for set_nodes! to set
    adaptive && unknown_horizon && throw(ArgumentError(
        "CollocationExaCore: `adaptive` and `unknown_horizon` both take over h and t; pick one."
    ))

    tag = Collocation(nodes, K; roots, basis, polynomial)
    core = ExaCore(T; tag = tag, kwargs...)
    adaptive && return _make_adaptive(core)
    unknown_horizon && return _make_unknown_horizon(core)
    return core
end

# `adaptive = true` build path
function _make_adaptive(core)
    tag, mesh = _tag(core), _mesh(core)
    h, t = _hval(mesh), _tval(mesh)
    core, hp = ExaModels.add_par(core, map(n -> 1:n, size(h))...; value = h)
    core, tp = ExaModels.add_par(core, map(n -> 1:n, size(t))...; value = t)
    return ExaCore(
        core;
        tag = CollocationTag(
            tag.mode, _with_parameters(mesh, hp, tp), tag.block, tag.resid,
        ),
    )
end

# `unknown_horizon = true` build path. The mesh keeps its nominal numbers and the residuals read
# the relative placements off them, so tscale at its start value reproduces the numeric mesh.
function _make_unknown_horizon(core)
    tag, mesh = _tag(core), _mesh(core)
    tnom = _tnoms(mesh)
    core, ts = ExaModels.add_var(
        core, 1:length(tnom); start = tnom, lvar = zero(eltype(tnom)), name = Val(:tscale),
    )
    return ExaCore(
        core;
        tag = CollocationTag(tag.mode, _with_horizon_scale(mesh, ts), tag.block, tag.resid),
    )
end

# ---------- helper functions ----------

_tag(c) = getfield(c, :tag)
_mesh(c) = _tag(c).mesh
_mode(c) = _tag(c).mode
_weights(c) = _mode(c).weights
_nintervals(c) = (h = _hval(_mesh(c)); size(h, ndims(h)))
_num_meshes(c) = _num_meshes(_mesh(c))
_degree(c) = length(_weights(c).taus)

_hval(m::CollocationMesh) = getfield(m, :h)
_tval(m::CollocationMesh) = getfield(m, :t)
_hpar(m::CollocationMesh) = getfield(m, :hpar)
_tpar(m::CollocationMesh) = getfield(m, :tpar)
_horizon_scale(m::CollocationMesh) = getfield(m, :horizon_scale)

_isstateform(c) = _mode(c).basis isa StateForm
_isadaptive(m::CollocationMesh) = _hpar(m) !== nothing
_is_unknown_horizon(m::CollocationMesh) = _horizon_scale(m) !== nothing

# Copied rather than grown in place, so a core does not report what was added to a sibling
# derived from the same parent. The mesh stays shared: set_nodes! moves it for every core.
_addblock(c::CollocationExaCore, z::CollocationVariable) =
    _retag(c, push!(copy(_tag(c).block), z), _tag(c).resid)
_addresidual(c::CollocationExaCore, res::Residual) =
    _retag(c, _tag(c).block, push!(copy(_tag(c).resid), res))

_retag(c, block, resid) =
    ExaCore(c; tag = CollocationTag(_tag(c).mode, _tag(c).mesh, block, resid))

# The residuals recorded for a variable, in the order they were added
_residuals(c::CollocationExaCore, var) = [r for r in _tag(c).resid if r.var === var]

# Swap the handle a helper wrapped back into the core, so `core.z` and `model.z` give the
# CollocationVariable rather than the bare Variable add_var registered.
function _rehandle(c::CollocationExaCore, old, new, name)
    var = _swaphandle(getfield(c, :var), old, new)
    refs = name === nothing ? getfield(c, :refs) : (; getfield(c, :refs)..., _name_of(name) => new)
    return ExaCore(c; var = var, refs = refs)
end

# Keep the storage container ExaModels handed us: Tuple under `concrete = Val(true)`, else the
# Vector{Any} that `_materialize` needs. A plain `map` narrows the vector eltype and breaks it.
_swaphandle(v::Tuple, old, new) = map(x -> x === old ? new : x, v)
_swaphandle(v::AbstractVector, old, new) = Any[x === old ? new : x for x in v]

_name_of(::Val{N}) where {N} = N

# The layout a constraint helper needs off a state block, with the checks it has to make.
# The block carries its own layout, so this only validates -- there is nothing to look up.
function _block_layout(c::CollocationExaCore, z, who::Symbol)
    z isa CollocationVariable || throw(ArgumentError(
        "$who: `z` must be a variable created by add_var_collocation, got a $(typeof(z))"
    ))

    K = _degree(c)
    z.krange == 0:K || throw(ArgumentError(
        "$who: cannot create collocation constraints for a CollocationVariable created with " *
        "include_boundary = false."
    ))

    return _nleading(z), K
end

# ---------- property forwarding ----------

# The tag's contents and the quantities derived from them read flat off either the core or
# the model, which both carry the same tag.
const _MODE_FIELDS = (:roots, :basis, :polynomial, :weights)
const _TAG_FIELDS = (:mode, :mesh, :block, :resid)

# The alias drops ExaCore's own `VT <: AbstractVector{T}` bound, which leaves these ambiguous
# with ExaModels' getproperty over `E <: Union{ExaCore, ExaModel}`. Restating it in the where
# clause is what two_stage.jl does throughout.
Base.getproperty(c::CollocationExaCore{T, VT, B}, name::Symbol) where {T, VT <: AbstractVector{T}, B} =
    _getprop(c, name)
Base.getproperty(m::CollocationExaModel{T, VT}, name::Symbol) where {T, VT <: AbstractVector{T}} =
    _getprop(m, name)

function _getprop(c, name::Symbol)
    hasfield(typeof(c), name) && return getfield(c, name)

    # A registered handle takes precedence over a derived name: N and K in particular are
    # what a model would call its own variables.
    refs = getfield(c, :refs)
    hasfield(typeof(refs), name) && return getfield(refs, name)

    tag = _tag(c)
    name in _TAG_FIELDS && return getfield(tag, name)
    name === :N && return _nintervals(c)
    name === :K && return length(tag.mode.weights.taus)
    name === :nodes && return tag.mesh.nodes
    name === :M && return _num_meshes(c)
    name === :adaptive && return _isadaptive(tag.mesh)
    name === :unknown_horizon && return _is_unknown_horizon(tag.mesh)
    name in _MODE_FIELDS && return getfield(tag.mode, name)
    return getfield(c, name)
end

Base.propertynames(c::CollocationExaCore{T, VT, B}) where {T, VT <: AbstractVector{T}, B} =
    _propnames(c)
Base.propertynames(m::CollocationExaModel{T, VT}) where {T, VT <: AbstractVector{T}} =
    _propnames(m)
_propnames(c) = (
    fieldnames(typeof(c))...,
    _TAG_FIELDS..., :N, :K, :M, :nodes, :adaptive, :unknown_horizon, _MODE_FIELDS...,
    keys(getfield(c, :refs))...,
)

Base.show(io::IO, c::CollocationExaCore{T, VT, B}) where {T, VT <: AbstractVector{T}, B} =
    _show_collocation(io, c, "A CollocationExaCore")
Base.show(io::IO, m::CollocationExaModel{T, VT}) where {T, VT <: AbstractVector{T}} =
    _show_collocation(io, m, "A CollocationExaModel")

function _show_collocation(io::IO, c, header)
    tag, M = _tag(c), _num_meshes(c)
    vars = join(
        ["$(b.name)[$(join((_blockaxes(b)..., "i", "k=$(b.krange)"), ", "))]" for b in tag.block],
        ", ",
    )
    modes = join(filter(!isempty, [
        M == 1 ? "" : "$M meshes",
        _isadaptive(tag.mesh) ? "adaptive" : "",
        _is_unknown_horizon(tag.mesh) ? "unknown horizon" : "",
    ]), ", ")
    print(
        io,
        """
        $header

          mesh    N = $(_nintervals(c)) intervals, K = $(length(tag.mode.weights.taus)) degree$(isempty(modes) ? "" : ", $modes")
          horizon $(_horizons(tag.mesh, M))
          mode    $(tag.mode.basis), $(tag.mode.polynomial), $(tag.mode.roots)
          vars    $(isempty(vars) ? "-" : vars)
        """,
    )
end

# One span per mesh, first and last only past a few so many of them stay one line
function _horizons(mesh::CollocationMesh, M)
    ends(m) = (n = _mesh_nodes(mesh, m); "[$(first(n)), $(last(n))]")
    M <= 4 && return join((ends(m) for m in 1:M), " ")
    return "$(ends(1)) … $(ends(M))"
end

# The block's declared axes as shown, naming the mesh index a spanning block carries last
_blockaxes(b) = b.mesh === nothing ? (b.dims[1:(end - 1)]..., "m=$(last(b.dims))") : b.dims
