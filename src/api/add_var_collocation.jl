# ExaModels.add_var with the mesh axes m=1,...,M where M>1, i=1,...,N, k=0/1,...,K appended.
"""
    add_var_collocation(core, dims...; include_boundary = true, mesh = nothing, name = nothing, kwargs...)

Adds a `CollocationVariable` with dimensions `dims` and appended mesh indices from
[`CollocationExaCore`](@ref) to `core`. Mesh indices consist of the mesh index `m in 1:M` if
`nodes` is a vector of meshes, the interval index `i in 1:N`, and interpolation index `k in krange`.
Returns `(core, CollocationVariable)`.

# Keyword Arguments
- `include_boundary` : `true` for `k = 0,…,K`, `false` for `k = 1,…,K`
- `mesh` : pin the `CollocationVariable` to the chosen mesh for when adding constraints
- `name` : when given as `Val(:name)`, registers the variable in `core` for later retrieval as `core.name`. See [`@add_var_collocation`](@ref) for the idiomatic named interface.
- remaining kwargs passed on to `ExaModels.add_var`: `start`, `lvar`, `uvar`, `tag`

# Example
```julia
# if M = 1,
julia> c, z = add_var_collocation(c, 1:3, 1:2) # z[v,c,i,k], 3 × 2 × N × (K+1)

# if M > 1,
julia> c, y = add_var_collocation(c, 1:2; include_boundary = false, mesh = 2) # y[v,i,k], 2 × N × K

julia> c, u = add_var_collocation(c) # u[m,i,k], M × N × (K+1)
```
"""
function add_var_collocation(
        core::CollocationExaCore,
        dims...;
        include_boundary::Bool = true,
        mesh = nothing,
        name = nothing,
        kwargs...,
    )
    K, M = _degree(core), _num_meshes(core)
    krange = include_boundary ? (0:K) : (1:K)
    pin = _meshpin(mesh, M, _hval(_mesh(core)))

    # A block over every mesh carries the mesh index as its last declared dimension
    dims = map(_dimrange, dims)
    dims = pin === nothing ? (dims..., 1:M) : dims
    _checkfill(kwargs, _nintervals(core) * length(krange) * prod(length, dims; init = 1))

    core, var = ExaModels.add_var(
        core, dims..., 1:_nintervals(core), krange;
        name = name, kwargs...,
    )

    z = CollocationVariable(var, dims, krange, pin)
    return _addblock(_rehandle(core, var, z, name), z), z
end

_dimrange(d::Integer) = 1:Int(d)
_dimrange(d) = d

# add_var fills start/lvar/uvar by position and does not check their length, so one short of
# the block leaves the rest uninitialized
function _checkfill(kwargs, len)
    for key in (:start, :lvar, :uvar)
        haskey(kwargs, key) || continue
        n = _fillcount(kwargs[key])
        n === nothing || n == len || throw(DimensionMismatch(
            "add_var_collocation: `$key` needs $len entries for this block, got $n. A block " *
            "spanning all M meshes carries the mesh index as its last declared dimension."
        ))
    end
end

_fillcount(v::Number) = nothing
_fillcount(v) = Base.IteratorSize(v) isa Union{Base.HasLength, Base.HasShape} ?
    length(v) : nothing

# Which mesh a block lives on: every one by default, a single one when named, and
# mesh 1 outright where `nodes` was one vector, so that mesh grows no index
function _meshpin(mesh, M, h)
    mesh === nothing && return h isa AbstractMatrix ? nothing : 1
    mesh isa Integer && 1 <= mesh <= M || throw(ArgumentError(
        "add_var_collocation: `mesh` must name one of the $M meshes, got $mesh"
    ))
    return Int(mesh)
end

"""
    @add_var_collocation(core, [name,] dims...; kwargs...)

Macro interface for [`add_var_collocation`](@ref). Updates `core` in the calling scope.

- **Named** (`@add_var_collocation(core, name, dims...)`): binds `name` to the new
  `CollocationVariable` in the local scope and registers it in `core` for later retrieval as
  `core.name` or `model.name`.
- **Anonymous** (`@add_var_collocation(core, dims...)`): equivalent to
  `c, name = add_var_collocation(c, dims...)`.

Accepts the same keyword arguments as [`add_var_collocation`](@ref).

# Example
```julia
julia> @add_var_collocation(c, z, 1:3) # z is now in scope; core.z also works
```
"""
macro add_var_collocation(exs...)
    args, kwargs = _split_collocation_args(exs)
    isempty(args) && error("@add_var_collocation requires a core argument")

    core = args[1]
    named = length(args) >= 2 && args[2] isa Symbol
    name = named ? args[2] : nothing
    dims = args[(named ? 3 : 2):end]
    var = gensym(:var)

    return quote
        local $var
        $(esc(core)), $var = add_var_collocation(
            $(esc(core)),
            $(map(esc, dims)...);
            name = $(_name_val(name)),
            $(map(esc, kwargs)...),
        )
        $(name === nothing ? var : :($(esc(name)) = $var))
        $var
    end
end
