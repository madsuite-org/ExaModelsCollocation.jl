# ExaModels.add_var with the two mesh axes i=1,...,N, k=0/1,...,K appended.
"""
    add_var_collocation(core, dims...; include_boundary = true, name = nothing, kwargs...)

Adds a `CollocationVariable` with dimensions `dims` and appended mesh indices from
[`CollocationExaCore`](@ref) to `core`. Mesh indices consist of the interval index `i in 1:N`
and interpolation index `k in krange`. Returns `(core, CollocationVariable)`.

# Keyword Arguments
- `include_boundary` : `true` for `k = 0,…,K`, `false` for `k = 1,…,K`
- `name` : when given as `Val(:name)`, registers the variable in `core` for later retrieval as `core.name`. See [`@add_var_collocation`](@ref) for the idiomatic named interface.
- remaining kwargs passed on to `ExaModels.add_var`: `start`, `lvar`, `uvar`, `tag`

# Example
```julia
julia> c, z = add_var_collocation(c, 1:3, 1:2) # z[v,c,i,k], 3 × 2 × N × (K+1)
```
"""
function add_var_collocation(
        core::CollocationExaCore,
        dims...;
        include_boundary::Bool = true,
        name = nothing,
        kwargs...,
    )
    K = _degree(core)
    krange = include_boundary ? (0:K) : (1:K)
    dims = map(_dimrange, dims)

    core, var = ExaModels.add_var(
        core, dims..., 1:_nintervals(core), krange;
        name = name, kwargs...,
    )

    z = CollocationVariable(var, dims, krange)
    return _addblock(_rehandle(core, var, z, name), z), z
end

_dimrange(d::Integer) = 1:Int(d)
_dimrange(d) = d

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
