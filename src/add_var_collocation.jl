# ExaModels.add_var with the two mesh axes i=1,...,N, k=0/1,...,K appended.
"""
    add_var_collocation(core, dims...; include_boundary = true, name = nothing, kwargs...)

Adds variables with dimensions `dims` over the collocation mesh of `core`. `dims` is the
shape at a single collocation point; the interval index over `1:N` and the collocation index
over `krange` are appended. Passing none gives a scalar state, `z[i,k]`. Returns
`(core, CollocationVariable)`.

## Keyword Arguments
- `include_boundary` : `true` (default) gives `k = 0,…,K`, carrying the interval-left boundary node that continuity needs; `false` gives `k = 1,…,K`.
- `name` : when given as `Val(:name)`, registers the variable in `core` for later retrieval as `core.name` or `model.name`. See [`@add_var_collocation`](@ref) for the idiomatic named interface.
- Remaining keyword arguments are passed on to `ExaModels.add_var`. `start`, `lvar`, `uvar` and `tag` are shaped to the **allocated** block, `(dims..., 1:N, krange)`.

## Example
```julia
julia> core = CollocationExaCore(range(0.0, 5.0; length = 21), 3);

julia> core, z = add_var_collocation(core, 1:3, 1:2);   # z[v,c,i,k], 3 × 2 × N × (K+1)

julia> core, u = add_var_collocation(core, 1:1, 1:2; include_boundary = false);   # k = 1,…,K
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

# ExaModels takes an Integer or a UnitRange per dimension, `n` meaning 1:n (_start/_length in
# its nlp.jl). The block carries its own dims and _covered_slots enumerates them to check that
# every slot was collocated, so they are normalized here rather than left in either spelling.
_dimrange(d::Integer) = 1:Int(d)
_dimrange(d) = d

"""
    @add_var_collocation(core, [name,] dims...; kwargs...)

Macro interface for [`add_var_collocation`](@ref). Updates `core` in the calling scope.

- **Named** (`@add_var_collocation(core, z, dims...)`): binds `z` to the new
  `CollocationVariable` in the local scope and registers it in `core` for later retrieval as
  `core.z` or `model.z`.
- **Anonymous** (`@add_var_collocation(core, dims...)`): equivalent to
  `core, z = add_var_collocation(core, dims...)`.

Accepts the same keyword arguments as [`add_var_collocation`](@ref).

## Example
```julia
core = CollocationExaCore(range(0.0, 5.0; length = 21), 3)
@add_var_collocation(core, z, 1:3, 1:2)                            # z[v,c,i,k], k = 0,…,K
@add_var_collocation(core, u, 1:1, 1:2; include_boundary = false)  # u[v,c,i,k], k = 1,…,K
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

# `name = Val(:z)` for a named macro call, `name = nothing` for an anonymous one
_name_val(name::Symbol) = Val(name)
_name_val(::Nothing) = nothing

# Split macro arguments into positional and keyword parts, accepting both
# `f(a, b; k = v)` and `f(a, b, k = v)` spellings.
function _split_collocation_args(exs)
    args = Any[]
    kwargs = Any[]
    for e in exs
        if e isa Expr && e.head === :parameters
            append!(kwargs, e.args)
        elseif e isa Expr && (e.head === :(=) || e.head === :kw)
            push!(kwargs, Expr(:kw, e.args[1], e.args[2]))
        else
            push!(args, e)
        end
    end
    return args, kwargs
end
