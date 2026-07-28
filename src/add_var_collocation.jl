# Collocation-aware wrapper around ExaModels.add_var.
# The caller declares the per-timepoint shape; the mesh indices (i, k) are appended.

"""
    add_var_collocation(core, dae, dims...; include_boundary = true, name = nothing, kwargs...)

Add a variable block laid out over the collocation mesh. A block declared with
per-timepoint dimensions `dims` is allocated with the interval index `i` and the
collocation index `k` appended, so

```julia
core, z = add_var_collocation(core, dae, 1:nz, 1:Nc)   #  z[v, c, i, k]
```

creates `nz × Nc × N × (K+1)` variables, where `N` and `K` come from `dae`.

`include_boundary` selects the collocation index range: `true` (the default) gives
`k = 0,…,K`, carrying the interval-left boundary node needed for continuity; `false` gives
`k = 1,…,K`, the collocation points alone.

Keyword arguments pass through to `ExaModels.add_var` and mean exactly what they do there —
including `name`, which registers the handle when given as `Val(:name)`. `start`, `lvar`,
`uvar`, and `tag` must be shaped to the **allocated** block, i.e. `(dims..., 1:N, krange)`.

Returns `(core, var)`, as `add_var` does. `dae` is mutable and is updated in place: the
block's layout is recorded either way and reachable with [`block`](@ref); a named handle is
additionally reachable as `dae.<name>`.

See [`@add_var_collocation`](@ref) for the form that names and binds it for you.
"""
function add_var_collocation(
        core::ExaCore,
        dae::DAEta,
        dims...;
        include_boundary::Bool = true,
        name = nothing,
        kwargs...,
    )
    _require_mesh(dae, :add_var_collocation)

    K = _degree(dae)
    krange = include_boundary ? (0:K) : (1:K)

    core, var = ExaModels.add_var(
        core, dims..., 1:_nintervals(dae), krange;
        name = name, kwargs...,
    )

    push!(getfield(dae, :blocks), VarBlock(var, dims, krange))
    name === nothing || _register!(dae, :vars, _name_of(name), var)
    return core, var
end

_name_of(::Val{N}) where {N} = N

"""
    @add_var_collocation(core, dae, name, dims...; kwargs...)

Macro interface for [`add_var_collocation`](@ref), relating to it exactly as
`ExaModels.@add_var` relates to `add_var`: `name` is written bare, becomes the `name = Val(…)`
keyword, and is bound in the calling scope. `core` is updated there too; `dae` is mutated in
place.

```julia
dae = DAEta(nodes, K)
@add_var_collocation(core, dae, z, 1:nz, 1:Nc)                        # z[v,c,i,k], k = 0,…,K
@add_var_collocation(core, dae, u, 1:nu, 1:Nc; include_boundary = false)  # u[v,c,i,k], k = 1,…,K
```

`z` and `u` are now bound locally, and also reachable as `dae.z` and `dae.u`.
"""
macro add_var_collocation(exs...)
    args, kwargs = _split_collocation_args(exs)
    length(args) >= 3 ||
        error("@add_var_collocation requires core, dae, and a name argument")

    core, dae, name = args[1], args[2], args[3]
    name isa Symbol ||
        error("@add_var_collocation: the third argument must be the variable name, got `$name`")
    dims = args[4:end]
    var = gensym(:var)

    return quote
        local $var
        $(esc(core)), $var = add_var_collocation(
            $(esc(core)),
            $(esc(dae)),
            $(map(esc, dims)...);
            name = $(Val(name)),
            $(map(esc, kwargs)...),
        )
        $(esc(name)) = $var
    end
end

# Split macro arguments into positional and keyword parts, accepting both
# `f(a, b; k = v)` (parameters block) and `f(a, b, k = v)` spellings.
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
