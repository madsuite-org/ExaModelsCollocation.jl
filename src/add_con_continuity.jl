# Continuity across interval junctions, for i = 1,...,N-1.
#
# Per basis (Biegler, Nonlinear Programming, Ch. 10.2.1), with f_ij = f(z[...,i,j], t[i,j]):
#
#   StateForm      (10.14a)  sum_{j=0..K} b[j] z[...,i,j]  =  z[...,i+1,0]
#   DerivativeForm (10.15a)  z[...,i+1,0] - z[...,i,0]     =  h[i] sum_{j=1..K} b[j] f_ij
#
# StateForm evaluates the state polynomial at tau = 1; DerivativeForm integrates the f
# add_con_collocation was already given. Both read their slots off that same record, so the
# call is the same either way: name the variable, and the mode decides what is built.

"""
    add_con_continuity(core, z; name = nothing, kwargs...)

Adds the constraints tying each interval's terminal polynomial value to the next interval's
boundary node, for `i = 1,…,N-1`. Returns `(core, Constraint)`.

Every slot of `z` is tied, and each must be covered by exactly one
[`add_con_collocation`](@ref) call, so those calls come first — for either basis.
`core.basis` sets what is built, with `f_ij = f(z[…,i,j], t[i,j])`:

| `basis` | residual |
|---|---|
| `StateForm` | `Σⱼ₌₀..ᴷ b[j] z[…,i,j] = z[…,i+1,0]` |
| `DerivativeForm` | `z[…,i+1,0] − z[…,i,0] = h[i] Σⱼ₌₁..ᴷ b[j] f_ij` |

## Keyword Arguments
- `name` : When given as `Val(:name)`, registers the constraint in `core` for later retrieval as `core.name` or `model.name`. See [`@add_con_continuity`](@ref) for the idiomatic named interface.
- Remaining keyword arguments are passed on to `ExaModels.add_con` and mean exactly what they do there: `lcon`, `ucon`, `start`, `tag`.

## Example
```julia
julia> core, cont = add_con_continuity(core, z);
```
"""
function add_con_continuity(
        core::CollocationExaCore,
        z;
        name = nothing,
        kwargs...,
    )
    nlead, K = _block_layout(core, z, :add_con_continuity)
    res = _residuals(core, z)
    slots = _covered_slots(res, z, :add_con_continuity)
    mesh, w, N = _mesh(core), _weights(core), _nintervals(core)

    # One junction row per slot and interval boundary, either way. Which junctions is the
    # mode's business, so they are crossed in here.
    rows = vec([(d..., i) for d in slots, i in 1:(N - 1)])

    # Radau and Lobatto collocate tau = 1, so the row is z[...,i,K] = z[...,i+1,0]
    if last(w.taus) == 1
        return ExaModels.add_con(
            core, Base.Generator(_cardinal_base(z, nlead, K), rows); name = name, kwargs...,
        )
    end

    base = _isstateform(core) ? _continuity_base(z, nlead) : _junction_base(z, nlead)
    core, con = ExaModels.add_con(
        core, Base.Generator(base, rows); name = name, kwargs...,
    )

    if _isstateform(core)
        # 10.14a. The weights ride on the state at j = 0,...,K, so the row needs no f.
        st = vec([(n, w.b[j + 1], r..., j) for (n, r) in enumerate(rows), j in 0:K])
        core, _ = ExaModels.add_con!(core, con, Base.Generator(_state_stencil(z, nlead), st))
    else
        # 10.15a. Every row of a recorded collocation iterator is an f_ik, so it carries its
        # own b[k] into the junction row of the slot it leads with. One augmentation per
        # recorded right-hand side, since each is a structurally distinct expression.
        pos = Dict(r => n for (n, r) in enumerate(rows))
        hp = mesh.hpar
        for r in res
            L = _row_layout(r.rows, nlead, _nmesh(core), :add_con_continuity)
            keep = [d for d in r.rows if d[L.i] < N && haskey(pos, (_slot(d, L)..., d[L.i]))]
            isempty(keep) && continue
            st = hp === nothing ?
                [(d..., pos[(_slot(d, L)..., d[L.i])], -mesh.h[d[L.i]] * w.b[d[L.k]]) for d in keep] :
                [(d..., pos[(_slot(d, L)..., d[L.i])], w.b[d[L.k]]) for d in keep]
            aug = hp === nothing ?
                (s -> s[L.len + 1] => s[L.len + 2] * r.f(s)) :
                (s -> s[L.len + 1] => -hp[s[L.i]] * s[L.len + 2] * r.f(s))
            core, _ = ExaModels.add_con!(core, con, Base.Generator(aug, st))
        end
    end

    return core, con
end

# The slots the recorded collocation calls cover, in the order they were added. Every slot of
# the block must be covered exactly once: twice and a junction row would count two
# right-hand sides, not at all and it would be silently left untied.
function _covered_slots(res, z, who::Symbol)
    slots = Any[]
    for r in res, s in r.slots
        s in slots && throw(ArgumentError(
            "$who: two collocation calls cover $(_slotstr(s)), so its junction row would " *
            "integrate both right-hand sides"
        ))
        push!(slots, s)
    end

    # A block declared with no dimensions is one slot, the empty tuple, so this covers it too.
    missing = [s for s in Iterators.product(z.dims...) if !(s in slots)]
    isempty(missing) || throw(ArgumentError(
        "$who: no add_con_collocation call covers " *
        join((_slotstr(s) for s in Iterators.take(missing, 3)), ", ") *
        (length(missing) > 3 ? ", …" : "") *
        ". Collocate every slot of the block first, or declare the rest as their own block."
    ))
    return slots
end

_slotstr(s) = isempty(s) ? "the block" : "z[$(join(s, ", "))]"

# z[zdims..., i, K] - z[zdims..., i+1, 0], off a junction row (zdims..., i)
_cardinal_base(z, nlead, K) = nlead == 0 ?
    (r -> z[r[1], K] - z[r[1] + 1, 0]) :
    nlead == 1 ?
    (r -> z[r[1], r[2], K] - z[r[1], r[2] + 1, 0]) :
    (r -> z[r[1], r[2], r[3], K] - z[r[1], r[2], r[3] + 1, 0])

# -z[zdims..., i+1, 0], off a junction row (zdims..., i); the b-sum rides on the stencil
_continuity_base(z, nlead) = nlead == 0 ?
    (r -> -z[r[1] + 1, 0]) :
    nlead == 1 ?
    (r -> -z[r[1], r[2] + 1, 0]) :
    (r -> -z[r[1], r[2], r[3] + 1, 0])

# z[zdims..., i+1, 0] - z[zdims..., i, 0], off a junction row (zdims..., i)
_junction_base(z, nlead) = nlead == 0 ?
    (r -> z[r[1] + 1, 0] - z[r[1], 0]) :
    nlead == 1 ?
    (r -> z[r[1], r[2] + 1, 0] - z[r[1], r[2], 0]) :
    (r -> z[r[1], r[2], r[3] + 1, 0] - z[r[1], r[2], r[3], 0])

"""
    @add_con_continuity(core, [name,] z; kwargs...)

Macro interface for [`add_con_continuity`](@ref). Updates `core` in the calling scope.

- **Named** (`@add_con_continuity(core, cont, z)`): binds `cont` to the new `Constraint` in
  the local scope and registers it in `core` for later retrieval as `core.cont` or
  `model.cont`.
- **Anonymous** (`@add_con_continuity(core, z)`): equivalent to
  `core, cont = add_con_continuity(core, z)`.

Accepts the same keyword arguments as [`add_con_continuity`](@ref).

## Example
```julia
@add_con_continuity(core, cont, z)
```
"""
macro add_con_continuity(exs...)
    args, kwargs = _split_collocation_args(exs)
    length(args) in (2, 3) ||
        error("@add_con_continuity requires core, an optional name, and a variable")

    core = args[1]
    name, zsym = length(args) == 3 ? (args[2], args[3]) : (nothing, args[2])
    name === nothing || name isa Symbol ||
        error("@add_con_continuity: the second argument must be the constraint name")

    con = gensym(:con)

    return quote
        local $con
        $(esc(core)), $con = add_con_continuity(
            $(esc(core)),
            $(esc(zsym));
            name = $(_name_val(name)),
            $(map(esc, kwargs)...),
        )
        $(name === nothing ? con : :($(esc(name)) = $con))
        $con
    end
end
