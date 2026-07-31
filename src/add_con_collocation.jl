# Collocation residual, as a thin wrapper over ExaModels.add_con.
#
# Relative to add_con, this does three things and nothing else:
#   1. puts the supplied expression into the residual form of the core's CollocationMode
#   2. attaches the basis-polynomial sum as a constraint augmentation (add_con!)
#   3. runs the iterator across the collocation points
#
# The two bases are structurally different residuals over the *same* variables
# (Biegler, Nonlinear Programming, Ch. 10.2.1). Writing f_ij = f(z[...,i,j], t[i,j]):
#
#   StateForm      (10.7)  sum_{j=0..K} A[j,k] z[...,i,j]  =  h[i] f_ik
#   DerivativeForm (10.8)  z[...,i,k] - z[...,i,0]         =  h[i] sum_{j=1..K} A[j,k] f_ij
#
# StateForm puts the state under the weights and evaluates f once per row; DerivativeForm
# puts f under the weights and evaluates it at every collocation point of the interval. zdot
# is never a variable in either -- it is f evaluated there -- so add_var_collocation
# allocates the same block for both, and only the residual changes.
#
# add_con_collocation takes rows that carry every index of z, so the stencil reads them
# straight off the tuple. The macro adds nothing to that: it rebinds `core` and writes the
# constraint name bare, exactly as @add_var_collocation does.

# add_con! keys its augmentation by position in the base iterator, so rows are flattened on
# the way in: a caller writing `[(v,c) for v in 1:Nz, c in 1:Nc]` gets a matrix, and
# column-major order is the order the rows were written in either way.
_flat(itr) = vec(collect(itr))

# The row the caller writes:
#
#     (z's own indices..., anything the right-hand side varies with..., i, k, t)
#
# z's indices lead, so a row says which slot it constrains without being told twice; i, k, t
# trail, so the mesh entries sit in the same place whatever else the row carries. A helper
# appends its own data past `len`, where the caller's f never looks.
struct RowLayout
    len::Int     # length of the caller's row
    nlead::Int   # leading entries, naming z's slot
    i::Int       # position of the interval index
    k::Int       # position of the collocation index
end

function _row_layout(itr, nlead, nmesh, who::Symbol)
    isempty(itr) && return RowLayout(nlead + nmesh, nlead, nlead + 1, nlead + 2)
    len = length(first(itr))
    len >= nlead + nmesh || throw(ArgumentError(
        "$who: that block carries $nlead leading dimensions, so the iterator rows read " *
        "($(nlead == 1 ? "v" : "v, c"), …, i, k$(nmesh == 3 ? ", t" : "")); the ones given " *
        "carry $len entries."
    ))
    return RowLayout(len, nlead, len - nmesh + 1, len - nmesh + 2)
end

# The slot of z a row constrains, and everything ahead of the mesh entries.
_slot(d, L::RowLayout) = ntuple(j -> d[j], L.nlead)
_head(d, L::RowLayout) = d[1:(L.i - 1)]

"""
    add_con_collocation(core, z, generator; name = nothing, kwargs...)

Adds the collocation residual for the variable `z` to `core`, enforcing `dz/dt = f` at every
collocation point of the mesh. Returns `(core, Constraint)`.

`generator` gives the right-hand side `f` over an iterator of flat tuples, destructured in
the generator body the way ExaModels iterators normally are:

    (z's own indices…, anything else f varies with…, i, k, t)

The leading indices are the slot of `z` that row constrains, so an index held fixed is
written as a literal. `i` and `k` run over the intervals and collocation points, and `t` is
the collocation time `core.mesh.t[i,k]`, carried in the tuple because a traced index reads
it off the tuple rather than out of a plain array; leave it unused if `f` is autonomous. On
an adaptive mesh `t` is a parameter block, and a graph node cannot ride in an iterator tuple,
so the row ends at `k` and `f` indexes `core.mesh.tpar[i,k]` instead. One call per
structurally distinct `f`.

`core.basis` sets the residual the expression is put into. With `f_ij = f(z[…,i,j], t[i,j])`,
for `k = 1,…,K`:

| `basis` | residual |
|---|---|
| `StateForm` | `Σⱼ₌₀..ᴷ A[j,k] z[…,i,j] = h[i] f_ik` |
| `DerivativeForm` | `z[…,i,k] − z[…,i,0] = h[i] Σⱼ₌₁..ᴷ A[j,k] f_ij` |

`h[i]` is attached by the helper, and `k = 0` is the interval-left boundary node
[`add_var_collocation`](@ref) allocates with `include_boundary = true`.

## Keyword Arguments
- `name` : When given as `Val(:name)`, registers the constraint in `core` for later retrieval as `core.name` or `model.name`. See [`@add_con_collocation`](@ref) for the idiomatic named interface.
- Remaining keyword arguments are passed on to `ExaModels.add_con` and mean exactly what they do there: `lcon`, `ucon`, `start`, `tag`.

## Example
```julia
julia> itr = [(v, c, i, k, core.mesh.t[i,k])
              for v in 1:Nz, c in 1:Nc, i in 1:core.N, k in 1:core.K];

julia> core, coll = add_con_collocation(core, z,
           -decay[v] * z[v,c,i,k] for (v,c,i,k,t) in itr);       # z[v,c] is the row's slot
```
"""
function add_con_collocation(
        core::CollocationExaCore,
        z,
        gen::Base.Generator;
        name = nothing,
        kwargs...,
    )
    nlead, K = _block_layout(core, z, :add_con_collocation)
    mesh, w = _mesh(core), _weights(core)

    itr, f = _flat(gen.iter), gen.f
    L = _row_layout(itr, nlead, _nmesh(core), :add_con_collocation)
    hp = mesh.hpar    # nothing on a numeric mesh, the h parameter block on an adaptive one

    local con
    if _isstateform(core)
        # 10.7. Base rows carry -h f; the weights ride on the state, j = 0,...,K. On a
        # numeric mesh h is appended past the caller's row, in plain Julia before any
        # tracing, so f reads its own row and leaves the entry past it alone; on an adaptive
        # one it is indexed off the parameter block by the row's own i.
        base = hp === nothing ? [(row..., mesh.h[row[L.i]]) for row in itr] : itr
        rhs = hp === nothing ? (r -> -r[L.len + 1] * f(r)) : (r -> -hp[r[L.i]] * f(r))
        core, con = ExaModels.add_con(
            core, Base.Generator(rhs, base); name = name, kwargs...,
        )

        st = vec([
            (n, w.A[j + 1, d[L.k]], _slot(d, L)..., d[L.i], j)
            for (n, d) in enumerate(itr), j in 0:K
        ])
        core, _ = ExaModels.add_con!(core, con, Base.Generator(_state_stencil(z, nlead), st))
    else
        # 10.8. Base rows carry z[...,i,k] - z[...,i,0]; the weights ride on f, which is
        # re-evaluated at every collocation point j = 1,...,K of the same interval. The
        # stencil rows are the caller's own rows with the mesh entries moved to that point,
        # so the very same f means f_ij here and f_ik above.
        base = [(_slot(d, L)..., d[L.i], d[L.k]) for d in itr]
        core, con = ExaModels.add_con(
            core, Base.Generator(_derivative_base(z, nlead), base);
            name = name, kwargs...,
        )

        # Each stencil row is the caller's own row with the mesh entries moved to j, then the
        # augmentation key and weight appended past it. On an adaptive mesh the row carries
        # no t -- the caller reads it off the parameter block, so writing tpar[i,k] means
        # tpar[i,j] here for free -- and h rides on the generator instead of the weight.
        st = hp === nothing ?
            vec([
                (_head(d, L)..., d[L.i], j, mesh.t[d[L.i], j],
                 n, -mesh.h[d[L.i]] * w.A[j, d[L.k]])
                for (n, d) in enumerate(itr), j in 1:K
            ]) :
            vec([
                (_head(d, L)..., d[L.i], j, n, w.A[j, d[L.k]])
                for (n, d) in enumerate(itr), j in 1:K
            ])
        aug = hp === nothing ?
            (s -> s[L.len + 1] => s[L.len + 2] * f(s)) :
            (s -> s[L.len + 1] => -hp[s[L.i]] * s[L.len + 2] * f(s))
        core, _ = ExaModels.add_con!(core, con, Base.Generator(aug, st))
    end

    # Which slots were collocated, and with what f. Continuity reads its rows off this, and
    # under DerivativeForm integrates the same f, so it is recorded for either basis.
    core = _addresidual(core, Residual(z, itr, f, unique(_slot(d, L) for d in itr)))
    return core, con
end

# The two stencils that index z, one per leading-dimension count _block_layout admits.
# n => a * z[slot..., i, j], off a row (n, a, slot..., i, j)
_state_stencil(z, nlead) = nlead == 1 ?
    (s -> s[1] => s[2] * z[s[3], s[4], s[5]]) :
    (s -> s[1] => s[2] * z[s[3], s[4], s[5], s[6]])

# z[slot..., i, k] - z[slot..., i, 0], off a row (slot..., i, k)
_derivative_base(z, nlead) = nlead == 1 ?
    (r -> z[r[1], r[2], r[3]] - z[r[1], r[2], 0]) :
    (r -> z[r[1], r[2], r[3], r[4]] - z[r[1], r[2], r[3], 0])

"""
    @add_con_collocation(core, [name,] z, generator; kwargs...)

Macro interface for [`add_con_collocation`](@ref). Updates `core` in the calling scope.

- **Named** (`@add_con_collocation(core, coll, z, generator)`): binds `coll` to the new
  `Constraint` in the local scope and registers it in `core` for later retrieval as
  `core.coll` or `model.coll`.
- **Anonymous** (`@add_con_collocation(core, z, generator)`): equivalent to
  `core, coll = add_con_collocation(core, z, generator)`.

Accepts the same keyword arguments as [`add_con_collocation`](@ref).

## Example
```julia
julia> @add_var_collocation(core, z, 1:Nz)

julia> @add_var_collocation(core, u, 1:Nu; include_boundary = false)

julia> itr = [(v, l[v], i, k, core.mesh.t[i,k])
              for v in 1:Nz, i in 1:core.N, k in 1:core.K];

julia> @add_con_collocation(core, coll, z,
           z[v,i,k] * u[l,i,k] * cos(t) for (v,l,i,k,t) in itr)
```
"""
macro add_con_collocation(exs...)
    args, kwargs = _split_collocation_args(exs)
    length(args) in (3, 4) ||
        error("@add_con_collocation requires core, an optional name, a variable, and a generator")

    core = args[1]
    name, zsym, gen = length(args) == 4 ? args[2:4] : (nothing, args[2], args[3])
    name === nothing || name isa Symbol ||
        error("@add_con_collocation: the second argument must be the constraint name")

    con = gensym(:con)

    return quote
        local $con
        $(esc(core)), $con = add_con_collocation(
            $(esc(core)),
            $(esc(zsym)),
            $(esc(gen));
            name = $(_name_val(name)),
            $(map(esc, kwargs)...),
        )
        $(name === nothing ? con : :($(esc(name)) = $con))
        $con
    end
end
