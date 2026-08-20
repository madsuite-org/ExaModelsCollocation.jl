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

# add_con! keys its augmentation by position in the base iterator, so rows are flattened on
# the way in: a caller writing `[(v,c) for v in 1:Nz, c in 1:Nc]` gets a matrix, and
# column-major order is the order the rows were written in either way.
_flat(itr) = vec(collect(itr))

# A slot index that is the same for every row, written as a literal rather than carried
struct Fixed{V}
    v::V
end

# The row the residual is built over:
#
#     (anything the right-hand side varies with..., i, k)
#
# i and k trail, the mesh's own data is appended past them, and where the slot indices sit is
# the probe's to say.
struct RowLayout{S, M}
    len::Int     # length of the row, before the mesh appends its own data
    i::Int       # position of the interval index
    k::Int       # position of the collocation index
    slot::S      # per declared dimension: a row position, or a Fixed literal
    m::M         # where the mesh index sits, or a Fixed one for a block pinned to a mesh
end

function _row_layout(itr, slot, meshat, who::Symbol)
    isempty(itr) && throw(ArgumentError(
        "$who: the iterator is empty, so there is no collocation constraint to add."
    ))
    len = length(first(itr))
    len >= 2 || throw(ArgumentError(
        "$who: the iterator rows must end in (…, i, k); the ones given carry $len entries."
    ))
    return RowLayout(len, len - 1, len, slot, meshat)
end

# A row position or a literal, resolved against the row
_at(p::Fixed, r) = p.v
_at(p, r) = r[p]

# The slot of z a row constrains
_slotof(d, L::RowLayout) = map(p -> _at(p, d), L.slot)

# The mesh a row belongs to: the entry its block's mesh axis names, or the one it is pinned to
_meshof(d, L::RowLayout) = _at(L.m, d)

# Position of the base row a given row of the iterator feeds at collocation point k
function _basepos(pos, d, L::RowLayout, k)
    key = (_slotof(d, L)..., d[L.i], k)
    haskey(pos, key) || throw(ArgumentError(
        "add_con_collocation: the iterator has no row at $key; it must cover every " *
        "collocation point of each interval."
    ))
    return pos[key]
end

# ---------- the probe ----------

# A traced row entry carries its own position in its type, so tracing the target once says
# both which block is collocated and which entries of the row name its slot.
_slotpos(::ExaModels.DataIndexed{I, J}, who) where {I, J} = J
_slotpos(v::Integer, who) = Fixed(v)
_slotpos(v, who) = throw(ArgumentError(
    "$who: a slot index must be an entry of the iterator row or a literal, got a $(typeof(v))"
))

# Counts the names the caller's pattern binds, since destructuring asks for them one at a
# time. What it hands back is what a DataSource would, so the trace itself is unchanged.
struct ArityProbe <: ExaModels.AbstractNode
    n::Base.RefValue{Int}
end

@inline function Base.indexed_iterate(p::ArityProbe, idx, start = 1)
    p.n[] = max(p.n[], idx)
    return (ExaModels.DataIndexed(ExaModels.DataSource(), idx), idx + 1)
end

function _probe(gen::Base.Generator, who::Symbol)
    n = Ref(0)
    probe = gen.f(ArityProbe(n))
    probe isa Pair && probe.first isa CollocationSlot || throw(ArgumentError(
        "$who: the generator must yield `z[idxs…] => f`, naming the collocation variable and " *
        "the slot its right-hand side is for"
    ))
    slot = probe.first
    return slot.var, map(v -> _slotpos(v, who), slot.idx), n[]
end

# The two spellings a caller may write, told apart by how many names the pattern binds against
# how long a row of the iterator is: data only, and the mesh is crossed in here, or rows that
# already end in (…, i, k), and it is the caller's. Nothing about a row's contents decides it,
# since (v, c, i, k) and (v, c, d1, d2) are the same length with the same slot positions.
function _crossrows(data, arity, N, K, who::Symbol)
    isempty(data) && return data

    len = length(first(data))
    arity == len + 3 && return vec([(d..., i, k) for d in data, i in 1:N, k in 1:K])
    arity in (len, len + 1) && return data

    throw(ArgumentError(
        "$who: a row of the iterator carries $len $(len == 1 ? "entry" : "entries") and the " *
        "generator binds $arity names, which is neither spelling: bind $(len + 3), naming " *
        "i, k and t, for data rows the mesh is crossed into, or $(len + 1) for rows already " *
        "ending in (…, i, k)."
    ))
end

# t on an adaptive mesh, built at trace time so no node ever enters the data. Rebuilt entry by
# entry rather than splatted, since a traced row defines indexed_iterate but not iterate.
_appendt(r, ::Val{L}, tp, lay) where {L} =
    (ntuple(j -> r[j], Val(L))..., _tat(tp, _meshof(r, lay), r[lay.i], r[lay.k]))

# t under a horizon scale, off the two constants `_trows` left past the row
_scalet(r, ::Val{L}, ts, lay) where {L} =
    (ntuple(j -> r[j], Val(L))..., r[L + 1] + ts[_meshof(r, lay)] * r[L + 2])

# ---------- where h and t come from ----------
#
#                    h                                  t
#   numeric          -h[mi] folds into the constant     appended to the row as data
#   adaptive         the bare weight, h off hpar        built off tpar at trace time
#   unknown horizon  -h[mi]/T folds in, scale off row   t0 + scale*share, share appended as data
#
# Two pairs, each writing into the data row and reading it back at trace time, so a residual site
# names the quantity and the mesh decides the rest. The sign always rides in the constant, so the
# traced factor is whatever is left of h. `mi` is the mesh the row belongs to, the literal 1 for
# one mesh.

# The width folded into a row constant: the whole interval numerically, its share of the nominal
# span under a scale, and unity where a traced factor carries it instead.
_hwidth(m, mi, i) = _is_unknown_horizon(m) ? _hat(_hval(m), mi, i) / _tnom(m, mi) :
    _isadaptive(m) ? one(eltype(_hval(m))) : _hat(_hval(m), mi, i)

# The constant a row stores for a weight `a` at interval `i` of mesh `mi`
_hcoef(m, mi, i, a) = -_hwidth(m, mi, i) * a

# -h*a*expr, off a row carrying that constant at `p`, which on a numeric mesh is all of it
_hterm(m, L, r, p, expr) =
    _is_unknown_horizon(m) ? _horizon_scale(m)[_meshof(r, L)] * r[p] * expr :
    _isadaptive(m) ? _hat(_hpar(m), _meshof(r, L), r[L.i]) * r[p] * expr :
    r[p] * expr

# The rows ExaModels stores. A numeric mesh carries t as data, an unknown horizon its start and
# the relative share, both constants, since a traced index cannot reach into a plain array.
_trows(m, itr, L) = _is_unknown_horizon(m) ? [_share(m, d, L) for d in itr] :
    _isadaptive(m) ? itr :
    [(d..., _tat(_tval(m), _meshof(d, L), d[L.i], d[L.k])) for d in itr]

_share(m, d, L) = (mi = _meshof(d, L);
    (d..., _t0(m, mi), (_tat(_tval(m), mi, d[L.i], d[L.k]) - _t0(m, mi)) / _tnom(m, mi)))

# The right-hand side read off one of those rows
_rhsof(m, gen, L) = _is_unknown_horizon(m) ?
    (r -> last(gen.f(_scalet(r, Val(L.len), _horizon_scale(m), L)))) :
    _isadaptive(m) ? (r -> last(gen.f(_appendt(r, Val(L.len), _tpar(m), L)))) :
    (r -> last(gen.f(r)))

"""
    add_con_collocation(core, z[dims...] => generator; name = nothing, kwargs...)

Adds the collocation constraints for the `CollocationVariable` to `core`, enforcing `dz/dt = f`
at every collocation point of the mesh in [`CollocationExaCore`](@ref).
Returns `(core, Constraint)`.

# Arguments
- `z` : a `CollocationVariable` from [`add_var_collocation`](@ref)
- `dims...` : the indices for `CollocationVariable` over which the collocation constraints are added
- `generator` : right-hand side function `f` for a `CollocationVariable`

# Keyword Arguments
- `name` : when given as `Val(:name)`, registers the constraint in `core` for later retrieval as `core.name`. See [`@add_con_collocation`](@ref) for the idiomatic named interface.
- remaining kwargs passed on to `ExaModels.add_con`: `lcon`, `ucon`, `start`, `tag`

# Example
```julia
julia> c, z = add_var_collocation(c, 1:Nz, 1:Nexp)

julia> c, rate = ExaModels.add_var(c, 1:Nz)

julia> itr = [(v, exp) for v in 1:Nz, exp in 1:Nexp]

julia> c, coll = add_con_collocation(c,
           z[v,exp] => -rate[v]*z[v,exp,i,k] + rate[v]*cos(t) # right-hand side function expression added for z[v,exp]
           for (v, exp, i, k, t) in itr) # (i, k) appended to itr autmoatically, t also if adaptive = false
```
"""
function add_con_collocation(
        core::CollocationExaCore,
        gen::Base.Generator;
        name = nothing,
        kwargs...,
    )
    z, slot, arity = _probe(gen, :add_con_collocation)
    nlead, K = _block_layout(core, z, :add_con_collocation)
    length(slot) == nlead || throw(ArgumentError(
        "add_con_collocation: that block declares $nlead dimensions, so its slot reads " *
        "z[$(join(fill("…", nlead), ", "))]; the target given carries $(length(slot))."
    ))

    mesh, w = _mesh(core), _weights(core)
    itr = _crossrows(
        _flat(gen.iter), arity, _nintervals(core), K, :add_con_collocation,
    )
    L = _row_layout(itr, slot, _meshpos(z, slot), :add_con_collocation)

    rows = _trows(mesh, itr, L)
    f = _rhsof(mesh, gen, L)
    nrow = length(first(rows))

    local con
    if _isstateform(core)
        # 10.7. Base rows carry -h f; the weights ride on the state, j = 0,...,K. The h constant
        # is appended past the row in plain Julia before any tracing, so f reads its own row and
        # leaves the entry past it alone.
        base = [(r..., _hcoef(mesh, _meshof(r, L), r[L.i], one(eltype(w.A)))) for r in rows]
        rhs = r -> _hterm(mesh, L, r, nrow + 1, f(r))
        core, con = ExaModels.add_con(
            core, Base.Generator(rhs, base); name = name, kwargs...,
        )

        st = vec([
            (n, w.A[j + 1, d[L.k]], _slotof(d, L)..., d[L.i], j)
            for (n, d) in enumerate(itr), j in 0:K
        ])
        core, _ = ExaModels.add_con!(core, con, Base.Generator(_state_stencil(z, nlead), st))
    else
        # 10.8. One base row z[...,i,k] - z[...,i,0] per collocation point of each slot.
        base = [(_slotof(d, L)..., d[L.i], d[L.k]) for d in itr]
        core, con = ExaModels.add_con(
            core, Base.Generator(_derivative_base(z, nlead), base);
            name = name, kwargs...,
        )

        # Each row of the iterator is an f_ij, carrying A[j,k] into the base row of every k.
        pos = Dict(r => n for (n, r) in enumerate(base))
        st = vec([
            (rows[n]..., _basepos(pos, d, L, k), _hcoef(mesh, _meshof(d, L), d[L.i], w.A[d[L.k], k]))
            for (n, d) in enumerate(itr), k in 1:K
        ])
        aug = s -> s[nrow + 1] => _hterm(mesh, L, s, nrow + 2, f(s))
        core, _ = ExaModels.add_con!(core, con, Base.Generator(aug, st))
    end

    # Which slots were collocated, and with what f. Continuity reads its rows off this, and
    # under DerivativeForm integrates the same f, so it is recorded for either basis.
    core = _addresidual(core, Residual(z, f, rows, unique(_slotof(d, L) for d in itr), L))
    return core, con
end

# `n` entries of a row from `off`, taken by position because a traced row defines
# indexed_iterate but not iterate. The count is a Val so the stencil is built once, not per row.
_rowidx(r, off, ::Val{n}) where {n} = ntuple(q -> r[off + q], Val(n))

# The stencils that index z. A block declared with no dimensions is z[i,k], so its rows carry
# no slot at all, which _rowidx reads as the empty tuple.
# n => a * z[slot..., i, j], off a row (n, a, slot..., i, j)
function _state_stencil(z, nlead)
    V = Val(nlead + 2)
    return s -> s[1] => s[2] * z[_rowidx(s, 2, V)...]
end

# z[slot..., i, k] - z[slot..., i, 0], off a row (slot..., i, k)
function _derivative_base(z, nlead)
    V, S = Val(nlead + 2), Val(nlead + 1)
    return r -> z[_rowidx(r, 0, V)...] - z[_rowidx(r, 0, S)..., 0]
end

"""
    @add_con_collocation(core, [name,] z[dims...], generator; kwargs...)

Macro interface for [`add_con_collocation`](@ref). Updates `core` in the calling scope.

- **Named** (`@add_con_collocation(core, name, z[dims...], generator)`): binds `name` to the new
  `Constraint` in the local scope and registers it in `core` for later retrieval as `core.name`
  or `model.name`.
- **Anonymous** (`@add_con_collocation(core, z[dims...], generator)`): equivalent to
  `c, name = add_con_collocation(c, z[dims...] => generator)`.

Accepts the same keyword arguments as [`add_con_collocation`](@ref).

# Example
```julia
julia> @add_var_collocation(c, z, 1:4)

julia> @add_var_collocation(c, u, 1:3; include_boundary = false)

julia> itr = [(v, l[v]) for v in 1:4]

julia> @add_con_collocation(c, coll, z[v],
           z[v]*u[l]*cos(t) # right-hand side function expression added for z[v], can freely use t
           for (v, l) in itr) # automatically iterated over all N,K with t included
```
"""
macro add_con_collocation(exs...)
    args, kwargs = _split_collocation_args(exs)
    length(args) in (3, 4) ||
        error("@add_con_collocation requires core, an optional name, a target, and a right-hand side")

    core = args[1]
    name, target, rhs = length(args) == 4 ? (args[2], args[3], args[4]) : (nothing, args[2], args[3])
    name === nothing || name isa Symbol ||
        error("@add_con_collocation: the second argument must be the constraint name")

    gen = _collocation_generator(core, target, rhs)
    con = gensym(:con)

    return quote
        local $con
        $(esc(core)), $con = add_con_collocation(
            $(esc(core)),
            $(esc(gen));
            name = $(_name_val(name)),
            $(map(esc, kwargs)...),
        )
        $(name === nothing ? con : :($(esc(name)) = $con))
        $con
    end
end

# The generator the function is given: the target moved inside where its indices are bound,
# the mesh crossed in when the caller left it out, and every operand completed against i, k.
function _collocation_generator(core, target, rhs)
    target isa Expr && target.head === :ref || (target = Expr(:ref, target))
    rhs isa Expr && rhs.head === :flatten &&
        error("@add_con_collocation: write the iterator as `for … in …, … in …`, not nested")
    body, clauses = rhs isa Expr && rhs.head === :generator ?
        (rhs.args[1], rhs.args[2:end]) : (rhs, Any[])

    names = isempty(clauses) ? Symbol[] :
        reduce(vcat, [_pattern_names(c.args[1]) for c in clauses])
    :t in names && error(
        "@add_con_collocation: t is already available in the generator, so it does not " *
        "belong in the iterator"
    )
    hasi, hask = :i in names, :k in names
    hasi == hask || error(
        "@add_con_collocation: give the iterator both i and k or neither, not one of them"
    )
    !hasi || names[(end - 1):end] == [:i, :k] || error(
        "@add_con_collocation: i and k must be the last two entries of the iterator row"
    )

    # Rows the function is handed, always ending (…, i, k)
    mesh = hasi ? Any[] : Any[Expr(:(=), :i, :(1:($(core)).N)), Expr(:(=), :k, :(1:($(core)).K))]
    row = Expr(:tuple, names..., (hasi ? () : (:i, :k))...)
    rows = Expr(:comprehension, Expr(:generator, row, clauses..., mesh...))

    pattern = Expr(:tuple, row.args..., :t)
    skip = Set{Symbol}([names..., :i, :k, :t])
    return Expr(
        :generator,
        Expr(:call, :(=>), target, _complete(body, skip)),
        Expr(:(=), pattern, rows),
    )
end
