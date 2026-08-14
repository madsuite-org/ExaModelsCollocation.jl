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
# i and k trail, t is appended past them, and where the slot indices sit is the probe's to say.
struct RowLayout{S}
    len::Int     # length of the row, one short of a numeric mesh's rows
    i::Int       # position of the interval index
    k::Int       # position of the collocation index
    slot::S      # per declared dimension: a row position, or a Fixed literal
end

function _row_layout(itr, slot, who::Symbol)
    isempty(itr) && throw(ArgumentError(
        "$who: the iterator is empty, so there is no collocation constraint to add."
    ))
    len = length(first(itr))
    len >= 2 || throw(ArgumentError(
        "$who: the iterator rows must end in (…, i, k); the ones given carry $len entries."
    ))
    return RowLayout(len, len - 1, len, slot)
end

# The slot of z a row constrains
_slotof(d, L::RowLayout) = map(p -> p isa Fixed ? p.v : d[p], L.slot)

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
_appendt(r, ::Val{L}, tp, ip, kp) where {L} = (ntuple(j -> r[j], Val(L))..., tp[r[ip], r[kp]])

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
    hp, tp = _hpar(mesh), _tpar(mesh)
    itr = _crossrows(
        _flat(gen.iter), arity, _nintervals(core), K, :add_con_collocation,
    )
    L = _row_layout(itr, slot, :add_con_collocation)

    # The rows ExaModels stores, and the right-hand side read off one of them. A numeric mesh
    # carries t as data, an adaptive one indexes it at trace time.
    rows = tp === nothing ? [(d..., _tval(mesh)[d[L.i], d[L.k]]) for d in itr] : itr
    f = tp === nothing ? (r -> last(gen.f(r))) :
        (r -> last(gen.f(_appendt(r, Val(L.len), tp, L.i, L.k))))
    nrow = L.len + (tp === nothing ? 1 : 0)

    local con
    if _isstateform(core)
        # 10.7. Base rows carry -h f; the weights ride on the state, j = 0,...,K. On a
        # numeric mesh h is appended past the row, in plain Julia before any tracing, so f
        # reads its own row and leaves the entry past it alone; on an adaptive one it is
        # indexed off the parameter block by the row's own i.
        base = hp === nothing ? [(r..., _hval(mesh)[r[L.i]]) for r in rows] : rows
        rhs = hp === nothing ? (r -> -r[nrow + 1] * f(r)) : (r -> -hp[r[L.i]] * f(r))
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
        st = hp === nothing ?
            vec([
                (rows[n]..., _basepos(pos, d, L, k), -_hval(mesh)[d[L.i]] * w.A[d[L.k], k])
                for (n, d) in enumerate(itr), k in 1:K
            ]) :
            vec([
                (rows[n]..., _basepos(pos, d, L, k), w.A[d[L.k], k])
                for (n, d) in enumerate(itr), k in 1:K
            ])
        aug = hp === nothing ?
            (s -> s[nrow + 1] => s[nrow + 2] * f(s)) :
            (s -> s[nrow + 1] => -hp[s[L.i]] * s[nrow + 2] * f(s))
        core, _ = ExaModels.add_con!(core, con, Base.Generator(aug, st))
    end

    # Which slots were collocated, and with what f. Continuity reads its rows off this, and
    # under DerivativeForm integrates the same f, so it is recorded for either basis.
    core = _addresidual(core, Residual(z, f, rows, unique(_slotof(d, L) for d in itr), L))
    return core, con
end

# The stencils that index z, one per leading-dimension count _block_layout admits. A block
# declared with no dimensions is z[i,k], so its rows carry no slot at all.
# n => a * z[slot..., i, j], off a row (n, a, slot..., i, j)
_state_stencil(z, nlead) = nlead == 0 ?
    (s -> s[1] => s[2] * z[s[3], s[4]]) :
    nlead == 1 ?
    (s -> s[1] => s[2] * z[s[3], s[4], s[5]]) :
    (s -> s[1] => s[2] * z[s[3], s[4], s[5], s[6]])

# z[slot..., i, k] - z[slot..., i, 0], off a row (slot..., i, k)
_derivative_base(z, nlead) = nlead == 0 ?
    (r -> z[r[1], r[2]] - z[r[1], 0]) :
    nlead == 1 ?
    (r -> z[r[1], r[2], r[3]] - z[r[1], r[2], 0]) :
    (r -> z[r[1], r[2], r[3], r[4]] - z[r[1], r[2], r[3], 0])

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
