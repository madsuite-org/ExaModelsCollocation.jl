# Continuity across interval junctions, for i = 1,...,N-1.
#
# Like the collocation residual, this takes a structurally different form per basis
# (Biegler, Nonlinear Programming, Ch. 10.2.1), with f_ij = f(z[...,i,j], t[i,j]):
#
#   StateForm      (10.14a)  sum_{j=0..K} b[j] z[...,i,j]  =  z[...,i+1,0]
#   DerivativeForm (10.15a)  z[...,i+1,0] - z[...,i,0]     =  h[i] sum_{j=1..K} b[j] f_ij
#
# StateForm evaluates the state polynomial at tau = 1, so it needs no right-hand side and one
# call covers every component and condition. DerivativeForm integrates f across the interval
# -- it is the Runge-Kutta step -- so it needs the same generator the collocation call took,
# and the same rule applies: one call per structurally distinct right-hand side.

"""
    continuity_itr(dae, leads...)

Convenience constructor for the iterator a `StateForm` [`@add_con_continuity`](@ref)
generator runs over: flat tuples `(leads...)`, one per state slot to be tied across
junctions. No mesh data rides along, because that residual needs none.

The junction index is **not** in the tuple. Which intervals get tied is fixed by the mode,
not by the caller, so the macro crosses these slots with `i = 1,…,N-1` itself, the same way
`@add_con_collocation` attaches `h`. Nothing is left for the generator body to write but the
state slice.

`DerivativeForm` continuity integrates the right-hand side across the interval, so it runs
over a [`collocation_itr`](@ref) instead and derives its own junction rows.

As with `collocation_itr`, any iterator of lead tuples works just as well.
"""
function continuity_itr(dae::DAEta, leads...)
    _require_mesh(dae, :continuity_itr)
    return vec([lead for lead in Iterators.product(leads...)])
end

"""
    @add_con_continuity(core, dae, name, z[leads...] for (leads...) in itr; kwargs...)
    @add_con_continuity(core, dae, name, z[leads...], generator; kwargs...)

Add the continuity constraints tying each interval's terminal polynomial value to the next
interval's boundary node.

Which of the two forms applies is set by `dae.mode`:

- **`StateForm`** (the first) evaluates the state polynomial at `τ = 1`,
  `Σⱼ b[j] z[…,i,j] = z[…,i+1,0]`. The residual is fixed entirely by the mode, so the
  generator body is just the state slice, the iterator says only *which slots* to tie, and
  one call covers everything. The junction index is crossed in by the macro:

  ```julia
  @add_con_continuity(core, dae, cont,
      z[v,c] for (v,c) in continuity_itr(dae, 1:Nz, 1:Nc))
  ```

- **`DerivativeForm`** (the second) integrates the right-hand side across the interval,
  `z[…,i+1,0] - z[…,i,0] = h[i] Σⱼ b[j] f(z[…,i,j], t[i,j])`. That needs the same slice and
  generator the matching [`@add_con_collocation`](@ref) call took, over a
  [`collocation_itr`](@ref) — junction rows are derived from it:

  ```julia
  @add_con_continuity(core, dae, cont1, z[1,c],
      z[2,c,i,k] for (c,i,k,t) in collocation_itr(dae, 1:Nc))
  ```

Passing the wrong one for the active mode is an error, not a silently different model.

Updates `core` and `dae` in the calling scope and binds `name` to the new constraint.
"""
macro add_con_continuity(exs...)
    args, kwargs = _split_collocation_args(exs)
    length(args) in (4, 5) || error(
        "@add_con_continuity requires core, dae, a name, and either a `z[...] for (...) in " *
        "itr` generator (StateForm) or a state slice plus a right-hand side generator " *
        "(DerivativeForm)"
    )
    core, dae, name = args[1], args[2], args[3]
    name isa Symbol ||
        error("@add_con_continuity: the third argument must be the constraint name")

    return if length(args) == 4
        _continuity_stateform(core, dae, name, args[4], kwargs)
    else
        _continuity_derivativeform(core, dae, name, args[4], args[5], kwargs)
    end
end

# 10.14a: sum_{j=0..K} b[j] z[...,i,j] - z[...,i+1,0] = 0, over (leads...) x i = 1,...,N-1.
# The caller's iterator names only the slots to tie; the junction index is the mode's
# business, so it is crossed in here under a gensym the caller never sees or writes.
function _continuity_stateform(core, dae, name, gen, kwargs)
    slice, loopvars, itrexpr = _parse_gen(gen, "@add_con_continuity")
    zsym, leads = _parse_slice(slice, "@add_con_continuity")

    itr, rows, b, K, N, con, st, n, j, w, ivar =
        gensym.((:itr, :rows, :b, :K, :N, :con, :stencil, :n, :j, :w, :i))

    # (leads..., i), with i bound to the macro's own name rather than one of the caller's
    pattern = Expr(:tuple, map(esc, loopvars)..., ivar)
    zref(idxs...) = :($(esc(zsym))[$(map(esc, leads)...), $(idxs...)])

    return quote
        local $itr = $(esc(itrexpr))
        local $b = _weights($(esc(dae))).b
        local $K = _degree($(esc(dae)))
        local $N = _nintervals($(esc(dae)))
        _check_slice($(esc(dae)), $(esc(zsym)), $(length(leads)), :add_con_continuity)
        _isstateform($(esc(dae))) || throw(ArgumentError(
            "@add_con_continuity: $(_mode($(esc(dae))).basis) continuity integrates the " *
            "right-hand side, so it needs the state slice and generator the matching " *
            "@add_con_collocation call took"
        ))

        local $rows = vec([(row..., $ivar) for row in $itr, $ivar in 1:($N - 1)])

        local $con
        $(esc(core)), $con = ExaModels.add_con(
            $(esc(core)),
            (
                -$(zref(:($ivar + 1), 0))
                for $pattern in $rows
            );
            name = $(Val(name)),
            $(map(esc, kwargs)...),
        )

        local $st = vec([
            ($n, d..., $j, $b[$j + 1]) for ($n, d) in enumerate($rows), $j in 0:$K
        ])
        $(esc(core)), _ = ExaModels.add_con!(
            $(esc(core)), $con,
            $n => $w * $(zref(ivar, j))
            for $(Expr(:tuple, n, map(esc, loopvars)..., ivar, j, w)) in $st
        )

        _register!($(esc(dae)), :cons, $(QuoteNode(name)), $con)
        $(esc(name)) = $con
    end
end

# 10.15a: z[...,i+1,0] - z[...,i,0] - h[i] sum_{j=1..K} b[j] f_ij = 0.
# The generator is the collocation one, over (leads..., i, k, t); the N-1 junction rows are
# derived from its leading dimensions, and the stencil rebinds the loop variables to each
# collocation point j of interval i so the same expression means f_ij.
function _continuity_derivativeform(core, dae, name, slice, gen, kwargs)
    zsym, leads = _parse_slice(slice, "@add_con_continuity")
    rhs, loopvars, itrexpr = _parse_gen(gen, "@add_con_continuity")

    nv = length(loopvars)
    nv >= 3 || error(
        "@add_con_continuity: the iterator must yield (leads..., i, k, t); " *
        "build it with `collocation_itr(dae, leads...)`"
    )
    nlead, ipos = nv - 3, nv - 2
    junctionvars = loopvars[1:(nlead + 1)]            # (leads..., i)
    ivar = loopvars[ipos]

    itr, w, mesh, K, N, con, rows, st, n, j, a =
        gensym.((:itr, :w, :mesh, :K, :N, :con, :rows, :stencil, :n, :j, :a))

    zref(ielem) = :($(esc(zsym))[$(map(esc, leads)...), $ielem, 0])

    return quote
        local $itr = $(esc(itrexpr))
        local $w = _weights($(esc(dae)))
        local $mesh = _mesh($(esc(dae)))
        local $K = _degree($(esc(dae)))
        local $N = _nintervals($(esc(dae)))
        _check_slice($(esc(dae)), $(esc(zsym)), $(length(leads)), :add_con_continuity)
        _isstateform($(esc(dae))) && throw(ArgumentError(
            "@add_con_continuity: StateForm continuity is fixed by the mode and takes no " *
            "right-hand side; pass the state slice alone as the generator body"
        ))

        # One junction row per distinct lead tuple and interval boundary
        local $rows = vec([
            (lead..., i)
            for lead in unique([row[1:$nlead] for row in $itr]), i in 1:($N - 1)
        ])

        local $con
        $(esc(core)), $con = ExaModels.add_con(
            $(esc(core)),
            (
                $(zref(:($(esc(ivar)) + 1))) - $(zref(esc(ivar)))
                for $(esc(_tuple(junctionvars))) in $rows
            );
            name = $(Val(name)),
            $(map(esc, kwargs)...),
        )

        local $st = vec([
            (
                $n, r[1:$nlead]..., r[$nlead + 1], $j, $mesh.t[r[$nlead + 1], $j],
                -$mesh.h[r[$nlead + 1]] * $w.b[$j],
            )
            for ($n, r) in enumerate($rows), $j in 1:$K
        ])
        $(esc(core)), _ = ExaModels.add_con!(
            $(esc(core)), $con,
            $n => $a * $(esc(rhs))
            for $(Expr(:tuple, n, map(esc, loopvars)..., a)) in $st
        )

        _register!($(esc(dae)), :cons, $(QuoteNode(name)), $con)
        $(esc(name)) = $con
    end
end
