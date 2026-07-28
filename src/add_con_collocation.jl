# Collocation residual, as a thin wrapper over ExaModels.add_con.
#
# Relative to add_con, this does three things and nothing else:
#   1. puts the supplied expression into the residual form of the DAEta's CollocationMode
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
# puts f under the weights and evaluates it at every collocation point of the element. zdot
# is never a variable in either -- it is f evaluated there -- so add_var_collocation
# allocates the same block for both, and only the residual changes.

"""
    collocation_itr(dae, leads...)

Convenience constructor for the iterator an [`@add_con_collocation`](@ref) generator runs
over. Elements are flat tuples, destructured in the generator body the way ExaModels
iterators normally are:

    (leads..., i, k, t)

`leads` are whichever of the block's own dimensions vary across this constraint — dimensions
pinned in the state slice are simply left out. `i` and `k` run over the elements and
collocation points. `t` is the collocation time `mesh.t[i,k]`, carried in the tuple because a
traced index cannot look it up in a plain array; leave it unused if the right-hand side is
autonomous.

Nothing here is privileged: any iterator whose tuples end in `(…, i, k, t)` works, so build
your own comprehension when you want a different sweep.

## Example

```julia
# z declared over (v, c). Pin v = 1, vary c:
collocation_itr(dae, 1:Nc)          # → (c, i, k, t)

# vary both:
collocation_itr(dae, 1:Nz, 1:Nc)    # → (v, c, i, k, t)
```
"""
function collocation_itr(dae::DAEta, leads...)
    _require_mesh(dae, :collocation_itr)
    mesh = _mesh(dae)
    N, K = _nelements(dae), _degree(dae)
    return vec([
        (lead..., i, k, mesh.t[i, k])
        for lead in Iterators.product(leads...), i in 1:N, k in 1:K
    ])
end

"""
    @add_con_collocation(core, dae, name, z[leads...], generator; kwargs...)

Add the collocation residual for the state slice `z[leads...]`, enforcing `dz/dt = f` at
every collocation point.

Behaves like `ExaModels.@add_con` — the generator supplies the right-hand side `f` and runs
over a [`collocation_itr`](@ref) — except that the expression is wrapped into the residual
form of `dae.mode` and the basis-polynomial sum is attached as an augmentation. The call is
the same for either basis; only what the helper builds from it changes.

The state slice says which variable is being collocated. Its indices may be literals, which
pin that dimension, or names bound by the iterator, which vary with it:

```julia
# one equation for component 1, across every condition
@add_con_collocation(core, dae, coll1, z[1,c],
    z[2,c,i,k] for (c,i,k,t) in collocation_itr(dae, 1:Nc))

# one equation shared by every component, when the right-hand side is uniform in v
@add_con_collocation(core, dae, coll, z[v,c],
    -decay[v] * z[v,c,i,k] for (v,c,i,k,t) in collocation_itr(dae, 1:Nz, 1:Nc))
```

Updates `core` and `dae` in the calling scope and binds `name` to the new constraint.
"""
macro add_con_collocation(exs...)
    args, kwargs = _split_collocation_args(exs)
    length(args) == 5 || error(
        "@add_con_collocation requires core, dae, a name, a state slice z[...], and a generator"
    )
    core, dae, name, slice, gen = args

    name isa Symbol ||
        error("@add_con_collocation: the third argument must be the constraint name")
    zsym, leads = _parse_slice(slice, "@add_con_collocation")
    rhs, loopvars, itrexpr = _parse_gen(gen, "@add_con_collocation")

    # collocation_itr elements are (leads..., i, k, t)
    nv = length(loopvars)
    nv >= 3 || error(
        "@add_con_collocation: the iterator must yield (leads..., i, k, t); " *
        "build it with `collocation_itr(dae, leads...)`"
    )
    nlead, ipos, kpos = nv - 3, nv - 2, nv - 1
    ivar, kvar = loopvars[ipos], loopvars[kpos]

    itr, w, mesh, K, con, st, base, n, j, a, h =
        gensym.((:itr, :w, :mesh, :K, :con, :stencil, :base, :n, :j, :a, :h))

    # z[leads..., i, <k>], with the slice's own leading indices
    zref(kexpr) = :($(esc(zsym))[$(map(esc, leads)...), $(esc(ivar)), $kexpr])

    return quote
        local $itr = $(esc(itrexpr))
        local $w = _weights($(esc(dae)))
        local $mesh = _mesh($(esc(dae)))
        local $K = _degree($(esc(dae)))
        _check_slice($(esc(dae)), $(esc(zsym)), $(length(leads)), :add_con_collocation)

        local $con
        if _isstateform($(esc(dae)))
            # 10.7. Base rows carry -h f; the weights ride on the state, j = 0,...,K.
            # h multiplies the residual and is never named by the caller, so it is appended
            # here rather than carried in the iterator. Plain Julia, before any tracing.
            local $base = [(row..., $mesh.h[row[$ipos]]) for row in $itr]
            $(esc(core)), $con = ExaModels.add_con(
                $(esc(core)),
                (
                    -$h * $(esc(rhs))
                    for $(Expr(:tuple, map(esc, loopvars)..., h)) in $base
                );
                name = $(Val(name)),
                $(map(esc, kwargs)...),
            )

            local $st = vec([
                ($n, d..., $j, $w.A[$j + 1, d[$kpos]])
                for ($n, d) in enumerate($itr), $j in 0:$K
            ])
            $(esc(core)), _ = ExaModels.add_con!(
                $(esc(core)), $con,
                $n => $a * $(zref(j))
                for $(Expr(:tuple, n, map(esc, loopvars)..., j, a)) in $st
            )
        else
            # 10.8. Base rows carry z[...,i,k] - z[...,i,0]; the weights ride on f, which is
            # re-evaluated at each collocation point j = 1,...,K of the same element. The
            # stencil rebinds the caller's loop variables to that point, so the very same
            # expression means f_ij here and f_ik above.
            $(esc(core)), $con = ExaModels.add_con(
                $(esc(core)),
                (
                    $(zref(esc(kvar))) - $(zref(0))
                    for $(esc(_tuple(loopvars))) in $itr
                );
                name = $(Val(name)),
                $(map(esc, kwargs)...),
            )

            local $st = vec([
                (
                    $n, d[1:$nlead]..., d[$ipos], $j, $mesh.t[d[$ipos], $j],
                    -$mesh.h[d[$ipos]] * $w.A[$j, d[$kpos]],
                )
                for ($n, d) in enumerate($itr), $j in 1:$K
            ])
            $(esc(core)), _ = ExaModels.add_con!(
                $(esc(core)), $con,
                $n => $a * $(esc(rhs))
                for $(Expr(:tuple, n, map(esc, loopvars)..., a)) in $st
            )
        end

        _register!($(esc(dae)), :cons, $(QuoteNode(name)), $con)
        $(esc(name)) = $con
    end
end

# Split the state slice AST: `z[v,c]` -> (variable name, [index expressions])
function _parse_slice(ex, who)
    Meta.isexpr(ex, :ref) || error(
        "$who: expected a state slice like `z[v,c]`, got `$ex`"
    )
    return ex.args[1], ex.args[2:end]
end

# `expr for (a,b,...) in itr` -> (expr, [a,b,...], itr)
function _parse_gen(ex, who)
    Meta.isexpr(ex, :generator) || error("$who: expected a generator, got `$ex`")
    spec = ex.args[2]
    Meta.isexpr(spec, :(=)) || error("$who: expected a single `for ... in ...` clause")
    lhs, itr = spec.args
    loopvars = Meta.isexpr(lhs, :tuple) ? lhs.args : [lhs]
    return ex.args[1], loopvars, itr
end

_tuple(vars) = length(vars) == 1 ? vars[1] : Expr(:tuple, vars...)

# The slice must name a block created by add_var_collocation, with the right arity
function _check_slice(dae::DAEta, z, nleads::Integer, who::Symbol)
    nlead, _ = _block_layout(dae, z, who)
    nleads == nlead || throw(ArgumentError(
        "$who: that block declares $nlead leading dimensions, but the slice gives $nleads"
    ))
    return nothing
end
