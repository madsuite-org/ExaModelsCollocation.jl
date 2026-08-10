# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

The README is the API reference — signatures, arguments, keywords, examples. This file is only
what the README and the source do not already say: why the code is shaped this way, and the
traps that fail silently.

## Scope

An **internal helper module**, not a package that owns your problem: no single entry point, no
dimension inference, no initialization, no integrator, no solver wrapper. You assemble the model
with `ExaModels.add_*` and reach for a helper only where collocation actually changes something.

- Dependencies are **ExaModels** and **FastGaussQuadrature**. Do not add others.
- **One file per exported function, named after it.** Do not grow past the three helpers.
- Initial, terminal, and path conditions carry no collocation content — plain `ExaModels.@add_con`.
- Keep the README short and shaped like ExaModels' own `add_var` docs. Rationale, derivations,
  and Biegler cross-references live here, in `src` comments, or in `test/`.

## The discretization

`N` counts intervals, not boundaries: `h = diff(nodes)`, so 21 nodes give `N = 20` and `t` is
`20 × K`. Biegler's indexing is used throughout, including where it reuses letters: `j` indexes
the basis polynomial being summed (and the mesh, `t[i,j]`), `k` the collocation point being
enforced.

**The basis changes the structure of the equations, not just the weights**, and both forms use
the *same variables*: `ż_{i,j}` is never a decision variable, it is the right-hand side evaluated
at `z[…,i,j]`, and Biegler's separate interval-entry coefficient `z_{i-1}` is just the `k = 0`
index. With `f_ij = f(z[…,i,j], t[i,j])`:

| | collocation | continuity |
|---|---|---|
| `StateForm` | (10.7) `Σⱼ₌₀..ᴷ A[j,k] z[…,i,j] = h[i] f_ik` | (10.14a) `Σⱼ₌₀..ᴷ b[j] z[…,i,j] = z[…,i+1,0]` |
| `DerivativeForm` | (10.8) `z[…,i,k] − z[…,i,0] = h[i] Σⱼ₌₁..ᴷ A[j,k] f_ij` | (10.15a) `z[…,i+1,0] − z[…,i,0] = h[i] Σⱼ₌₁..ᴷ b[j] f_ij` |

`StateForm` puts the state under the weights and evaluates `f` once per row; `DerivativeForm`
puts `f` under the weights and re-evaluates it at every collocation point, which is what makes
it the implicit Runge-Kutta step (`A` the Butcher tableau, `b` its quadrature weights).

**When the family collocates `τ = 1` — Radau, Lobatto — continuity is neither of the above.** It
is `z[…,i,K] = z[…,i+1,0]`, and `add_con_continuity` returns early with no `add_con!` layer. Two
identities make it exact, not approximate, because `polynomial.jl` short-circuits at an exact
node hit: under `StateForm` the basis is cardinal there (`b == [0,…,0,1]`), and under
`DerivativeForm` `A[:,K] == b`, so 10.15a minus the `k = K` collocation row leaves it. The second
is a **row operation**, valid only because the collocation rows exist and use the same `f` —
which is what makes `_covered_slots` load-bearing rather than merely defensive.

**The caller writes continuity once either way.** `add_con_collocation` records its rows and
right-hand side as a `Residual`, and continuity reads the slots (and, under `DerivativeForm`, the
`f`) back off that record, so the call is the same for either basis: name the variable, and the
mode decides. Keep that property — a second spelling of continuity is what this replaced. The
record is also what makes the ordering real: collocation comes first, and a slot with no residual
behind it throws.

**In `DerivativeForm` a row of the caller's iterator is an `f_ij`, not an `f_ik`.** 10.7 wants one
right-hand side per row; 10.8 wants `K` of them summed into each row, and the iterator supplies
those, so its mesh index is the summation index. Both helpers carry the row's weight to where it
belongs rather than rebuilding it at another point — that is what lets one written expression
serve either mode.

Under Lobatto, `τ₁ = 0` coincides with the `k = 0` index, so the `k = 1` row reduces to
`z[…,i,1] = z[…,i,0]`: one redundant variable and one trivial equation per interval. Consistent,
and the observed order is unaffected.

## ExaModels idioms this module must follow

`ExaModelsPEtab.jl/src/nlp/collocation.jl` and `nlp/continuity.jl` are the reference
implementation of this exact pattern; read them before changing a helper.

1. **Mirror the upstream signature.** `(core, dims_or_gen...)`, name as an optional
   `name = Val(:z)` **keyword** — never a positional `Symbol` — returning `(core, handle)`.
   Nothing is threaded alongside `core`; it carries the discretization itself.
2. **Iterators are flat arrays of tuples, destructured in the generator**: `for (v,c,i,k,t) in
   itr`. Not NamedTuples with `d.field`.
3. **Traced loop indices cannot index a plain Julia array.** `A[j+1,k]` or `t[i,k]` inside a user
   expression throws `invalid index … of type DataIndexed`. Numeric data must ride in the
   iterator tuple — hence the caller's row carrying `t`. Data the *caller* never references stays
   out: the helper appends `h` itself, before tracing.
4. **`add_con!` keys rows by position in the base iterator, not by index value.** A row's leading
   indices may name a restricted slice (`2:2`) where the two diverge, so base iterators are
   `vec`'d flat (`_flat`, in the helper — a caller never writes `vec`) and augmentations are
   keyed by linear position. Getting this wrong yields an INFEASIBLE model, not an error.
5. **One `add_con` call per structurally distinct algebraic expression, and no more.** Everything
   that merely *varies* goes in the iterator, including the value a constraint equals. **A block
   sets the floor**: one call cannot span two blocks, so splitting the state costs a call per
   block on everything that was uniform across it.
6. **One `add_con!` generator element is one term, not a sum.** Elements sharing a row index are
   accumulated by ExaModels, which is how a `Σⱼ` becomes a single SIMD kernel. Never build the
   summation inside a traced lambda.
7. **The stencil lives in the function; the macro adds nothing to it.** Rows read
   `(z's own indices…, whatever else f varies with…, i, k, t)`; `RowLayout` locates the slot and
   the mesh entries by counting from both ends, so the middle is the caller's to order.
   `_block_layout` admits **zero to two** leading dimensions and every stencil has a branch per
   count — a block with none has no slot at all, and its one slot is the empty tuple that
   `Iterators.product()` already yields. Look blocks up by handle identity, not by `Symbol`.
   The caller builds the iterator: a product helper cannot express a row like `(v, l[v], i, k, t)`
   whose later entries depend on the leading ones (`test/bruno.jl`'s `sweep`).

`add_var` admits an `Integer` dimension as well as a `UnitRange`, so `_dimrange` normalizes them
before the block stores them — `_covered_slots` enumerates `dims`, and over an `Integer` it would
see the single slot `n` and pass vacuously.

The leading dimensions are the caller's to name — the helpers impose no meaning on them, and
neither should the README.

## The container contract

`CollocationExaCore(nodes, K)` fixes the mesh and mode **once**; every helper reads them off the
core it is already given.

**It is an `ExaCore`, not a wrapper around one.** `ExaCore` is concrete and cannot be subtyped;
the extension point is its `tag` parameter, following ExaModels' own `two_stage.jl`. Two things
follow, and both are the point: every `ExaModels.add_*` dispatches on `ExaCore{T}` so it works
here **unforwarded** — one object mixes plain and collocation constraints — and `ExaModel(c)`
passes `c.tag` through, so the mesh survives into the model, which is what `set_nodes!` needs
after a solve. `LegacyExaCore` is mutable only for the deprecated API; do not copy it.

**The alias drops `ExaCore`'s own `VT <: AbstractVector{T}` bound.** A method written
`c::CollocationExaCore` is therefore *ambiguous* with ExaModels' `getproperty`/`show` over
`E <: Union{ExaCore, ExaModel}` — and it fails at the first `add_var`, not at load. Restate the
bound in the `where` clause, as `two_stage.jl` does on every one of its methods:

```julia
Base.getproperty(c::CollocationExaCore{T,VT,B}, name::Symbol) where {T,VT<:AbstractVector{T},B}
```

**`blocks` and `residuals` are `Vector`s grown in place**, so the tag's type is fixed at
construction — as `two_stage.jl` does with `var_scen`. The consequence is real: a core and every
core derived from it **share one tag**, so a block added to the later one is visible on the
earlier one. Only the `ExaCore` fields are copy-on-update.

**Named handles are not tracked here.** `add_var(c, …; name = Val(:z))` already registers into the
inner `refs`, which `ExaModel` carries over, so `c.z` and `model.z` fall out of forwarding
`getproperty`. `add_var_collocation` only swaps the `CollocationVariable` in for the bare
`Variable` afterwards, through `_rehandle`.

`N`, `K`, `nodes`, `adaptive`, and the `mode` fields are **derived** through `getproperty` — do
not add them back as stored fields. Internally use `_mesh`, `_mode`, `_weights`, `_degree`,
`_nintervals`, and `getfield` for real fields.

**`CollocationVariable <: ExaModels.AbstractVariable` carries its own layout** (`dims`, `krange`),
so there is no side table and "was this created by `add_var_collocation`?" is a type check. This
works because ExaModels writes every indexing method generically over `AbstractVariable` and
because `core.var` is inert bookkeeping the evaluator never reads. **Constraints are the
opposite**: `offset0(::Constraint, i)` and `_constraint_dims(::Constraint)` are typed concretely
and `core.cons` *is* read in the evaluator hot loop, so `add_con_collocation` returns the plain
`Constraint` and the record stays a separate `Residual`. Do not wrap it.

`_split_collocation_args` accepts both `f(a; k = v)` and `f(a, k = v)` — reuse it for new macros.

## The adaptive mesh

`h` and `t` are the only t-space data the residuals read — `A`, `b`, `taus` are τ-space and do not
move — so they are the only two an adaptive mesh has to make mutable. Under `adaptive = true` each
is *additionally* allocated as an `add_par` block; the numeric arrays stay, because bounds,
`start`, and initialization by integration need plain numbers.

**Both paths ship, and `adaptive` picks.** The numeric path is not dead weight: it folds
`-h[i]·A[j,k]` into one constant, where the parameter path allocates `N·K + N` parameters and
traces a product of two symbolic terms.

**A graph node cannot ride in an iterator tuple.** `add_con` and `ExaModel` accept a
`Vector{Tuple{Int,ParameterNode}}` silently and then `cons!` throws `Cannot convert Node2 to
Float64` — a build-time silence, so it is worth stating twice. That is why an adaptive row ends
`(…, i, k)` and a numeric one ends `(…, i, k, t)`: with a parameter mesh the caller indexes
`mesh.tpar[i,k]` inside the right-hand side instead of destructuring `t`.

`set_nodes!` redistributes a **fixed** number of intervals; adding one changes the variable count
and needs a rebuild.

## Testing

Tests must **exercise every build path**: each roots family × each basis × a range of `K`.

**Four files, and keep them to four.** A new helper gets a testset in `api.jl`, not a file of its
own.

`test/collocation.jl` — the mode math. Check weights against the identities that define them (a
Lagrange weight vector applied to the nodal values of a polynomial of degree `≤ length(nodes)-1`
reproduces exactly the operation it encodes, so monomials pin every entry) rather than against
stored numbers. Assert the shapes too; that is what catches a weight matrix wrong for its basis.

`test/api.jl` — every exported helper, closing with both residual forms on `dz/dt = -z` solved
through them, using invariants a fixed tolerance would miss:

- **The bases agree.** Algebraically equivalent discretizations over the same variables must reach
  the same solution to solver tolerance.
- **The order is right.** Terminal error decays at `O(h²ᴷ)` for Legendre, `O(h²ᴷ⁻¹)` for Radau,
  `O(h²ᴷ⁻²)` for Lobatto. A mis-weighted stencil can still converge; it loses order.
- **The interfaces and the meshes pin to each other.** Macros vs. functions must give identical
  residuals row for row; `adaptive = true` must *reproduce* the numeric mesh, not merely come
  close; after `set_nodes!` the model must match one rebuilt from scratch — which is what catches
  `h` moving without `t`.

**The two whole models are deliberately opposite in how they declare the state**, so both
spellings keep working. `test/vanderpol.jl` is **one block per state** — `z1[i,k]` with no
declared dimensions, covering the zero-leading-dimension branch of every stencil, and showing the
cost of splitting. `test/bruno.jl` is **one block for everything**, `z[v,c,i,k]`, and is the only
regression test for `start` on `add_var_collocation` and for the multi-residual path: seven
species obey four structurally distinct expressions, so four `@add_con_collocation` calls over
disjoint slots feed a single `@add_con_continuity`. Its objective is pinned to the negative
log-likelihood PEtab.jl reports at the collection's nominal parameters, `-46.68818145`, to
`atol = 1e-6`; that one assertion covers the whole discretization, so do not loosen it to make an
unrelated change pass.

`ExaModels.solution(result, z)` returns a plain 1-based array, so a block indexed `k = 0,…,K`
lands on `1,…,K+1` there.

```julia
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. -e 'using ExaModelsCollocation'
```

Julia compat is `1.11+`, the floor ExaModels itself sets (CI tests 1.11, 1.12, and `pre`);
ExaModels is pinned to `0.11`. Test-only dependencies go in `test/Project.toml`.
