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
- **One file per exported function, named after it**, under `src/api/`: the three builders
  (`add_var_collocation`, `add_con_collocation`, `add_con_continuity`) plus `set_nodes!` and
  `interpolate`, which move and read the mesh rather than build on it. Do not grow the builders
  past three. `src/api/macros.jl` is only what the three macros share, and `test/` mirrors
  the source directories.
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

`ExaModelsPEtab.jl/src/nlp/collocation.jl` and `nlp/continuity.jl` are the largest caller, and
still on the **pre-probe** spelling: a bare `z` argument, a numeric row ending in `t_ij`, and
`mesh.tpar[i,k]` in the adaptive branch. Porting them is a separate job on branch `emc`. Read
them for what a real caller needs, not for how a call is written.

1. **Mirror the upstream signature.** `(core, dims_or_gen...)`, name as an optional
   `name = Val(:z)` **keyword** — never a positional `Symbol` — returning `(core, handle)`.
   Nothing is threaded alongside `core`; it carries the discretization itself.
2. **Iterators are flat arrays of tuples, destructured in the generator**: `for (v,c,i,k,t) in
   itr`. Not NamedTuples with `d.field`. A row carries what `f` varies with and the pattern
   names `i`, `k` and `t` past it, all three put there by the helper, so a row is never written
   with the mesh in it. Rows that already end `(…, i, k)` are still read as written.
3. **Traced loop indices cannot index a plain Julia array.** `A[j+1,k]` or `t[i,k]` inside a user
   expression throws `invalid index … of type DataIndexed`. Numeric data must ride in the
   iterator tuple — hence `t` being appended to the data on a numeric mesh rather than looked up
   where it is read. Data the *caller* never references stays out: the helper appends `h` itself,
   before tracing.
4. **`add_con!` keys rows by position in the base iterator, not by index value.** A row's slot
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
7. **The stencil lives in the function, and so does the mesh.** Rows read
   `(whatever f varies with…, i, k)` by the time `RowLayout` sees them; it takes `i` and `k` off
   the end and carries the slot positions the probe found, so the rest is the caller's to order
   and the slot indices need not lead. Which spelling the caller wrote is decided by **how many
   names the pattern binds** against how long a row is, `_crossrows` counting the first off
   `ArityProbe`'s `indexed_iterate`: `len + 3` crosses `1:N × 1:K` in, `len` or `len + 1` takes
   the row as written, and `len + 2` is an error rather than a third reading, since contiguous
   bands would let a stray name in an already-crossed pattern re-cross it into a wrong model.
   Nothing about a row's *contents* can decide this, `(v,c,i,k)` and `(v,c,d1,d2)` being the same
   length with the same slot positions. `_block_layout` admits **zero to two** leading dimensions
   and every stencil has a branch per count — a block with none has no slot at all, and its one
   slot is the empty tuple that `Iterators.product()` already yields. Look blocks up by handle
   identity, not by `Symbol`.
8. **The target is traced out of the generator, as `add_con!`'s two-argument form is**
   (`ExaModels/src/nlp.jl`). `z[v,c]` cannot be an argument: `v` and `c` exist only inside the
   generator. Indexing a block by its declared dimensions alone therefore gives a
   `CollocationSlot` rather than a node — the arities never collide, a full index being
   `length(dims) + 2` long — and one `gen.f(ArityProbe(n))` recovers the block, off
   `DataIndexed`'s type parameter which row entries name its slot, and the pattern's arity in
   the same pass. What the probe hands back is what a `DataSource` would, so only the counting
   is added to the trace.
9. **The macro completes operands at trace time, not at expansion time.** It cannot know whether
   a name is a block, so every operand goes through `_colidx`, which appends `(i,k)` only to a
   `CollocationVariable` given exactly its `dims`. This is what makes `z[v]`, `u[l]` and a bare
   zero-dimension block all work, and it costs nothing in the kernel. A caller writing the
   iterator out still can: `i` and `k` are read off the row when the pattern binds them.
   A `ref` mentioning `end` or `begin` is left **as written** (`_indexonly`), since rewriting
   `a[end]` into `_colidx(a, i, k, end)` moves `end` out of index position where it has no
   meaning and fails with `UndefVarError`.
   **`i`, `k` and `t` are reserved inside a right-hand side.** They name the mesh whether or
   not the iterator binds them, so a caller's own value under one of those names is silently
   replaced. Gensyms cannot fix it: the short iterator coexists with `z[v,i,k]` written out in
   the body, so those names must resolve to the mesh there.

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

**`block` and `resid` are `Vector`s of fixed type**, so the tag's type is fixed at
construction — as `two_stage.jl` does with `var_scen`. They are **copied on every add**
(`_addblock`, `_addresidual`, both through `_retag`), so a core does not report what was added
to a sibling derived from the same parent, matching the copy-on-update the `ExaCore` fields
already have. The **mesh stays shared**, which is what lets `set_nodes!` move it for every core
built off it.

**Named handles are not tracked here.** `add_var(c, …; name = Val(:z))` already registers into the
inner `refs`, which `ExaModel` carries over, so `c.z` and `model.z` fall out of forwarding
`getproperty`. `add_var_collocation` only swaps the `CollocationVariable` in for the bare
`Variable` afterwards, through `_rehandle`.

`N`, `K`, `nodes`, `adaptive`, and the `mode` fields are **derived** through `getproperty` — do
not add them back as stored fields. Internally use `_mesh`, `_mode`, `_weights`, `_degree`,
`_nintervals`, and `getfield` for real fields.

**`refs` is consulted before the derived names**, so a handle registered as `Val(:K)` or
`Val(:N)` is what `c.K` and `c.N` give back. Those two are exactly what a model calls its own
variables, and the derived name silently shadowing them was unrecoverable, since a plain
`ExaModels.add_var(c, …; name = Val(:K))` never reaches this package.

**`CollocationVariable <: ExaModels.AbstractVariable` carries its own layout** (`dims`, `krange`),
so there is no side table and "was this created by `add_var_collocation`?" is a type check. This
works because ExaModels writes every indexing method generically over `AbstractVariable` and
because `core.var` is inert bookkeeping the evaluator never reads. **Constraints are the
opposite**: `offset0(::Constraint, i)` and `_constraint_dims(::Constraint)` are typed concretely
and `core.cons` *is* read in the evaluator hot loop, so `add_con_collocation` returns the plain
`Constraint` and the record stays a separate `Residual`. Do not wrap it.

**`Residual.f` called on a row of `Int`s is a numeric right-hand side**, and that is why a second,
numeric one is not stored beside it. Indexing resolves `z[s…,i,k]` to a `Var` carrying its
*absolute* index, so the result is a node over concrete leaves — `node(nothing, x, θ)` evaluates
it, and `z[s…,i,k].i` is the `x` index to write, with no offset arithmetic to redo. Everything the
expression reads that is not a state comes back at its solved value, which a closure stored at
`add_con_collocation` time could not have captured: in `ExaModelsPEtab.jl/src/nlp/collocation.jl`
the expression reads `p`, so a numeric field would need the problem-specific signature
`f(zvals, pvals, cvvals, gvals, t)`. To read `f` *off* the collocation points — what an error
estimate needs — doctor copies of `x` and `θ` at one scratch `k`; `examples/refinement.jl` does
exactly this. Four traps: an autonomous `f` on a numeric mesh traces to a plain `Real`, not a node;
every block must be moved to the new point before any residual is evaluated, since one expression
reads the whole state vector at its point; under Lobatto the `k = 1` coefficient sits on the `k = 0`
node, so an interpolation over both divides by zero and must drop one, which is what
`_interp_basis` in `api/interpolate.jl` does and what callers should reach for instead of
rebuilding it; and `r.fiter` is
`fwhich × N × K` long, so a row lookup that scans it is quadratic in the mesh — key it once into a
`Dict`.

`r.fiter` are the rows `f` takes, which means the caller's data crossed with the mesh where it
was not already, plus `t` where the mesh is numeric, and `r.fwhere` is the `RowLayout` that
built them. Read the layout off the record rather than
recomputing it from a length: the slot positions came from the probe and are not recoverable by
counting.

`_split_collocation_args` accepts both `f(a; k = v)` and `f(a, k = v)` — reuse it for new macros.
It lives in `src/api/macros.jl` with `_name_val` and the rewrite.

## The adaptive mesh

`h` and `t` are the only t-space data the residuals read — `A`, `b`, `taus` are τ-space and do not
move — so they are the only two an adaptive mesh has to make mutable. Under `adaptive = true` each
is *additionally* allocated as an `add_par` block; the numeric arrays stay, because bounds,
`start`, and initialization by integration need plain numbers.

**Both paths ship, and `adaptive` picks.** The numeric path is not dead weight: it folds
`-h[i]·A[j,k]` into one constant, where the parameter path allocates `N·K + N` parameters and
traces a product of two symbolic terms.

**`mesh.h` and `mesh.t` are one name each**, resolving through `getproperty` to the parameter
block when adaptive and the array otherwise, since indexing them inside a residual is the only
thing either is written for. Internals take the numbers with `_hval`/`_tval`. **Anything outside
the package that wants numbers on an adaptive mesh has to recompute them** — `diff(nodes)` for
`h`, `nodes[i] + h[i]*taus[k]` for `t` — which is what `examples/bruno.jl` does for its RK4
start guess. That is the cost of the single name, and it is paid only where a start value or a
bound is being built, never in an expression.

**A graph node cannot ride in an iterator tuple.** `add_con` and `ExaModel` accept a
`Vector{Tuple{Int,ParameterNode}}` silently and then `cons!` throws `Cannot convert Node2 to
Float64` — a build-time silence, so it is worth stating twice. That is why `t` reaches the
generator differently on each mesh while the caller's row is the same: numerically it is
appended to the data, adaptively `_appendt` builds `tpar[i,k]` at trace time. `_appendt` rebuilds
the row with `ntuple(j -> r[j], Val(L))` rather than splatting it, because a traced row defines
`indexed_iterate` but not `iterate`.

`set_nodes!` redistributes a **fixed** number of intervals; adding one changes the variable count
and needs a rebuild.

## Testing

Tests must **exercise every build path**: each roots family × each basis × a range of `K`.

**`test/` mirrors `src/`, one file per source file**, and `runtests.jl` includes them in that
order. A new helper gets `test/api/<its name>.jl` because it got `src/api/<its name>.jl`, and
nothing else earns a file.

**The whole models live in `examples/` only, and CI does not run them.** No copy of van der Pol
or Bruno sits in `test/`, so **the Bruno objective pin is not enforced by `Pkg.test()`** and
nothing in CI solves a problem of that size. Run all three by hand after changing a helper,
`refinement.jl` first: it is the only one that reads `f` back off the recorded `Residual` and
moves the mesh with `set_nodes!`, so it
exercises the probe, the parameter path of `t`, and the mesh naming together. It now calls
`interpolate` for every evaluation off the collocation points, so its own `_blocksol` is the
only thing left that ExaModels does not hand it: `solution()` wants a `result`, and the error
estimate holds a raw primal vector instead.

**`test/api/interpolate.jl` reuses `solve_decay` and `MODES`** out of
`test/api/add_con_collocation.jl` rather than rebuilding the decay problem, which is why
`runtests.jl` includes it after. It pins the polynomial against `exp(-t)` off the mesh, against
the stored coefficient at a node, and the returned shape against the block's declared
dimensions.

**Every example is a top-level script that solves**, so `julia --project=examples
examples/<name>.jl` runs one end to end and no file carries a module wrapper. Each
`examodel_<name>` returns an `ExaModel` and takes `adaptive` as a keyword defaulting to `false`,
so the `----- Solve -----` section at the bottom reads its handles back off the model
(`model.p`) rather than off a returned tuple. Only `refinement.jl` plots, and it is the only one
that needs `adaptive = true`.

`test/collocation/` — the mode math. Check weights against the identities that define them (a
Lagrange weight vector applied to the nodal values of a polynomial of degree `≤ length(nodes)-1`
reproduces exactly the operation it encodes, so monomials pin every entry) rather than against
stored numbers. Assert the shapes too; that is what catches a weight matrix wrong for its basis.

`test/api/` — every exported helper, closing with both residual forms on `dz/dt = -z` solved
through them, using invariants a fixed tolerance would miss:

- **The bases agree.** Algebraically equivalent discretizations over the same variables must reach
  the same solution to solver tolerance.
- **The order is right.** Terminal error decays at `O(h²ᴷ)` for Legendre, `O(h²ᴷ⁻¹)` for Radau,
  `O(h²ᴷ⁻²)` for Lobatto. A mis-weighted stencil can still converge; it loses order.
- **The spellings pin to each other.** `z[v]` and `z[v,i,k]`, an omitted `i,k` and a written-out
  one, slot indices that lead the row and slot indices that do not, and the macro against the
  function must all give identical residuals row for row — compare `cons`, since a solve
  tolerance hides a wrong row. `adaptive = true` must *reproduce* the numeric mesh, not merely
  come close; after `set_nodes!` the model must match one rebuilt from scratch — which is what
  catches `h` moving without `t`.

**Both whole models declare the state as one block**: `z[v,i,k]` in `examples/vanderpol.jl` and
`z[v,c,i,k]` in `examples/bruno.jl`, with three structurally distinct expressions there and four
here, so that many `@add_con_collocation` calls over disjoint slots feed a single
`@add_con_continuity` in each. The **zero-leading-dimension** branch of every stencil is covered
by `examples/refinement.jl` and `test/api/add_con_collocation.jl` instead — van der Pol's `u` is
declared with no dimensions but is a control, so no residual is written over it. Bruno's
`NLL_REF`, `-46.68818145`, is the negative log-likelihood PEtab.jl reports at the collection's
nominal parameters, and matching it to `atol = 1e-6` covers the whole discretization in one
number. The script prints that gap, `1.9e-8` from the nominal start on the `K = 4`, `N = 36`
uniform Radau mesh it ships with. That gap is the printed precision of `NLL_REF` itself, not
discretization error: `N = 36`, `72` and `180` agree on the objective to `1e-13`. The mesh has to
be uniform on a **multiple of 36** intervals, since every measurement time is a multiple of 5 over
`[0, 180]` and the objective reads `z[v,c,i,K]`, the right end of interval `i`. **Neither runs
under `Pkg.test()`**, so the pin holds only when the example is run by hand.

`ExaModels.solution(result, z)` returns a plain 1-based array, so a block indexed `k = 0,…,K`
lands on `1,…,K+1` there.

```julia
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. -e 'using ExaModelsCollocation'
julia --project=examples examples/refinement.jl
```

Julia compat is `1.11+`, a floor of this package's own choosing and above what the dependencies
ask for: ExaModels 0.12 sets `1.9` and FastGaussQuadrature `1.10`. Nothing here uses a 1.12-only
construct, so raising it would only drop users (CI tests 1.11, 1.12, and `pre`);
ExaModels is pinned to `0.12`. Test-only dependencies go in `test/Project.toml`.

**0.12 removed `set_parameter!`**, folding it into `set_value!(::ExaCore, …)`, so both the core
and the model path of `set_nodes!` call `set_value!` now. The suite did not catch that on its
own: nothing exercised `set_nodes!` on a *core*, which was the one uncovered line in `core.jl`.
`test/api/set_nodes.jl` covers it now. Registry 0.12 also has **no `Colon` indexing**, so the
two forwarding lines in `handles.jl` are dormant until `jc/colon-indexing` lands on it.
