# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Goal

ExaModelsCollocation.jl applies orthogonal collocation on finite elements to models built
with [ExaModels.jl](https://github.com/exanauts/ExaModels.jl). It is an **internal helper
module**, not a package that owns your problem: no single entry point, no dimension
inference, no initialization, no integrator, no solver wrapper. You assemble the model with
`ExaModels.add_*` and reach for a helper only where collocation actually changes something.

Concretely, it supplies three things and nothing else:

1. the **weights and collocation points** for a chosen mode,
2. the **collocation and continuity equations** built from them, and
3. `CollocationExaCore`, an `ExaCore` carrying that discretization so the helpers never ask
   for it twice.

Dependencies are restricted to **ExaModels** and **FastGaussQuadrature**. Do not add others.

There is no `docs/` site — the README is the API reference, and it is kept **short**, shaped
the way ExaModels documents `add_var`: signature block, one or two sentences of what it does,
`### Keyword Arguments`, and an example only where it disambiguates. A function and its macro
share a section. Rationale, derivations, Biegler cross-references, and worked models do not
belong there — they live in this file, in `src` comments, or in `test/`. Every code block in
the README must run as written; execute them before committing.

Convention used in comments and tests, not enforced anywhere: `z` differential state,
`y` algebraic state, `u` control, `p` free decision parameter, `theta` mutable parameter,
`t` time.

## The discretization, in the order the code builds it

The horizon is split into `N` intervals at `N+1` boundaries. Inside each interval, the state is
a degree-`K` polynomial, and the differential equations are enforced at `K` collocation
points. `src/collocation/` builds that in four steps, and `src/ExaModelsCollocation.jl`
includes them in exactly this dependency order:

1. **`taus.jl`** — where in the reference interval `[0,1]` the equations are enforced. The
   collocation points `τ₁..τ_K` are roots of Gauss-Jacobi polynomials (Biegler, *Nonlinear
   Programming*, Thm. 10.1), mapped from `[-1,1]`: `GaussRadau`, `GaussLegendre`,
   `GaussLobatto`. Only the true collocation points are returned; the `τ₀ = 0` anchor is a
   basis concern, added in step 3.
2. **`polynomial.jl`** — the interpolating polynomial family and the basis operators built
   from it. Every operator is "basis function `j`, evaluated somehow at point `k`", laid out
   `[j,k]` so a constraint reads one column per collocation point: `delljk` (derivative),
   `ell1j` (value at `τ=1`), `Omegajk` (integral to each point), `Omega1j` (integral to 1).
   Each takes the nodes it builds the basis over — none of them re-derives or re-splits its
   argument. Lagrange is preferred because its coefficients are the profile values, so they
   inherit the same variable bounds (Biegler Ch. 10.2.1).
3. **`basis.jl`** — which nodes to hand those operators, i.e. what the polynomial represents.
   `StateForm` represents `z`, so it prepends the `τ₀ = 0` anchor and differentiates:
   `A` is `(K+1) × K`, `b` is length `K+1`, indexed `A[j+1,k]` and `b[j+1]` for `j = 0,…,K`.
   `DerivativeForm` represents `dz/dt`, so it uses the `K` collocation points alone and
   integrates: `A` is `K × K`, `b` is length `K`.
4. **`mesh.jl`** — placement along physical time: `h[i] = nodes[i+1] - nodes[i]` and
   `t[i,j] = nodes[i] + h[i]·τ_j`.

`src/core/` comes last and holds the result: `handles.jl` the variable handle, `core.jl` the
container, in that include order.

**`N` counts intervals, not boundary points.** `h = diff(nodes)`, `N = length(h)`, so 21
nodes give `N = 20` and `t` is `20 × K`.

`GaussLobatto` places a collocation point on `τ = 0`, which collides with the anchor
`StateForm` prepends — repeated nodes give `NaN` barycentric weights — so `_set_mesh!`
switches it to `DerivativeForm`, which needs no anchor, with a warning.

Biegler's indexing is used throughout, including where it reuses letters: `j` indexes the
basis polynomial being summed (and the mesh, `t[i,j]`), `k` the collocation point being
enforced.

**The basis changes the structure of the equations, not just the weights.** Both forms use
the *same variables*: `ż_{i,j}` is never a decision variable, it is the right-hand side
evaluated at `z[…,i,j]`, and Biegler's separate interval-entry coefficient `z_{i-1}` is just
the `k = 0` index — no extra block. With `f_ij = f(z[…,i,j], t[i,j])`:

| | collocation | continuity |
|---|---|---|
| `StateForm` | (10.7) `Σⱼ₌₀..ᴷ A[j,k] z[…,i,j] = h[i] f_ik` | (10.14a) `Σⱼ₌₀..ᴷ b[j] z[…,i,j] = z[…,i+1,0]` |
| `DerivativeForm` | (10.8) `z[…,i,k] − z[…,i,0] = h[i] Σⱼ₌₁..ᴷ A[j,k] f_ij` | (10.15a) `z[…,i+1,0] − z[…,i,0] = h[i] Σⱼ₌₁..ᴷ b[j] f_ij` |

`StateForm` puts the state under the weights and evaluates `f` once per row; `DerivativeForm`
puts `f` under the weights, re-evaluating it at every collocation point of the interval, which
is what makes it the implicit Runge-Kutta step (`A` the Butcher tableau, `b` its quadrature
weights). The helpers branch on `_isstateform(core)` at runtime and build both.

One consequence: `DerivativeForm` continuity contains `f`, where `StateForm` continuity needs
none. **The caller writes it once either way.** `add_con_collocation` records its rows and its
right-hand side as a `Residual` on the container under `DerivativeForm`, and continuity reads
the `f` for a slot back off that record, so both bases take the same call — the slots to tie,
and nothing else. Keep that property: a second spelling of continuity is what this replaced.
The record is what makes the ordering real, so `add_con_collocation` comes first and a slot
with no residual behind it throws.

Under Lobatto, `τ₁ = 0` coincides with the `k = 0` index, so the `k = 1` row reduces to
`z[…,i,1] = z[…,i,0]`: one redundant variable and one trivial equation per interval. It is
consistent, and the observed order is unaffected.

## The helpers

**One file per exported function, named after it.** `src/add_con_collocation.jl` defines
`add_con_collocation`, and so on. Keep new helpers to that rule.

**Each is a function plus a macro of the same name**, related the way `ExaModels.add_var` and
`@add_var` are: the function does the work and returns `(core, handle)`, and the macro writes
the name bare, rebinds `core` in the calling scope, and calls the function. Everything a
residual depends on belongs in the function.

- `add_var_collocation` — `ExaModels.add_var` with the two mesh axes appended. The caller
  declares the **per-timepoint** shape; `z[v,c]` declared becomes `z[v,c,i,k]` allocated,
  `i = 1,…,N` and `k` over `krange`. Declaring none gives a scalar state, `z[i,k]`, the way
  `add_var(core)` gives a scalar variable. `include_boundary = true` (default) gives
  `k = 0,…,K`, carrying the interval-left boundary node continuity needs; `false` gives
  `k = 1,…,K`. Pass-through keywords (`start`, `lvar`, `uvar`, `tag`) are shaped to the
  **allocated** block, not the declared one.
- `add_con_collocation` — `ExaModels.add_con` plus exactly three things: the expression is
  put into the residual form of `core.mode`, the basis-polynomial sum is attached as an
  `add_con!` augmentation, and the iterator runs over the collocation points. Nothing else.
- `add_con_continuity` — the junction rows for `i = 1,…,N-1`, in the form the mode dictates.
  It takes the variable alone: the slots come off the `Residual`s the collocation calls
  recorded, and `i` is fixed by the mode, so the helper crosses both in itself rather than
  making the caller carry indices the residual never lets them use. Same principle as `h`:
  data the caller cannot reference stays out of the tuple. Every slot of the block must be
  covered exactly once — twice and a junction row would integrate two right-hand sides, not
  at all and it would be left silently untied — so `add_con_collocation` comes first, for
  either basis.

In `DerivativeForm`, both helpers re-evaluate the caller's right-hand side at every
collocation point `j` of the interval, by handing `f` a row of the caller's own iterator with
the trailing `(i, k, t)` moved to that point. The *same* expression therefore means `f_ij` in
the augmentation and `f_ik` in the base row, which is why the caller writes it once,
unchanged, for either mode. Continuity does the same thing from the other side: each row of
the iterator is an `f_ik` already, so it carries its own `b[k]` into the junction row of its
interval.

Initial, terminal, and path conditions carry **no collocation content**. Write them with
plain `ExaModels.@add_con`. Do not grow this module past the three helpers above.

## ExaModels idioms this module must follow

`ExaModelsPEtab.jl/src/collocation.jl` and `continuity.jl` are the reference implementation of
this exact pattern; read them before changing a helper.

1. **Mirror the upstream signature.** `add_var` takes `(core, dims...)` with the name as an
   optional `name = Val(:z)` **keyword** — never a positional `Symbol` — and returns
   `(core, var)`. The macro is what writes the name bare and binds it locally. Helpers here
   do the same, and there is nothing to thread alongside `core` — it carries the
   discretization itself. See the container contract below.
2. **Iterators are flat arrays of tuples, destructured in the generator**: `for (v,c,i,k,t)
   in itr`. Not NamedTuples with `d.field`.
3. **Traced loop indices cannot index a plain Julia array.** `A[j+1,k]` or `t[i,k]` inside a
   user expression throws `invalid index … of type DataIndexed`. Numeric data must ride in
   the iterator tuple — hence the caller's row carrying `t`. Data the *caller* never
   references does not belong there: the helper appends `h` itself, before tracing.
4. **`add_con!` keys rows by position in the base iterator, not by index value.** PEtab gets
   away with `(i,k,cidx) =>` because all its ranges start at 1, so value == position. Here a
   row's leading indices may name a restricted slice (`2:2`) where they diverge, so base
   iterators are `vec`'d flat and the augmentation is keyed by linear position. Getting this
   wrong yields an INFEASIBLE model, not an error. **The helper does that flattening** —
   `_flat` on the way in — so a caller hands over whatever shape the comprehension made and
   never writes `vec` itself. The tests are written that way; keep them that way.
5. **One `add_con` call per structurally distinct algebraic expression, and no more.** This
   is the whole point of ExaModels: everything that merely *varies* goes in the iterator,
   including the value a constraint equals. Bruno's seven species equations are four
   structurally distinct expressions, so they need four calls; its forty-two initial
   conditions collapse to two (numeric vs. `p`-linked). PEtab groups the same way in
   `_create_initial_conditions`. **A block is what sets the floor**: one call cannot span
   two blocks, so splitting the state costs a call per block on everything that was uniform
   across it — which is why Van der Pol, one block per state, writes three initial
   conditions where one shape would have done.
6. **The stencil lives in the function; the macro adds nothing to it.** `add_con_*` takes the
   variable and a generator whose rows read
   `(z's own indices…, whatever else f varies with…, i, k, t)`, and indexes `z` straight off
   the leading entries — a literal there holds that dimension at one value, a name bound by
   the iterator varies with it. `RowLayout` locates the slot and the mesh entries by counting
   from both ends, so the middle is the caller's to order. `_block_layout` admits **zero to
   two** leading dimensions and every stencil has a branch per count; a block with none has
   no slot at all, so its rows start at `i` and its one slot is the empty tuple, which is
   what `Iterators.product()` already yields in `_covered_slots`. `@add_con_*` takes the same
   arguments and only rebinds `core` and writes the name bare, so the two forms cannot drift.
   Keep the lookup by handle identity rather than a `Symbol`. The caller builds the iterator:
   a product helper cannot express a row like `(v, l[v], i, k, t)` whose later entries depend
   on the leading ones, which is the common case (`test/bruno.jl`'s `sweep`).
7. **The macros take an optional bare `name`, exactly as `ExaModels.@add_var` does.** Named
   binds it locally and registers it on `core`; anonymous registers nothing and
   returns the handle. `_name_val` turns the one into `Val(:name)` and the other into
   `nothing`, which is the keyword the function already takes.

The leading dimensions are the caller's to name — the helpers impose no meaning on them, and
neither should the README.

## The container contract

`CollocationExaCore(nodes, K)` fixes the mesh and mode **once**; every helper reads them off
the core it is already given.

**It is an `ExaCore`, not a wrapper around one.** `ExaCore` is a concrete struct and cannot be
subtyped; the extension point is its `tag` parameter, and this follows ExaModels' own
`two_stage.jl` exactly:

```julia
struct CollocationTag{MO,ME} <: ExaModels.AbstractExaModelTag ... end
const CollocationExaCore{T,VT,B} = ExaCore{T,VT,B,<:CollocationTag}
const CollocationExaModel{T,VT,E,V,P,O,C,R} = ExaModel{T,VT,E,V,P,O,C,<:CollocationTag,R}
```

Two things follow, and both are the point. Every `ExaModels.add_*` dispatches on `ExaCore{T}`,
so it works here **unforwarded** — a model mixes plain ExaModels constraints with collocation
ones on one object. And `ExaModel(c)` passes `c.tag` through, so the mesh survives into the
model, which is what `set_nodes!` needs after a solve. `LegacyExaCore` is the only
`AbstractExaCore` subtype and exists to be *mutable* for the deprecated API; do not copy it.

**The alias drops `ExaCore`'s own `VT <: AbstractVector{T}` bound.** A method written
`c::CollocationExaCore` is therefore *ambiguous* with ExaModels' `getproperty`/`show` over
`E <: Union{ExaCore, ExaModel}` — it fails at the first `add_var`, not at load. Restate the
bound in the `where` clause, which is what `two_stage.jl` does on every one of its methods:

```julia
Base.getproperty(c::CollocationExaCore{T,VT,B}, name::Symbol) where {T,VT<:AbstractVector{T},B}
```

Four fields on the tag, nothing stored twice:

- `mode` (`CollocationMode`) — `roots`, `basis`, `polynomial`, `weights`. τ-space.
- `mesh` (`CollocationMesh`) — `nodes`, `h`, `t`, and `hpar`, `tpar` under an adaptive mesh.
  t-space.
- `blocks` — every `CollocationVariable`.
- `residuals` — every `Residual`: the rows and the right-hand side one `add_con_collocation`
  call was given. Recorded for **both** bases: `StateForm` continuity has no use for the `f`,
  but both read their slots off it.

**Named handles are not tracked here.** `add_var(c, …; name = Val(:z))` already registers into
the inner `refs`, which `ExaModel` carries over, so `c.z` and `model.z` both fall out of
forwarding `getproperty` — there is no second registry to keep in step. `add_var_collocation`
only swaps the `CollocationVariable` in for the bare `Variable` afterwards, through
`_rehandle`, so the handle the caller holds is the handle `core.z` gives back.

**`blocks` and `residuals` are `Vector`s grown in place**, so the tag's type is fixed at
construction. This is what `two_stage.jl` does with `var_scen`, and what `ExaCore` itself does
with `x0` and `lvar` inside `add_var`. The consequence is real and worth knowing: a core and
every core derived from it **share one tag**, so a block added to the later one is visible on
the earlier one. Only the `ExaCore` fields are copy-on-update.

`N`, `K`, `nodes`, `adaptive`, and the `mode` fields are **derived**, surfaced through
`getproperty` on both the core and the model — do not add them back as stored fields.
Internally use `_mesh`, `_mode`, `_weights`, `_degree`, `_nintervals`, and `getfield` for real
fields.

**`CollocationVariable <: ExaModels.AbstractVariable` carries its own layout** — `dims` and
`krange` — so there is no side table to look a block up in, and "was this created by
`add_var_collocation`?" is a type check. This works because ExaModels writes every indexing
method generically over `AbstractVariable` (`.size`, `.offset`, `.length`) and because
`core.var` is inert bookkeeping the evaluator never reads. **Constraints are the opposite**:
`offset0(::Constraint, i)` and `_constraint_dims(::Constraint)` are typed concretely, and
`core.cons` *is* read in the evaluator hot loop, so `add_con_collocation` returns the plain
`Constraint` and the record stays a separate `Residual`. Do not wrap it.

`_split_collocation_args` in `add_var_collocation.jl` accepts both `f(a; k = v)` and
`f(a, k = v)` keyword spellings — reuse it for new macros.

## The adaptive mesh

`h` and `t` are the only t-space data the residuals read — `A`, `b` and `taus` are τ-space and
do not move — so they are the only two an adaptive mesh has to make mutable. Under
`adaptive = true` each is additionally allocated as an `add_par` block and the residuals index
the handle; the numeric arrays stay, because bounds, `start`, and initialization by
integration need plain numbers.

**Both paths ship, and `adaptive` picks.** The numeric path is not dead weight: it appends `h`
to the row in plain Julia and folds `-h[i]·A[j,k]` into one constant, where the parameter path
allocates `N·K + N` parameters and traces a product of two symbolic terms. Keep it.

**A graph node cannot ride in an iterator tuple.** `add_con` and `ExaModel` accept a
`Vector{Tuple{Int,ParameterNode}}` silently and then `cons!` throws `Cannot convert Node2 to
Float64` — a build-time silence, so this is worth stating twice. That is why an adaptive row
ends `(…, i, k)` and a numeric one ends `(…, i, k, t)`: with a parameter mesh the caller
indexes `mesh.tpar[i,k]` inside the right-hand side instead of destructuring `t`.
`_row_layout` builds the `RowLayout` from `_nmesh(core)` accordingly. It also means the same
expression means `f_ij` in a `DerivativeForm` augmentation and `f_ik` in the base row for free,
since `tpar` is indexed by whatever `k` the row carries.

`set_nodes!` recomputes `nodes`, `h`, `t` in place and writes both parameter blocks through.
It redistributes a **fixed** number of intervals; adding one changes the variable count and
needs a rebuild.

## Testing

Tests must **exercise every build path**: each roots family × each basis × a range of `K`.

**Four files, and keep them to four.** `collocation.jl` for the mode math, `api.jl` for every
helper the package exports, then `vanderpol.jl` and `bruno.jl` for the two whole models. A new
helper gets a testset in `api.jl`, not a file of its own.

`test/collocation.jl` covers the mode math. Check the weights against the identities that
define them — a Lagrange weight vector applied to the nodal values of a polynomial of degree
`≤ length(nodes)-1` reproduces exactly the operation it encodes, so monomials pin every entry
— rather than against stored numbers. Assert the shapes too; that is what catches a weight
matrix silently wrong for its basis.

`test/api.jl` covers `CollocationExaCore`, `add_var_collocation`, and the constraint helpers,
and closes
with both residual forms on `dz/dt = -z`, solved through them, using the two invariants that a
fixed tolerance would miss:

- **The bases agree.** `StateForm` and `DerivativeForm` are algebraically equivalent
  discretizations over the same variables, so on the same roots they must reach the same
  solution to solver tolerance.
- **The order is right.** Terminal error must decay at `O(h²ᴷ)` for Legendre, `O(h²ᴷ⁻¹)` for
  Radau, `O(h²ᴷ⁻²)` for Lobatto. A mis-weighted stencil can still converge; it loses order.

It also pins the two interfaces to each other: the same decay problem, built once through the
macros and once through the functions, must give identical residuals row for row. And it
pins the two **meshes** to each other the same
way: on the same nodes, `adaptive = true` must reproduce the numeric mesh's solution, not
merely come close; after a `set_nodes!`, the model must match one rebuilt on those nodes from
scratch. That second one is what catches `h` moving without `t`.

**The two whole models are deliberately opposite in how they declare the state, and that is
the point** — the helpers impose no shape, so both spellings have to keep working.

`test/vanderpol.jl` covers the helpers end to end on the optimal control problem, with **one
block per state**: `z1`, `z2`, `z3` and `u` are declared with no dimensions at all, so they
are `z1[i,k]`, and each takes its own `@add_con_collocation` and `@add_con_continuity`. It is
what covers the zero-leading-dimension branch of every stencil outside `api.jl`, and what
shows the cost of splitting: three states that share an initial-condition shape still need
three `@add_con` calls, because a call cannot span two blocks.

`test/bruno.jl` does the same for parameter estimation, on the PEtab Benchmark Collection's
`Bruno_JExpBot2016`, with **one block for everything**: all seven species are `z[v,c,i,k]`.
It is the regression test for two things nothing else covers: `start` on
`add_var_collocation` (the state profile is initialized by integrating at the starting
parameters), and grouping — seven species obey four structurally distinct expressions, so the
block takes four `@add_con_collocation` calls over disjoint slots and a single
`@add_con_continuity` that reads its slots off all four records. That is the multi-residual
path through `_covered_slots`, which nothing else exercises. Because it is one block, the
initial conditions collapse to two calls and the objective to one. Its objective is checked
against the negative log-likelihood PEtab.jl reports at the collection's nominal parameters,
`-46.68818145`, to `atol = 1e-6`; that single assertion covers the whole discretization, so do
not loosen it to make an unrelated change pass.

`ExaModels.solution(result, z)` returns a plain 1-based array, so a block indexed `k = 0,…,K`
lands on `1,…,K+1` there.

```julia
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. -e 'using ExaModelsCollocation'
```

Julia compat is `1.10+` (CI tests 1.11, 1.12, and `pre`); ExaModels is pinned to `0.11`.
Test-only dependencies go in `test/Project.toml`, not the root `Project.toml`.
