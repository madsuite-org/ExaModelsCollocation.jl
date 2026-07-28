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
3. `DAEta`, which carries that discretization so the helpers never ask for it twice.

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

`daeta.jl` comes last and holds the result.

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
weights). The helpers branch on `_isstateform(dae)` at runtime and build both.

One consequence: `StateForm` continuity needs no right-hand side, so `@add_con_continuity`
takes only the slice and one call covers everything; `DerivativeForm` continuity contains `f`,
so it takes the collocation call's slice *and* generator. Passing the wrong arity for the
active mode throws — do not make it silently build the other form.

Under Lobatto, `τ₁ = 0` coincides with the `k = 0` index, so the `k = 1` row reduces to
`z[…,i,1] = z[…,i,0]`: one redundant variable and one trivial equation per interval. It is
consistent, and the observed order is unaffected.

## The helpers

**One file per exported function, named after it.** `src/add_con_collocation.jl` defines
`add_con_collocation`, and so on. Keep new helpers to that rule.

- `add_var_collocation` — `ExaModels.add_var` with the two mesh axes appended. The caller
  declares the **per-timepoint** shape; `z[v,c]` declared becomes `z[v,c,i,k]` allocated,
  `i = 1,…,N` and `k` over `krange`. `include_boundary = true` (default) gives `k = 0,…,K`,
  carrying the interval-left boundary node continuity needs; `false` gives `k = 1,…,K`.
  Pass-through keywords (`start`, `lvar`, `uvar`, `tag`) are shaped to the **allocated**
  block, not the declared one.
- `@add_con_collocation` — `ExaModels.@add_con` plus exactly three things: the expression is
  put into the residual form of `dae.mode`, the basis-polynomial sum is attached as an
  `add_con!` augmentation, and the iterator runs over the collocation points. Nothing else.
- `@add_con_continuity` — the junction rows for `i = 1,…,N-1`, in the form the mode dictates.
  Under `StateForm` the caller's iterator names only the **slots** to tie; `i` is fixed by
  the mode, so the macro crosses it in under a gensym rather than making the caller
  destructure an index the residual never lets them use. Same principle as `h`: data the
  caller cannot reference does not belong in the tuple. `DerivativeForm` is the exception —
  it reuses the collocation generator, where `i` is already the caller's own.

In `DerivativeForm`, both helpers re-evaluate the caller's right-hand side at every
collocation point `j` of the interval. The stencil does that by rebinding the caller's own
loop variables to that point, so the *same* expression means `f_ij` in the augmentation and
`f_ik` in the base row. That is why the caller writes it once, unchanged, for either mode.

Initial, terminal, and path conditions carry **no collocation content**. Write them with
plain `ExaModels.@add_con`. Do not grow this module past the three helpers above.

## ExaModels idioms this module must follow

`ExaModelsPEtab.jl/src/collocation.jl` and `continuity.jl` are the reference implementation of
this exact pattern; read them before changing a helper.

1. **Mirror the upstream signature.** `add_var` takes `(core, dims...)` with the name as an
   optional `name = Val(:z)` **keyword** — never a positional `Symbol` — and returns
   `(core, var)`. The macro is what writes the name bare and binds it locally. Helpers here
   do the same. `ExaCore` is immutable and gets rebound; `DAEta` is mutable and is updated in
   place, so it is passed but never returned.
2. **Iterators are flat arrays of tuples, destructured in the generator**: `for (v,c,i,k,t)
   in itr`. Not NamedTuples with `d.field`.
3. **Traced loop indices cannot index a plain Julia array.** `A[j+1,k]` or `t[i,k]` inside a
   user expression throws `invalid index … of type DataIndexed`. Numeric data must ride in
   the iterator tuple — hence `collocation_itr` carrying `t`. Data the *caller* never
   references does not belong there: the macro appends `h` itself, before tracing.
4. **`add_con!` keys rows by position in the base iterator, not by index value.** PEtab gets
   away with `(i,k,cidx) =>` because all its ranges start at 1, so value == position. Here
   `leads` may be a restricted slice (`2:2`) where they diverge, so base iterators are
   `vec`'d flat and the augmentation is keyed by linear position. Getting this wrong yields
   an INFEASIBLE model, not an error.
5. **One `add_con` call per structurally distinct algebraic expression, and no more.** This
   is the whole point of ExaModels: everything that merely *varies* goes in the iterator,
   including the value a constraint equals. Van der Pol's three state equations genuinely
   differ, so they need three calls; its three initial conditions collapse to two (numeric
   vs. `p`-linked). PEtab groups the same way in `_create_initial_conditions`.
6. **The constraint helpers are macros, taking a state slice `z[leads...]`.** The macro
   parses the slice to learn which variable is collocated: literal indices pin a dimension,
   names bound by the iterator vary with it. That is how the stencil knows to emit
   `z[1,c,i,j]` versus `z[v,c,i,j]`. Do not reintroduce a `Symbol`-keyed lookup.

The leading dimensions are the caller's to name — the helpers impose no meaning on them, and
neither should the README.

## The DAEta contract

`DAEta(nodes, K)` fixes the mesh and mode **once**; every helper reads them off the container.

Five fields, nothing stored twice:

- `mode` (`CollocationMode`) — `roots`, `basis`, `polynomial`, `weights`. τ-space.
- `mesh` (`CollocationMesh`) — `nodes`, `h`, `t`. t-space.
- `vars`, `cons` — `NamedTuple`s of the handles that were given a name.
- `blocks` — `Vector` of every `VarBlock`, named or not.

This mirrors `ExaCore`, which keeps every variable in `core.var` and only the named ones in
`core.refs`. `_block_layout` and `block` find a block by **handle identity**, never by
`Symbol`, so anonymous blocks work; `_block_layout` also rejects more than two leading dims.

`N`, `K`, `nodes`, and the `mode` fields are **derived**, surfaced through `getproperty` — do
not add them back as stored fields. Internally use `_mesh`, `_mode`, `_weights`, `_degree`,
`_nintervals` rather than reaching for fields that do not exist. Because `DAEta` overrides
`getproperty`, internal code must use `getfield`/`setfield!` for real fields.

`_split_collocation_args` in `add_var_collocation.jl` accepts both `f(a; k = v)` and
`f(a, k = v)` keyword spellings — reuse it for new macros.

## Testing

Tests must **exercise every build path**: each roots family × each basis × a range of `K`.

`test/collocation.jl` covers the mode math. Check the weights against the identities that
define them — a Lagrange weight vector applied to the nodal values of a polynomial of degree
`≤ length(nodes)-1` reproduces exactly the operation it encodes, so monomials pin every entry
— rather than against stored numbers. Assert the shapes too; that is what catches a weight
matrix silently wrong for its basis.

`test/add_con_collocation.jl` covers both residual forms on `dz/dt = -z`, solved through the
helpers, using the two invariants that a fixed tolerance would miss:

- **The bases agree.** `StateForm` and `DerivativeForm` are algebraically equivalent
  discretizations over the same variables, so on the same roots they must reach the same
  solution to solver tolerance.
- **The order is right.** Terminal error must decay at `O(h²ᴷ)` for Legendre, `O(h²ᴷ⁻¹)` for
  Radau, `O(h²ᴷ⁻²)` for Lobatto. A mis-weighted stencil can still converge; it loses order.

`test/vanderpol.jl` covers the helpers end to end on the optimal control problem.

`test/bruno.jl` does the same for parameter estimation, on the PEtab Benchmark Collection's
`Bruno_JExpBot2016`. It is the regression test for two things nothing else covers: `start` on
`add_var_collocation` (the state profile is initialized by integrating at the starting
parameters), and grouping — seven species equations reduce to four `@add_con_collocation`
calls. Its objective is checked against the negative log-likelihood PEtab.jl reports at the
collection's nominal parameters, `-46.68818145`, to `atol = 1e-6`; that single assertion
covers the whole discretization, so do not loosen it to make an unrelated change pass.

`ExaModels.solution(result, z)` returns a plain 1-based array, so a block indexed `k = 0,…,K`
lands on `1,…,K+1` there.

```julia
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. -e 'using ExaModelsCollocation'
```

Julia compat is `1.10+` (CI tests 1.11, 1.12, and `pre`); ExaModels is pinned to `0.11`.
Test-only dependencies go in `test/Project.toml`, not the root `Project.toml`.
