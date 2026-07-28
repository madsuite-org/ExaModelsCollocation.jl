# ExaModelsCollocation.jl

Helper functions for collocation in [ExaModels.jl](https://github.com/exanauts/ExaModels.jl).

Each helper mirrors its `ExaModels.add_*` counterpart and adds only what collocation
requires: the mesh axes on a variable block, the residual form of the chosen mode on a
constraint. What your declared dimensions *mean* is up to you — nothing here imposes an
index convention.

---

## `DAEta`

```julia
dae = DAEta(nodes, K; roots = GaussRadau(), basis = StateForm(), polynomial = Lagrange())
dae = DAEta()
```

The collocation metadata threaded through the helpers. Holds the mesh and the discretization
mode, and accumulates the blocks you create. `DAEta` is mutable and helpers update it in
place, so it is passed but never rebound.

The mesh is fixed once, here — helpers read `N` and `K` off the container rather than taking
them again. The no-argument form builds an empty container; attach a mesh later with
`set_mesh!`.

### Arguments

- `nodes` : the `N+1` element boundaries along `t`, nondecreasing. `N = length(nodes) - 1`,
  so `N` counts **intervals**, not boundary points.
- `K` : degree of the interpolating polynomial, `K ≥ 1`.

### Keyword Arguments

- `roots` : collocation points, `GaussRadau()` (default), `GaussLegendre()`, or `GaussLobatto()`.
- `basis` : differential-state representation, `StateForm()` (default) or `DerivativeForm()`.
- `polynomial` : interpolating polynomial, `Lagrange()` (default).

`basis` selects which polynomial is interpolated, and so the **structure** of the equations
the `add_con_*` helpers build — not just the weights. Both use the same variables, because
`ż_{i,j}` is never a decision variable, it is the right-hand side evaluated at `z[…,i,j]`,
and the element-entry coefficient `z_{i-1}` is just the `k = 0` index.

`GaussLobatto` puts a collocation point on `τ = 0`, which collides with the anchor
`StateForm` interpolates; it switches to `DerivativeForm` with a warning, where no anchor is
needed. Terminal accuracy is `O(h²ᴷ)` for Legendre, `O(h²ᴷ⁻¹)` for Radau, `O(h²ᴷ⁻²)` for
Lobatto.

### Fields

Two pieces of data, held once each:

- `mode` : `CollocationMode` — `roots`, `basis`, `polynomial`, and `weights`
  (`A` collocation weights, `b` continuity weights, `taus`). Reference-element data in `τ`-space.
- `mesh` : `CollocationMesh` — `nodes`, `h` element lengths, `t` collocation times.
  Physical placement in `t`-space.

Plus the registries, mirroring how `ExaCore` keeps every variable in `core.var` and only the
named ones in `core.refs`:

- `vars`, `cons` : `NamedTuple`s of the handles that were given a name
- `blocks` : `Vector` of every `VarBlock`, named or not — look one up with `block(dae, z)`

Named handles are forwarded to the container, so `dae.z` and `dae.vars.z` are the same. The
`mode` fields are forwarded too, and `dae.N`, `dae.K`, `dae.nodes` are derived, so all of it
reads flat without being stored twice.

### Example

```julia
julia> using ExaModels, ExaModelsCollocation

julia> dae = DAEta(range(0, 5; length = 21), 3);

julia> dae.N, dae.K            # 21 boundaries -> 20 intervals
(20, 3)

julia> size(dae.mesh.t)
(20, 3)
```

---

## `add_var_collocation`

```julia
add_var_collocation(core, dae, dims...; include_boundary = true, name = nothing, kwargs...)
```

Adds a variable block laid out over the collocation mesh of `dae`. `dims` is the shape at a
single point in time — whatever axes you want — and two mesh axes are appended: the element
index over `1:N`, then the collocation index over `krange`. So

```julia
core, z = add_var_collocation(core, dae, 1:nz, 1:Nc)   # → z[dims..., i, k]
```

allocates `nz × Nc × N × (K+1)` variables. Returns `(core, var)`, exactly as `add_var` does;
`dae` is mutated in place.

### Keyword Arguments

- `include_boundary` : `true` (default) gives `k = 0,…,K`, carrying the element-left boundary
  node that continuity constraints need; `false` gives `k = 1,…,K`, the collocation points alone.
- everything else passes through to `ExaModels.add_var` and means what it does there,
  including `name`, which registers the handle when given as `Val(:name)`. `start`, `lvar`,
  `uvar`, and `tag` are shaped to the **allocated** block, `(dims..., 1:N, krange)`.

### Example

```julia
julia> c = ExaModels.ExaCore(concrete = Val(true));

julia> dae = DAEta(range(0, 5; length = 21), 3);

julia> c, z = add_var_collocation(c, dae, 1:3, 1:2);

julia> c.nvar        # 3 × 2 × 20 × 4
480

julia> c, u = add_var_collocation(c, dae, 1:1, 1:2; include_boundary = false);

julia> block(dae, u).krange
1:3
```

---

## `@add_var_collocation`

```julia
@add_var_collocation(core, dae, name, dims...; kwargs...)
```

Macro interface for `add_var_collocation`, relating to it exactly as `ExaModels.@add_var`
relates to `add_var`: the name is written bare, becomes the `name = Val(…)` keyword, and is
bound in the calling scope along with the updated `core`.

### Example

```julia
c   = ExaModels.ExaCore(concrete = Val(true))
dae = DAEta(nodes, K)

@add_var_collocation(c, dae, z, 1:nz, 1:Nc)                            # z[v,c,i,k], k = 0,…,K
@add_var_collocation(c, dae, y, 1:ny, 1:Nc; include_boundary = false)  # y[v,c,i,k], k = 1,…,K
@add_var_collocation(c, dae, u, 1:nu, 1:Nc; include_boundary = false)  # u[v,c,i,k], k = 1,…,K

c, p = ExaModels.add_var(c, np)   # mesh-free blocks stay plain ExaModels
```

`z`, `y`, `u` are now bound locally, and also reachable as `dae.z`, `dae.y`, `dae.u`.

---

## `block`

```julia
block(dae, var) -> VarBlock
```

The layout recorded for a handle: its declared `dims` and its `krange`. Looked up by handle
identity, so it works whether or not the block was named.

---

## `collocation_itr`

```julia
collocation_itr(dae, leads...)
```

Builds the iterator an `@add_con_collocation` generator runs over. Elements are flat tuples,
destructured in the generator body the way ExaModels iterators normally are:

```
(leads..., i, k, t)
```

`leads` are whichever of the block's dimensions vary across this constraint — dimensions
pinned in the state slice are simply left out. `i` and `k` run over the elements and
collocation points. `t = mesh.t[i,k]` rides along because a traced loop index cannot look it
up in a plain array; leave it unused when the right-hand side is autonomous.

Nothing about this is privileged: any iterator whose tuples end in `(…, i, k, t)` works, so
write your own comprehension when you want a different sweep.

---

## `@add_con_collocation`

```julia
@add_con_collocation(core, dae, name, z[leads...], generator; kwargs...)
```

Behaves like `ExaModels.@add_con` — the generator supplies the right-hand side `f` — except
that it additionally

1. puts the expression into the residual form of `dae.mode`,
2. attaches the basis-polynomial sum as a constraint augmentation, and
3. runs the iterator across the collocation points.

The call is the same for either basis; what gets built is not. Writing
$f_{ij} = f(z_{\ldots,i,j},\, t_{i,j})$, for $k = 1,\dots,K$:

$$\textsf{StateForm} \quad \sum_{j=0}^{K} A_{jk}\, z_{\ldots,i,j} \;=\; h_i\, f_{ik}$$

$$\textsf{DerivativeForm} \quad z_{\ldots,i,k} - z_{\ldots,i,0} \;=\; h_i \sum_{j=1}^{K} A_{jk}\, f_{ij}$$

`StateForm` puts the state under the weights and evaluates `f` once per row.
`DerivativeForm` puts `f` under the weights — it is the implicit Runge-Kutta step, with `A`
the Butcher tableau — so your expression is re-evaluated at every collocation point of the
element. Neither needs a variable the other does not.

The state slice says which variable is being collocated. Its indices may be **literals**,
pinning that dimension, or **names bound by the iterator**, varying with it. Use one call per
structurally distinct right-hand side, and no more:

```julia
# distinct right-hand sides: pin v, vary c
@add_con_collocation(core, dae, coll1, z[1,c],
    z[2,c,i,k] for (c,i,k,t) in collocation_itr(dae, 1:Nc))

# shared right-hand side: v goes in the iterator, one call covers every component
@add_con_collocation(core, dae, coll, z[v,c],
    -decay[v] * z[v,c,i,k] for (v,c,i,k,t) in collocation_itr(dae, 1:Nz, 1:Nc))
```

The element length `h_i` never appears in the iterator: it multiplies the residual and is
never yours to reference, so the macro attaches it.

Updates `core` and `dae` in the calling scope and binds `name` to the new constraint.

---

## `@add_con_continuity`

```julia
@add_con_continuity(core, dae, name, z[leads...] for (leads..., i) in itr; kwargs...)
@add_con_continuity(core, dae, name, z[leads...], generator; kwargs...)
continuity_itr(dae, leads...)
```

Ties each element's terminal polynomial value to the next element's boundary node, over the
`N-1` junctions. Which form applies is set by `dae.mode`, and passing the wrong one is an
error rather than a silently different model.

**`StateForm`** evaluates the state polynomial at `τ = 1`:

$$\sum_{j=0}^{K} b_j\, z_{\ldots,i,j} \;=\; z_{\ldots,i+1,0}$$

That is fixed entirely by the mode, so the generator body is just the state slice — there is
no expression to supply, and one call covers every component and condition. `continuity_itr`
yields `(leads..., i)`; no mesh data rides along because the residual needs none.

```julia
@add_con_continuity(core, dae, cont,
    z[v,c] for (v,c,i) in continuity_itr(dae, 1:Nz, 1:Nc))
```

**`DerivativeForm`** integrates the right-hand side across the element — the Runge-Kutta
step, with `b` its quadrature weights:

$$z_{\ldots,i+1,0} - z_{\ldots,i,0} \;=\; h_i \sum_{j=1}^{K} b_j\, f_{ij}$$

So it takes the same slice and generator the matching `@add_con_collocation` call took, over
a `collocation_itr`; the junction rows are derived from it. `f` appears, so the one-call-per-
distinct-right-hand-side rule applies here too.

```julia
@add_con_continuity(core, dae, cont1, z[1,c],
    z[2,c,i,k] for (c,i,k,t) in collocation_itr(dae, 1:Nc))
```

---

## `set_mesh!`

```julia
set_mesh!(dae, nodes, K; roots, basis, polynomial)
```

Attaches the mode and mesh to `dae`, in place. Same arguments as the two-argument `DAEta`
constructor, which calls it. Use it to fill in a `DAEta()`.

---

## Scope

This module provides the collocation pieces only: the collocation and continuity equations,
the weights and collocation points, and `DAEta` to track them across modes.

Everything else is plain ExaModels. Initial, terminal, and path conditions carry no
collocation content, so write them directly — and follow the same rule there, one `add_con`
per structurally distinct expression, with everything that merely varies in the iterator:

```julia
# every numeric initial condition in one call
ExaModels.@add_con(core, ic_fix,
    z[v,c,1,0] - val for (v,c,val) in [(1,1,0.0), (3,1,0.0)])

# the p-linked one is a different expression, so it needs its own
ExaModels.@add_con(core, ic_p,
    z[v,c,1,0] - p[m] for (v,c,m) in [(2,1,1)])
```

---

## Worked example

Van der Pol, driven to rest with minimum control effort. The running cost is transcribed as
an augmented state `z3`, so the objective is its terminal value.

```julia
using ExaModels, ExaModelsCollocation, MadNLP

nz, nu, Nc = 3, 1, 1
tf, N, K = 5.0, 19, 3

dae  = DAEta(range(0.0, tf; length = N + 1), K)
core = ExaModels.ExaCore(; concrete = Val(true))

@add_var_collocation(core, dae, z, 1:nz, 1:Nc)                            # k = 0,…,K
@add_var_collocation(core, dae, u, 1:nu, 1:Nc; include_boundary = false)  # k = 1,…,K
ExaModels.@add_var(core, p, 1:2; lvar = [-2.0, -1.0], uvar = [2.0, 1.0])
ExaModels.@add_par(core, th, [1.0])

itr = collocation_itr(dae, 1:Nc)

@add_con_collocation(core, dae, coll1, z[1,c],
    z[2,c,i,k] for (c,i,k,t) in itr)

@add_con_collocation(core, dae, coll2, z[2,c],
    th[1] * z[2,c,i,k] * (1 - z[1,c,i,k]^2) - z[1,c,i,k] + u[1,c,i,k] + p[2] * cos(t)
    for (c,i,k,t) in itr)

@add_con_collocation(core, dae, coll3, z[3,c],
    z[1,c,i,k]^2 + z[2,c,i,k]^2 + u[1,c,i,k]^2 for (c,i,k,t) in itr)

@add_con_continuity(core, dae, cont,
    z[v,c] for (v,c,i) in continuity_itr(dae, 1:nz, 1:Nc))

ExaModels.@add_con(core, ic_fix, z[v,c,1,0] - val for (v,c,val) in [(1,1,0.0), (3,1,0.0)])
ExaModels.@add_con(core, ic_p,   z[v,c,1,0] - p[m] for (v,c,m) in [(2,1,1)])

ExaModels.@add_obj(core, z[3,1,N,K] for _ in 1:1)

result = madnlp(ExaModels.ExaModel(core))
```

`z` is indexed `k = 0,…,K`, but `ExaModels.solution(result, z)` returns a plain 1-based
array, so that axis lands on `1,…,K+1`.
