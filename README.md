# ExaModelsCollocation.jl

Data structure and helper functions for orthogonal collocation in [ExaModels.jl](https://github.com/exanauts/ExaModels.jl).

For complete examples, refer to `test/vanderpol.jl` (optimal control) and `test/bruno.jl` (parameter
estimation).

---

## `DAEta`

```julia
DAEta(nodes, K; roots = GaussRadau(), basis = StateForm(), polynomial = Lagrange())
```
Collocation metadata used by the collocation helper functions.

### Arguments
- `nodes` : interval boundary placements (`N+1` interval boundaries, where `N` is the number of intervals).
- `K` : degree of the interpolating polynomial

### Keyword Arguments
- `roots` : collocation family (`GaussRadau()`, `GaussLegendre()`, or `GaussLobatto()`)
- `basis` : differential-state representation (`StateForm()` or `DerivativeForm()`)
- `polynomial` : interpolating polynomial (`Lagrange()` only)

### Fields
- `mode` : `roots`, `basis`, `polynomial`, `weights` (`A` collocation, `b` continuity, `taus`)
- `mesh` : `nodes`, `h` interval lengths, `t` time
- `vars`, `cons` : `NamedTuple`s of the named constrained handles for augmentation.
- `blocks` : `VarBlock` collocation variable dimensions 

---

## `add_var_collocation`

```julia
add_var_collocation(core, dae, dims...; include_boundary = true, name = nothing, kwargs...)
```

Adds a variable block laid out over the collocation mesh of `dae`. `dims` is the shape at a
single point in time; the interval index over `1:N` and the collocation index over `krange`
are appended. Returns `(core, var)`, `dae` is mutated in place.

### Keyword Arguments
- `include_boundary` : `true` (default) gives `k = 0,…,K`, carrying the interval-left boundary node that continuity needs; `false` gives `k = 1,…,K`, the collocation points alone.
- remaining kwargs passed on to `ExaModels.add_var`. `start`, `lvar`, `uvar`, and `tag` are shaped to the **allocated** block, `(dims..., 1:N, krange)`.

### Example
```julia
julia> c, z = add_var_collocation(c, dae, 1:3, 1:2);   # z[v,c,i,k], 3 × 2 × N × (K+1)
```

---

## `@add_var_collocation`

```julia
@add_var_collocation(core, dae, name, dims...; kwargs...)
```

Macro form of `add_var_collocation`, relating to it as `ExaModels.@add_var` does to
`add_var`: `name` is written bare and bound in the calling scope along with `core`.

### Example
```julia
julia> @add_var_collocation(c, dae, u, 1:1, 1:2; include_boundary = false);   # k = 1,…,K
```

---

## `collocation_itr`

```julia
collocation_itr(dae, leads...)
```

Builds an iterator for `@add_con_collocation` over the collocation points by appending
`i, k, t` to `leads`, where `t = mesh.t[i,k]`.

### Arguments
- `leads` : variable dimensions that vary across the constraint

### Example
```julia
julia> itr = collocation_itr(dae, 1:Nc);         # (c, i, k, t)

julia> itr = collocation_itr(dae, 1:Nz, 1:Nc);   # (v, c, i, k, t)
```

---

## `@add_con_collocation`

```julia
@add_con_collocation(core, dae, name, z[leads...], generator; kwargs...)
```

Adds the collocation residual for a state slice, enforcing `dz/dt = f` at every collocation
point. With `f_ij = f(z[…,i,j], t[i,j])`, for `k = 1,…,K`:

| `basis` | residual |
|---|---|
| `StateForm` | `Σⱼ₌₀..ᴷ A[j,k] z[…,i,j] = h[i] f_ik` |
| `DerivativeForm` | `z[…,i,k] − z[…,i,0] = h[i] Σⱼ₌₁..ᴷ A[j,k] f_ij` |

`h[i]` is attached by the macro. Updates `core` and `dae` in the calling scope and binds `name`.

### Arguments
- `z[leads...]` : the collocated variable. Literal indices pin a dimension; names bound by the iterator vary with it
- `generator` : `f` over a `collocation_itr`. One call per structurally distinct `f`, and no more

### Keyword Arguments
- passed on to `ExaModels.add_con`

### Example
```julia
julia> @add_con_collocation(c, dae, coll1, z[1,c],  # pin v = 1, vary c
           z[2,c,i,k]                               # the right-hand side function expression
           for (c,i,k,t) in collocation_itr(dae, 1:Nc));

julia> c, decay = ExaModels.add_var(c, Nz);

julia> @add_con_collocation(c, dae, coll, z[v,c],   # one call, every v
           -decay[v] * z[v,c,i,k]                   # the right-hand side function expression
           for (v,c,i,k,t) in collocation_itr(dae, 1:Nz, 1:Nc));
```

---

## `continuity_itr`

```julia
continuity_itr(dae, leads...)
```

Builds an iterator for a `StateForm` `@add_con_continuity` over the state slots to tie. The
junction index `i = 1,…,N-1` is appended by the macro, not carried here.

### Arguments
- `leads` : variable dimensions that vary across the constraint

### Example
```julia
julia> itr = continuity_itr(dae, 1:Nz, 1:Nc);    # (v, c)
```

---

## `@add_con_continuity`

```julia
@add_con_continuity(core, dae, name, z[leads...] for (leads...) in itr; kwargs...)
@add_con_continuity(core, dae, name, z[leads...], generator; kwargs...)
```

Ties each interval's terminal polynomial value to the next interval's boundary node, for
`i = 1,…,N-1`. The form is set by `dae.basis`; passing the wrong one throws.

| `basis` | residual | call |
|---|---|---|
| `StateForm` | `Σⱼ₌₀..ᴷ b[j] z[…,i,j] = z[…,i+1,0]` | slice only, over a `continuity_itr` |
| `DerivativeForm` | `z[…,i+1,0] − z[…,i,0] = h[i] Σⱼ₌₁..ᴷ b[j] f_ij` | slice and generator, over a `collocation_itr` |

`StateForm` needs no `f`, so one call covers every slot. `DerivativeForm` contains `f`, so it
takes the same slice and generator as its `@add_con_collocation` call.

### Arguments
- `z[leads...]` : the collocated variable, as in `@add_con_collocation`
- `generator` : `DerivativeForm` only — `f` over a `collocation_itr`

### Keyword Arguments
- passed on to `ExaModels.add_con`

### Example
```julia
julia> @add_con_continuity(c, dae, cont,                            # StateForm
           z[v,c] for (v,c) in continuity_itr(dae, 1:Nz, 1:Nc));

julia> @add_con_continuity(c, dae, cont1, z[1,c],                   # DerivativeForm
           z[2,c,i,k] for (c,i,k,t) in collocation_itr(dae, 1:Nc));
```

---

## `block`

```julia
block(dae, var) -> VarBlock
```

The layout recorded for a handle: its declared `dims` and its `krange`. Looked up by handle
identity, so it works whether or not the block was named.
