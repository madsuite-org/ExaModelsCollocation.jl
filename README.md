# ExaModelsCollocation.jl

Helper functions for implementing orthogonal collocation in [ExaModels.jl](https://github.com/exanauts/ExaModels.jl).

### Feature Summary
- `CollocationExaCore` : an `ExaCore` containing collocation metadata used by collocation helper functions
- `add_var_collocation`/`@add_var_collocation` : creates variable over every collocation point
- `add_con_collocation`/`@add_con_collocation` : creates collocation constraints over every collocation point
- `add_con_continuity`/`@add_con_continuity` : creates continuity constraints over every interval

For complete examples, refer to `test/vanderpol.jl` (optimal control) and `test/bruno.jl` (parameter
estimation).

---

## `CollocationExaCore`

```julia
CollocationExaCore(nodes, K; roots = GaussRadau(), basis = StateForm(), 
  polynomial = Lagrange(), adaptive = false, kwargs...)
```
Creates an intermediate data object `ExaCore`, which contains collocation metadata used by collocation helper functions.

### Arguments
- `nodes` : interval boundary placements for `N+1` boundaries, where `N` is the number of intervals
- `K` : degree of the interpolating polynomial (number of collocation points per interval)

### Keyword Arguments
- `roots` : collocation family, `GaussRadau()`, `GaussLegendre()`, or `GaussLobatto()`
- `basis` : differential-state representation, `StateForm()` or `DerivativeForm()`
- `polynomial` : interpolating polynomial, `Lagrange()` only
- `adaptive` : whether interval widths are mutable `ExaModels` parameters
- remaining kwargs passed on to `ExaCore`: `backend`, `minimize`, `name`

### Properties
- `mode` : `roots`, `basis`, `polynomial`, `weights`: `A` collocation, `b` continuity, `taus`
- `mesh` : `nodes`, `h` interval lengths, `t` time (`hpar`, `tpar` if `adaptive = true`)
- `blocks` : `CollocationVariable` dimensions

### Example
```julia
julia> nodes = range(0.0, 5.0; length = 21) # interval boundary placements

julia> core = CollocationExaCore(nodes, 3) # N=20, K=3

julia> core = ExaCore(concrete = Val(true); tag = Collocation(nodes, 3)) # also works

julia> core = ExaCore(core; tag = Collocation(nodes, 3)) # also works
```

### `set_nodes!`

```julia
set_nodes!(core_or_model, nodes)
```

If `adaptive = true` for a `CollocationExaCore`, relocates the placement of `nodes`.

---

## `add_var_collocation`

```julia
add_var_collocation(core, dims...; include_boundary = true, name = nothing, kwargs...)
```

Adds variables with dimensions specified by `dims` over the collocation mesh in `CollocationExaCore` to
`core`. `dims` is the collocation variable dimensions, the interval index over `1:N` and 
the interpolation index over `krange` are appended. Returns `(core, CollocationVariable)`.

### Keyword Arguments
- `include_boundary` : `true` (default) gives `k = 0,…,K` with `k = 0` being the interval-left boundary node, `false` gives `k = 1,…,K`
- `name` : when given as `Val(:name)`, registers the variable in `core` for later retrieval as `core.name`. See `@add_var_collocation` for the idiomatic named interface.
- remaining kwargs passed on to `ExaModels.add_var`: `start`, `lvar`, `uvar`, `tag`

### Example
```julia
julia> c, z = add_var_collocation(c, 1:3, 1:2) # z[v,c,i,k], 3 × 2 × N × (K+1)
```

---

## `@add_var_collocation`

```julia
@add_var_collocation(core, [name,] dims...; kwargs...)
```

Macro interface for `add_var_collocation`. Updates `core` in the calling scope.
- **Named** (`@add_var_collocation(core, name, dims...)`): binds `name` to the new `CollocationVariable` in the local scope
  and registers it in `core` for later retrieval as `core.name` or `model.name`.
- **Anonymous** (`@add_var_collocation(core, dims...)`): equivalent to `c, name = add_var_collocation(c, dims...)`.

Accepts the same keyword arguments as `add_var_collocation`.

### Example
```julia
julia> @add_var_collocation(c, z, 1:3) # z (z[v,i,k], 3 × N × (K+1)) is now in scope; core.z also works
```

---


## `add_con_collocation`

```julia
add_con_collocation(core, z, generator; name = nothing, kwargs...)
```

Adds the collocation constraints for the variable `z` to `core`, enforcing `dz/dt = f` at every
collocation point of the mesh in `CollocationExaCore`. Returns `(core, Constraint)`.

### Arguments
- `z` : a `CollocationVariable` from `add_var_collocation`
- `generator` : right-hand side function `f` for a `CollocationVariable`

### Keyword Arguments
- `name` : when given as `Val(:name)`, registers the constraint in `core` for later retrieval as `core.name`. See `@add_con_collocation` for the idiomatic named interface.
- remaining kwargs passed on to `ExaModels.add_con`: `lcon`, `ucon`, `start`, `tag`

### Example
```julia
julia> c, z = add_var_collocation(c, 1:Nz, 1:Nc)

julia> c, decay = ExaModels.add_var(c, 1:Nz)

julia> itr = [(v, c, i, k, mesh.t[i,k]) for v in 1:Nz, c in Nc, i in 1:N, k in 1:K]

julia> c, coll = add_con_collocation(c, z,
           -decay[v]*z[v,c,i,k] # right-hand side function expression at collocation point i,k
           for (v, c, i, k, t) in itr)
```

---

## `@add_con_collocation`

```julia
@add_con_collocation(core, [name,] z, generator; kwargs...)
```

Macro interface for `add_con_collocation`. Updates `core` in the calling scope.
- **Named** (`@add_con_collocation(core, name, z, generator)`): binds `name` to the new `Constraint`
  in the local scope and registers it in `core` for later retrieval as `core.name` or `model.name`.
- **Anonymous** (`@add_con_collocation(core, z, generator)`): equivalent to
  `c, name = add_con_collocation(c, z, generator)`.

Accepts the same keyword arguments as `add_con_collocation`.

### Example
```julia
julia> @add_var_collocation(c, z, 1:4)

julia> @add_var_collocation(c, u, 1:3; include_boundary = false)

julia> itr = [(v, l[v], i, k, mesh.t[i,k]) for v in 1:4, i in 1:N, k in 1:K]

julia> @add_con_collocation(c, coll, z,
           z[v,i,k]*u[l,i,k]*cos(t) # the right-hand side function expression
           for (v, l, i, k, t) in itr)
```

---

## `add_con_continuity`

```julia
add_con_continuity(core, z; name = nothing, kwargs...)
```

Adds the continuity constraints for a `CollocationVariable` to `core`, enforcing each interval's terminal value
to equal the value at the next interval's left boundary node, for `i = 1,…,N-1`. Returns `(core, Constraint)`.

### Arguments
- `z` : a `CollocationVariable` from `add_var_collocation`

### Keyword Arguments
- `name` : when given as `Val(:name)`, registers the constraint in `core` for later retrieval as `core.name` or `model.name`. See `@add_con_continuity` for the idiomatic named interface.
- remaining kwargs passed on to `ExaModels.add_con`: `lcon`, `ucon`, `start`, `tag`

### Example
```julia
julia> c, cont = add_con_continuity(c, z)
```

---

## `@add_con_continuity`

```julia
@add_con_continuity(core, [name,] z; kwargs...)
```

Macro interface for `add_con_continuity`. Updates `core` in the calling scope.
- **Named** (`@add_con_continuity(core, name, z)`): binds `name` to the new `Constraint`
  in the local scope and registers it in `core` for later retrieval as `core.name` or `model.name`.
- **Anonymous** (`@add_con_continuity(core, z)`): equivalent to
  `c, name = add_con_continuity(c, z)`.

Accepts the same keyword arguments as `add_con_continuity`.

### Example
```julia
julia> @add_con_continuity(c, cont, z)
```
