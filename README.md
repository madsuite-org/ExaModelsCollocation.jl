# ExaModelsCollocation.jl

Helper functions for orthogonal collocation in [ExaModels.jl](https://github.com/madsuite-org/ExaModels.jl).

### Feature Summary
- `CollocationExaCore` : an `ExaCore` containing collocation metadata
- `add_var_collocation`/`@add_var_collocation` : creates variable over every collocation point
- `add_con_collocation`/`@add_con_collocation` : creates collocation constraints over every collocation point
- `add_con_continuity`/`@add_con_continuity` : creates continuity constraints over every interval

Refer to `examples/*` for complete examples.

---

## `CollocationExaCore`

```julia
CollocationExaCore(nodes, K; roots = GaussRadau(), basis = StateForm(), 
  polynomial = Lagrange(), adaptive = false, kwargs...)
```
Creates an intermediate data object `ExaCore`, which contains collocation metadata used by collocation helper functions.

### Arguments
- `nodes` : interval boundary placements for `N+1` boundaries for `N` intervals
- `K`     : degree of interpolating polynomial

### Keyword Arguments
- `roots`      : collocation family, `GaussRadau()`, `GaussLegendre()`, or `GaussLobatto()`
- `basis`      : differential-state representation, `StateForm()` or `DerivativeForm()`
- `polynomial` : interpolating polynomial, `Lagrange()`
- `adaptive`   : whether interval widths are mutable `ExaModels` parameters
- remaining kwargs passed on to `ExaCore`: `backend`, `minimize`, `name`

### Fields
- `mode`  : `roots`, `basis`, `polynomial`, `weights`
- `mesh`  : `nodes`, `h` interval lengths, `t` time
- `block` : `CollocationVariable` dimensions
- `resid` : `CollocationVariable` right-hand side functions
- `N`, `K`, `nodes`, `adaptive`

### Example
```julia
julia> nodes = range(0.0, 5.0; length = 21) # 21 interval boundary placements

julia> core = CollocationExaCore(nodes, 3) # N=20, K=3

julia> core = ExaCore(concrete = Val(true); tag = Collocation(nodes, 3)) # also works

julia> core = ExaCore(core; tag = Collocation(nodes, 3)) # also works
```

---

## `set_nodes!`

```julia
set_nodes!(model, nodes)
```

Relocates the placement of `nodes` of a `CollocationExaCore` model, given `adaptive = true`.

---

## `add_var_collocation`

```julia
add_var_collocation(core, dims...; include_boundary = true, name = nothing, kwargs...)
```

Adds a `CollocationVariable` with dimensions `dims` and appended mesh indicies from `CollocationExaCore` to `core`.
Mesh indicies consist of the interval index `i in 1:N` and interpolation index `k in krange`.
Returns `(core, CollocationVariable)`.

### Keyword Arguments
- `include_boundary` : `true` for `k = 0,…,K`, `false` for `k = 1,…,K`
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
julia> @add_var_collocation(c, z, 1:3) # z is now in scope; core.z also works
```

---


## `add_con_collocation`

```julia
add_con_collocation(core, z[dims...] => generator; name = nothing, kwargs...)
```

Adds the collocation constraints for the `CollocationVariable` to `core`, enforcing `dz/dt = f` at every
collocation point of the mesh in `CollocationExaCore`. Returns `(core, Constraint)`.

### Arguments
- `z` : a `CollocationVariable` from `add_var_collocation`
- `dims...` : the indicies for `CollocationVariable` over which the collocation constraints are added
- `generator` : right-hand side function `f` for a `CollocationVariable

### Keyword Arguments
- `name` : when given as `Val(:name)`, registers the constraint in `core` for later retrieval as `core.name`. See `@add_con_collocation` for the idiomatic named interface.
- remaining kwargs passed on to `ExaModels.add_con`: `lcon`, `ucon`, `start`, `tag`

### Example
```julia
julia> c, z = add_var_collocation(c, 1:Nz, 1:Nexp)

julia> c, rate = ExaModels.add_var(c, 1:Nz)

julia> itr = [(v, exp) for v in 1:Nz, exp in 1:Nexp]

julia> c, coll = add_con_collocation(c, 
           z[v,exp] => -rate[v]*z[v,exp] + rate[v]*cos(t) # right-hand side function expression added for z[v,exp], can use t
           for (v, exp) in itr) # automatically iterated over all N,K with t included
```

---

## `@add_con_collocation`

```julia
@add_con_collocation(core, [name,] z[dims...], generator; kwargs...)
```

Macro interface for `add_con_collocation`. Updates `core` in the calling scope.
- **Named** (`@add_con_collocation(core, name, z[dims...], generator)`): binds `name` to the new `Constraint`
  in the local scope and registers it in `core` for later retrieval as `core.name` or `model.name`.
- **Anonymous** (`@add_con_collocation(core, z[dims...], generator)`): equivalent to
  `c, name = add_con_collocation(c, z[dims...] => generator)`.

Accepts the same keyword arguments as `add_con_collocation`.

### Example
```julia
julia> @add_var_collocation(c, z, 1:4)

julia> @add_var_collocation(c, u, 1:3; include_boundary = false)

julia> itr = [(v, l[v]) for v in 1:4]

julia> @add_con_collocation(c, coll, z[v],
           z[v]*u[l]*cos(t) # right-hand side function expression added for z[v], can freely use t
           for (v, l) in itr) # automatically iterated over all N,K with t included
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
