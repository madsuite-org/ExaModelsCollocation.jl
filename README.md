# ExaModelsCollocation.jl

Data structure and helper functions for implementing orthogonal collocation in [ExaModels.jl](https://github.com/exanauts/ExaModels.jl).

### Feature Summary
- `DAEta` : data structure containing collocation metadata used by collocation helper functions
- `add_var_collocation`/`@add_var_collocation` : creates variable over every collocation point
- `add_con_collocation`/`@add_con_collocation` : creates collocation constraints over every collocation point
- `add_con_continuity`/`@add_con_continuity` : creates continuity constraints over every interval

For complete examples, refer to `test/vanderpol.jl` (optimal control) and `test/bruno.jl` (parameter
estimation).

---

## `DAEta`

```julia
DAEta(nodes, K; roots = GaussRadau(), basis = StateForm(), polynomial = Lagrange())
```
Creates a data object `DAEta` which contains collocation metadata used by the collocation helper functions.

### Arguments
- `nodes` : interval boundary placements for `N+1` boundaries, where `N` is the number of intervals
- `K` : degree of the interpolating polynomial (number of collocation points per interval)

### Keyword Arguments
- `roots` : collocation family, `GaussRadau()`, `GaussLegendre()`, or `GaussLobatto()`
- `basis` : differential-state representation, `StateForm()` or `DerivativeForm()`
- `polynomial` : interpolating polynomial, `Lagrange()` only, so it is left at the default

### Fields
- `mode` : `roots`, `basis`, `polynomial`, `weights`: `A` collocation, `b` continuity, `taus`
- `mesh` : `nodes`, `h` interval lengths, `t` time
- `vars`, `cons` : `NamedTuple`s of the named constrained handles for augmentation
- `blocks` : `VarBlock` collocation variable dimensions 

---

## `add_var_collocation`

```julia
add_var_collocation(core, dims...; include_boundary = true, name = nothing, kwargs...)
```

Adds variables with dimensions specified by `dims` over the collocation mesh in `CollocationExaCore` to
`core`. `dims` is the dimensions at a collocation point. The interval index over `1:N` and 
the collocation index over `krange` are appended. Returns `(core, CollocationVariable)`.

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
- **Named** (`@add_var_collocation(core, name, dims...)`): binds `name` to the new `Variable` in the local scope
  and registers it in `core` and `dae` for later retrieval as `core.name` or `model.name`.
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
collocation point of the mesh in `CollocationExaCore`. Returns `(core, CollocationConstraint)`.

### Arguments
- `z` : the collocated variable, from `add_var_collocation`
- `generator` : right-hand side function `f` of `z`

### Keyword Arguments
- `name` : when given as `Val(:name)`, registers the constraint in `core` for later retrieval as `core.name`. See `@add_con_collocation` for the idiomatic named interface.
- remaining kwargs passed on to `ExaModels.add_con`: `lcon`, `ucon`, `start`, `tag`

### Example
```julia
julia> c, z = add_var_collocation(c, 1:Nz, 1:Nc)

julia> c, decay = ExaModels.add_var(c, 1:Nz)

julia> itr = [(v,c) for v in 1:Nz, c in 1:Nc]

julia> c, coll = add_con_collocation(c, z,
           -decay[v] * z[v,c,i,k] # right-hand side function expression at collocation point i,k
           for (v,c) in itr)
```

---

## `@add_con_collocation`

```julia
@add_con_collocation(core, [name,] generator; kwargs...)
```

Macro interface for `add_con_collocation`. Updates `core` in the calling scope.
- **Named** (`@add_con_collocation(core, name, z[dims...], generator)`): binds `name` to the new `Constraint`
  in the local scope and registers it in `core` for later retrieval as `core.name` or `model.name`.
- **Anonymous** (`@add_con_collocation(core, z[dims...], generator)`): equivalent to
  `c, name = add_con_collocation(c, z, generator)`.

Accepts the same keyword arguments as `add_con_collocation`.

### Example
```julia
julia> @add_var_collocation(c, z, 1:4)

julia> @add_var_collocation(c, u, 1:3; include_boundary = false)

julia> @add_con_collocation(c, coll, z,  # v held at 1, c varying
           z[v,i,k] * u[l,i,k] * t^2  # the right-hand side function expression
           for (v,l) in itr)
```

---

## `add_con_continuity`

```julia
add_con_continuity(core, z; name = nothing, kwargs...)
```

Adds the continuity constraints for the variable `z` to `core`, enforcing each interval's terminal value
to equal the value at the next interval's left boundary node, for `i = 1,…,N-1`. Returns `(core, Constraint)`.

### Arguments
- `z` : a `CollocationVariable`, from `add_var_collocation`

### Keyword Arguments
- `name` : when given as `Val(:name)`, registers the constraint in `core` and `dae` for later retrieval as `dae.name` or `core.name`. See `@add_con_continuity` for the idiomatic named interface.
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
- **Named** (`@add_con_continuity(core, name, z)`): binds `name` to the new `CollocationConstraint`
  in the local scope and registers it in `core` for later retrieval as `core.name` or `model.name`.
- **Anonymous** (`@add_con_continuity(core, z)`): equivalent to
  `c, name = add_con_continuity(c, z)`.

Accepts the same keyword arguments as `add_con_continuity`.

### Example
```julia
julia> @add_con_continuity(c, cont, z)
```
