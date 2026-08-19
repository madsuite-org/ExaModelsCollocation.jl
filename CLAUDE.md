# ExaModelsCollocation.jl

Keep this file small. It holds only fixed conventions and the discretization identities that do
not change with the code. Do not restate anything recoverable elsewhere: signatures, keywords,
fields and examples are in the README, the internals are in the source and its comments, and the
concept summaries are in the strategy-file headers (`taus.jl`, `basis.jl`, `polynomial.jl`,
`mesh.jl`). If a fact is in the code, a docstring, or the README, it does not belong here.

## Scope and conventions

- An internal helper module, not a package that owns your problem: no entry point, no solver, no
  integrator. Assemble with `ExaModels.add_*` and reach for a helper only where collocation
  changes something. Initial, terminal, and path conditions are plain `ExaModels.@add_con`.
- Dependencies are **ExaModels** and **FastGaussQuadrature** only. Do not add others.
- **One file per exported symbol, named after it**, under `src/api/`, and `test/` mirrors `src/`
  file for file. The three builders (`add_var_collocation`, `add_con_collocation`,
  `add_con_continuity`) do not grow to a fourth. `src/api/macros.jl` is only what the macros share.
- The README is the API reference, kept short and shaped like ExaModels' own `add_var` docs.
  Rationale and derivations live in `src` comments or `test/`, never in the README.

## Usage traps invisible in the code

- **`i`, `k`, `t` are reserved inside a collocation right-hand side.** They name the mesh whether
  or not the iterator binds them, so a caller's own value under one of those names is silently
  replaced.
- A method written on `c::CollocationExaCore` must **restate ExaModels' `VT <: AbstractVector{T}`
  bound in its `where` clause**, or it is ambiguous with ExaModels' `getproperty`/`show` and fails
  at the first `add_var`, not at load.

## The discretization (Biegler, Nonlinear Programming, Ch. 10.2.1)

`N` counts intervals, not boundaries: 21 nodes give `N = 20`. `j` indexes the basis polynomial
(and the mesh, `t[i,j]`), `k` the collocation point. With `f_ij = f(z[…,i,j], t[i,j])`:

| | collocation | continuity |
|---|---|---|
| `StateForm` | (10.7) `Σⱼ₌₀..ᴷ A[j,k] z[…,i,j] = h[i] f_ik` | (10.14a) `Σⱼ₌₀..ᴷ b[j] z[…,i,j] = z[…,i+1,0]` |
| `DerivativeForm` | (10.8) `z[…,i,k] − z[…,i,0] = h[i] Σⱼ₌₁..ᴷ A[j,k] f_ij` | (10.15a) `z[…,i+1,0] − z[…,i,0] = h[i] Σⱼ₌₁..ᴷ b[j] f_ij` |

When the family collocates `τ = 1` (Radau, Lobatto) continuity collapses to the exact node
identity `z[…,i,K] = z[…,i+1,0]`, so only `DerivativeForm` on `GaussLegendre` puts `h` in a
junction row. Terminal error decays at `O(h²ᴷ)` for Legendre, `O(h²ᴷ⁻¹)` for Radau,
`O(h²ᴷ⁻²)` for Lobatto: a mis-weighted stencil still converges but loses order.

## Testing

- Exercise **every build path**: each roots family × each basis × a range of `K`. Pin the bases
  against each other (algebraically equivalent discretizations must reach the same solution) and
  against the order above, not against stored numbers.
- Whole models live in `examples/` and **CI does not run them**, so `Pkg.test()` does not enforce
  the Bruno objective pin.
