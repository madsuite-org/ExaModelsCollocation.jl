# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this package does

ExaModelsDAE.jl automatically transcribes differential-algebraic equations (DAEs) into algebraic constraints inside an [ExaModels.jl](https://github.com/exanauts/ExaModels.jl) `ExaCore`, so dynamic optimization problems can be solved on CPU or GPU backends. The user supplies plain Julia functions for the dynamics `dz/dt = f(z,y,u,p,t)`, initial conditions `z0(y,u,p)`, and optional algebraic/path constraints; the package discretizes them via a chosen collocation scheme and appends the resulting equations as constraints.

State/variable convention used throughout the API: `z` = differential state, `y` = algebraic state, `u` = control, `p` = parameter, `t` = time.

## Status: early scaffold

The package is a partial scaffold. `taus.jl`, `basis.jl`, and `polynomial.jl` are implemented; `daeta.jl` defines the `DAEta` struct (with TODOs). Many `src/dynamic/*.jl` files (`mesh.jl`, `initialize.jl`, `variables.jl`, `collocation.jl`, `continuity.jl`, the constraint files) and `src/utils.jl` are still **empty or stub files**, and `test/runtests.jl` has no real tests. Expect to add definitions rather than assume things work.

The **README is the spec for the intended public API**; treat it as the source of truth for how the pieces should fit together, and implement toward it.

## Architecture (intended)

`src/ExaModelsDAE.jl` is the module entry point. It `include`s files in dependency order and exports the single public entry point `add_dae`. The include order matters — `utils` first, then the `dynamic/` files (`basis`, `polynomial`, `taus`, `mesh`, `daeta`, `initialize`, …), then `exports`. `daeta.jl` (the `DAEta` struct) comes after `basis`/`polynomial` because it references `BasisWeights`.

The public flow (see README) is a single call:
```julia
core, data = EMD.add_dae(core, dzdt, z0;
    g, c, nodes, degree, basis, polynomial, roots)
```
which mutates the passed `ExaCore` and returns it alongside a `data` object. The discretization is configured by composable strategy types, each intended to live in its correspondingly named file:
- `basis.jl` — state/derivative representation: `StateForm()`, `DerivativeForm()`
- `polynomial.jl` — interpolating polynomial: `Lagrange()`
- `taus.jl` — collocation points: `GaussRadau()`, `GaussLegendre()`, `GaussLobatto()`
- `mesh.jl` — interval/node placement along `t` (auto-chosen if not supplied)

When implementing, keep this one-file-per-concept split and route the corresponding `add_dae` keyword argument to the matching module.

## Commands

Run from the repo root. The package uses the standard Julia/ExaModels workflow.

```julia
# Run the full test suite
julia --project=. -e 'using Pkg; Pkg.test()'

# Or from the Pkg REPL (press ] in a julia --project=. session)
]test

# Load/develop interactively
julia --project=. -e 'using ExaModelsDAE'

# Run a single testset while iterating (once tests exist), e.g.:
julia --project=. test/runtests.jl
```

Julia compat is `1.10+` (CI tests 1.11, 1.12, and `pre`). ExaModels compat is pinned to `0.11`. `test/Project.toml` declares test-only deps (`ExaModels`, `Test`); add new test dependencies there, not in the root `Project.toml`.
