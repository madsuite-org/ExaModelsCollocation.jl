# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this package does

ExaModelsDAE.jl automatically transcribes differential-algebraic equations (DAEs) into algebraic constraints inside an [ExaModels.jl](https://github.com/exanauts/ExaModels.jl) `ExaCore`, so dynamic optimization problems can be solved on CPU or GPU backends. The user supplies plain Julia functions for the dynamics `dz/dt = f(z,y,u,p,t)`, initial conditions `z0(y,u,p)`, and optional algebraic/path constraints; the package discretizes them via a chosen collocation scheme and appends the resulting equations as constraints.

State/variable convention used throughout the API: `z` = differential state, `y` = algebraic state, `u` = control, `p` = parameter, `t` = time.

## Status: early scaffold

The package is a skeleton. Most `src/*.jl` files (`basis.jl`, `roots.jl`, `nodes.jl`, `initialize.jl`, `exports.jl`, `utils.jl`) are **empty or stub files**, `polynomial.jl` and `structs.jl` contain incomplete definitions, and `test/runtests.jl` has no real tests. `structs.jl` references `AbstractNLPModel`/`NLPModelsMeta`/`Counters` that are not yet imported, so the module may not load cleanly — expect to add dependencies and definitions rather than assume things work.

The **README is the spec for the intended public API**; treat it as the source of truth for how the pieces should fit together, and implement toward it.

## Architecture (intended)

`src/ExaModelsDAE.jl` is the module entry point. It `include`s files in dependency order and exports the single public entry point `add_dae`. The include order matters — `structs.jl` first, then `utils`, `initialize`, `nodes`, `basis`, `polynomial`, `roots`, then `exports`.

The public flow (see README) is a single call:
```julia
core, data = EMD.add_dae(core, dzdt, z0;
    g, c, nodes, degree, basis, polynomial, roots)
```
which mutates the passed `ExaCore` and returns it alongside a `data` object. The discretization is configured by composable strategy types, each intended to live in its correspondingly named file:
- `basis.jl` — derivative representation: `Lagrange()`, `RungeKutta()`
- `polynomial.jl` — interpolating polynomial: `LagrangeInterpolation()`
- `roots.jl` — collocation points: `GaussRadau()`, `GaussLegendre()`, `GaussLobatto()`
- `nodes.jl` — interval/node placement along `t` (auto-chosen if not supplied)

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
