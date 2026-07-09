```@meta
CurrentModule = ExaModelsDAE
```

# ExaModelsDAE.jl

Automatically transcribes differential-algebraic equations (DAEs) into algebraic
constraints in [ExaModels.jl](https://github.com/exanauts/ExaModels.jl), so dynamic
optimization problems can be solved with the simultaneous (collocation-on-finite-elements)
method on CPU or GPU backends.

You supply the dynamics `dz/dt = f(z,y,u,p,theta,t)`, the initial condition `z₀`, and optional
algebraic / path / terminal constraints as plain Julia functions. A single call to
[`add_dae`](@ref) discretizes them and appends every structural constraint to an
`ExaModels.ExaCore`, returning the variable handles and mesh layout as a `dae` object. The
objective is then built separately against `dae`.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/mit-shin-group/ExaModelsDAE.jl")
```

## Contents

```@contents
Pages = ["usage.md", "examples.md"]
Depth = 2
```
