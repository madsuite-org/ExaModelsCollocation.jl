# Usage

## Problem form

Define a differential-algebraic system

```math
\frac{\mathrm{d}z}{\mathrm{d}t} = f(z,y,u,p,\theta,t),
\quad
z(t_0) = z_0(y,u,p,\theta),
\quad
g(z,y,u,p,\theta,t) = 0,
\quad
c(z,y,u,p,\theta,t) \le 0,
\quad
h_E(z(t_f),p,\theta) = 0,
\quad
t \in [t_0,t_f]
```

as Julia functions, each **returning a vector**:

```julia
f(z,y,u,p,θ,t)  = ...   # dz/dt
z0(y,u,p,θ)     = ...   # initial condition at t = t₀
g(z,y,u,p,θ,t)  = ...   # algebraic equality (= 0)
c(z,y,u,p,θ,t)  = ...   # path inequality    (≤ 0)
hE(zf,p,θ)      = ...   # terminal equality  (= 0)
```

where

* ``z(t)`` is a differential state variable
* ``y(t)`` is an algebraic state variable
* ``u(t)`` is a control variable
* ``p`` is a free decision parameter (an optimization variable)
* ``\theta`` is a mutable parameter (an ExaModels parameter, updatable via `set_parameter!` for parametric optimization)
* ``t`` is time.

`z0` and `hE` take **no `t`** argument — they are by definition the `t₀` and `t_f` points.

!!! note "Return a vector, don't mutate"
    Write each function to **return** a vector (not the in-place `dz[i] = …` form used by
    OrdinaryDiffEq). ExaModels traces the function **once** at construction to build an
    expression graph and then compiles it — the function is never called during the solve,
    so the in-place form buys no performance, and the return length is what supplies the
    problem dimensions. Write the body type-generically (`::Real`-like; no `::Float64`,
    `length(z)`, or mutation) so the same function works for symbolic tracing and, when
    used, numeric simulation.

## `add_dae`

```julia
using ExaModels
using ExaModelsDynamic as EMD

core = ExaModels.ExaCore(; backend = CUDA.Backend(), concrete = Val(true))

tspan = (t0, tf)                     # time horizon; a single number T ⇒ (zero(T), T)
init  = (u = ..., p = ..., θ = ...)  # initial guesses / parameter values

core, dae = EMD.add_dae(core, f, z0, tspan, init;
  g          = g,                           # Algebraic equality constraints g(x) = 0
  c          = c,                           # Path constraints c(x) ≤ 0
  hE         = hE,                          # Terminal constraints hE(z_f) = 0
  u          = u,                           # Fixed control profile u(t) (decision variable if omitted)
  bounds     = (z = (zL, zU), y = ..., u = ..., p = ...), # Variable bounds / simple path constraints
  nodes      = tstops,                      # Interval node placements (auto-chosen if not supplied)
  degree     = K,                           # Degree of interpolating polynomial (defaults to 4)
  basis      = EMD.Lagrange(),              # Derivative representation {Lagrange(), RungeKutta()}
  polynomial = EMD.LagrangeInterpolation(), # Interpolating polynomial
  roots      = EMD.GaussRadau(),            # Collocation points {GaussRadau(), GaussLegendre(), GaussLobatto()}
)
```

`add_dae` appends the collocation, continuity, initial-condition, algebraic, path, and
terminal constraints to `core` and returns `dae`, which holds the variable handles
(`dae.z`, `dae.z_f`, `dae.p`, `dae.θ`, …) and the mesh/collocation layout. `tspan` and
`init` are required positional arguments; everything else is keyword. **The objective is
built separately** against `dae` (see [Examples](examples.md)).

## `init` — guesses and presence

`init` supplies the initial guesses **and** declares which variable classes are present:

* `init.p`, `init.θ` — length-`np`/`nθ` vectors (time-invariant). `init.θ` is the
  parameter *value* (not merely a guess).
* `init.u`, `init.z`, `init.y` — either a scalar / length-`n` vector (broadcast to every
  collocation point) or a callable `t -> vector` (a time-varying guess, sampled at the
  collocation points).

An entry that is `[]` or omitted means **that class is absent**: `init.p = []` ⇒ no free
parameters, `init.θ = []` ⇒ no parameters. Likewise `u = []` or omitted means **no fixed
control profile** — the control becomes a decision variable (guessed by `init.u`), or is
absent if `init.u` is empty too. A fixed control profile is passed via the `u` keyword as a
callable `t -> vector` (the same shape as an `init.u` guess).

Each `init` entry maps to the ExaModels `add_var` `start`. Problem dimensions are inferred
from `init`, the `u` profile, and the function output lengths, so no explicit dimension
arguments are needed.

## `bounds` — and simple path constraints

Each `bounds` entry becomes an ExaModels variable bound (`lvar`/`uvar`) applied at **every
collocation point**, so a bound doubles as a simple (box) path constraint over the whole
horizon. Each side is a scalar (broadcast) or a per-component vector:

```julia
bounds = (z = ([-Inf, -Inf, -Inf], [2.0, Inf, Inf]),)  # enforces x₁(t) ≤ 2 for all t
```

Use `bounds` for simple state/control limits (handled natively by the solver); reserve `c`
for general nonlinear path constraints. This mirrors the bounds-vs-constraint split in
ExaModels, JuMP, and InfiniteOpt.

## Discretization

The collocation scheme is composed from three interchangeable choices, all sharing the same
roots:

* `roots` — collocation points: `GaussRadau()`, `GaussLegendre()`, `GaussLobatto()`.
* `basis` — derivative representation: `Lagrange()` (differentiation form) or
  `RungeKutta()` (Butcher integration form).
* `polynomial` — interpolating polynomial: `LagrangeInterpolation()`.

`nodes` gives the finite-element boundaries; if omitted, a mesh is generated (uniformly, or
adaptively from a forward solve when an integrator and simulation inputs are available).
`degree` sets the number of collocation points per element.
