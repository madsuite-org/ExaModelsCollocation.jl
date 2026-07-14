# `add_dae` Front-End Design Specification

Status: **design draft** (no implementation yet). Formalizes what the front-end API
must expose to be complete. Formulation follows Biegler, *Nonlinear Programming:
Concepts, Algorithms, and Applications to Chemical Processes* (2010), Ch. 10,
"Simultaneous Methods for Dynamic Optimization" — the collocation-on-finite-elements
NLP (Eq. 10.19), generalized so the collocation constraint can be written in either a
**Lagrange (differentiation)** or **Runge–Kutta (integration)** form.

## 1. Design goals

1. **One call does the transcription.** `add_dae` discretizes the DAE and appends every
   structural constraint (collocation, continuity, initial condition, algebraic, path,
   terminal) to the `ExaCore`. It does **not** set an objective.
2. **Return complete metadata.** `add_dae` returns a `dae` object carrying the variable
   handles and the full collocation/mesh layout. The objective — and any extra
   constraints — are written **manually** by the user against `dae`, decoupled from
   transcription.
3. **General basis.** Support Eq. 10.19 (Lagrange interpolation + Lagrange basis) *and*
   the equivalent Runge–Kutta form, selectable via `basis`, sharing the same roots.
4. **Complete problem data.** Accept initial conditions, variable bounds, and the
   terminal equality `hₑ(zf) = 0` (Eq. 10.19h).
5. **Parametric hooks.** Every user function gains a `theta` argument that becomes ExaModels
   `@add_par` parameters — fixed at solve time, mutable afterward via `set_parameter!`
   for parametric / sensitivity studies without rebuilding.
6. **Simulate-to-initialize.** When possible, a single forward DAE solve at `(p0, theta0,
   u(t))` supplies (a) all problem dimensions, (b) warm-start guesses for the discretized
   states, and (c) an adaptive mesh. The forward solver is an **optional dependency**,
   loaded only when it is actually needed to generate a mesh.

## 2. Mathematical formulation

Continuous DAE optimal-control problem on `t ∈ [t₀, t_f]`:

```
dz/dt = f(z, y, u, p, theta, t)                       (differential)
    0 = g(z, y, u, p, theta, t)                        (algebraic)
    0 ≥ c(z, y, u, p, theta, t)                        (path / inequality)
 z(t₀) = z₀(y, u, p, theta)                            (initial condition)
    0 = hₑ(zf, p, theta)                              (terminal condition, 10.19h)
```
Variable classes: `z` differential state, `y` algebraic state, `u` control, `p`
time-invariant **decision** parameters (NLP unknowns via `add_var`), `theta` fixed ExaModels
parameters (via `add_par`, mutable with `set_parameter!`), `t` time. `t` is placed last in
every user signature (after `theta`) so it stays the trailing argument. Every function
**returns a full vector** (length `nz` for `f`/`z₀`, `ny` for `g`, etc.) and must be
written **type-generically** (`::Real`-like, no `::Float64`, no `length(z)`/mutation) so
the same function works for symbolic tracing *and* numeric simulation.

**Out-of-place (return), not in-place (`dz[i]=…`).** Unlike OrdinaryDiffEq — which prefers
in-place RHS to avoid per-step allocation because it calls `f` every integration step —
ExaModels traces `f` **once at construction** to build an expression graph, then compiles
that graph; the user's `f` is never called during the solve. So in-place buys no
performance here, and the return form is required anyway: it drops straight into the
per-index constraint generators (immutable expression nodes) and gives `nz = length(f(…))`
for dimension inference (an in-place `dz` would need `nz` up front to allocate — circular).

### Mesh and collocation

- Horizon split into `N` finite elements with boundaries (nodes)
  `t₀ = τ̂₀ < τ̂₁ < … < τ̂_N = t_f`; element `i` has length `hᵢ = τ̂ᵢ − τ̂ᵢ₋₁`.
- Each element carries `K` interior collocation points at the roots `τ₁,…,τ_K ∈ (0,1]`
  of the chosen family (`GaussRadau` / `GaussLegendre` / `GaussLobatto`); `K` = `degree`.
- Collocation time in element `i`: `t_{i,j} = τ̂ᵢ₋₁ + hᵢ τ_j`.

Two families of variables:
- **Collocation states** `z_{i,j}, y_{i,j}, u_{i,j}` for `i = 1..N`, `j = 1..K`.
- **Boundary (element-junction) states** `zb_i` for `i = 0..N`, where `zb₀ = z(t₀)` and
  `zb_N = zf`. These carry continuity between elements and give the objective a clean
  handle for `zf = zb_N`.

### Collocation constraint — two interchangeable bases

Let `ℓ_k(·)` be the Lagrange basis over `{0, τ₁,…,τ_K}` and `ℓ̇` its derivative.

- **`Lagrange()` (differentiation form, Eq. 10.19b):**
  ```
  Σ_{k=0}^{K} zc_{i,k} · ℓ̇_k(τ_j)  =  hᵢ · f(z_{i,j}, y_{i,j}, u_{i,j}, p, theta, t_{i,j})
  ```
  where `zc_{i,0} = zb_{i-1}` and `zc_{i,k>0} = z_{i,k}`.
- **`RungeKutta()` (integration form):** with Butcher coefficients `a_{jk}`, `b_j` from the
  same roots:
  ```
  z_{i,j} = zb_{i-1} + hᵢ · Σ_{k=1}^{K} a_{jk} · f(z_{i,k}, …, theta, t_{i,k})
  ```

### Structural constraints appended by `add_dae`

| # | Name | Equation | Rows |
|---|------|----------|------|
| C1 | collocation | one of the two forms above | `nz · N · K` |
| C2 | continuity | `zb_i = Σ_{k=0}^{K} ℓ_k(1) · zc_{i,k}` (`= z_{i,K}` for Radau) | `nz · N` |
| C3 | initial condition | `zb₀ = z₀(y_{1,·}, u_{1,·}, p, theta)` | `nz` |
| C4 | algebraic | `g(z_{i,j}, y_{i,j}, u_{i,j}, p, theta, t_{i,j}) = 0` | `ny · N · K` |
| C5 | path | `c(z_{i,j}, …, theta, t_{i,j}) ≤ 0` | `nc · N · K` |
| C6 | terminal (10.19h) | `hₑ(zb_N, p, theta) = 0` | `nhE` |

Bounds `z♭ ≤ z ≤ z♯`, etc. are applied as `lvar/uvar` on the variable blocks at **every
collocation point**, so each `bounds` entry doubles as a **simple (box) path constraint**
over the whole horizon (e.g. `bounds.z` enforces `zL ≤ z(t) ≤ zU` for all `t`). Each side
is a scalar (broadcast) or a per-component vector — a pass-through to ExaModels `lvar/uvar`
(`Number | AbstractArray | Generator`). Use `bounds` for simple state/control limits
(solver-native, cheap); reserve `c` (C5) for general nonlinear path constraints. This
mirrors the bounds-vs-constraint split in ExaModels, JuMP, and InfiniteOpt.

## 3. Proposed `add_dae` signature

Mirrors the README. `tspan` and `init` are **required positional** args (`init` always
supplies `np`/`ntheta`); `u`/`bounds` and discretization options are keyword.

```julia
core, dae = EMD.add_dae(
    core,
    f,                  # dz/dt = f(z, y, u, p, theta, t)  → length-nz vector
    z0,                 # z(t₀) = z0(y, u, p, theta)        → length-nz vector   (NO t — it is the t₀ point)
    tspan,              # time horizon (t0, tf); a single number T expands to (zero(T), T)
    init;               # initial guesses / parameter values + presence declaration (see below)

    # ---- problem structure ----
    g   = nothing,      # algebraic g(z,y,u,p,theta,t) = 0  → length-ny vector
    c   = nothing,      # path      c(z,y,u,p,theta,t) ≤ 0  → length-nc vector
    hE  = nothing,      # terminal  hE(zf,p,theta) = 0      → length-nhE vector (10.19h; NO t — it is the t_f point)

    # ---- controls ----
    u = nothing,        # fixed control profile u(t)::t->Vector. If GIVEN, u is a FIXED input (and drives
                        #   simulation-init). If [] / OMITTED, no profile ⇒ u is a decision variable (guess init.u).

    # ---- bounds (NamedTuple of (lower, upper) per class) ----
    bounds = (;),       # e.g. (z=(zL,zU), y=…, u=…, p=…); each ↦ lvar/uvar at every collocation point

    # ---- discretization ----
    nodes      = nothing,              # element boundaries; if given, NO forward-solve mesh (integrator not loaded)
    degree     = 4,                    # collocation points per element
    polynomial = EMD.Lagrange(),       # {Lagrange()}
    basis      = EMD.StateForm(),      # {StateForm(), DerivativeForm()}
    roots      = EMD.GaussRadau(),     # {GaussRadau(), GaussLegendre(), GaussLobatto()}
    adaptive   = false,                # false: t[i,j]/τ_k/h[i] as constants; true: as ExaModels parameters (AMR)
)
```

`init` — a NamedTuple that both seeds the initial guess and **declares which classes exist**:
- `init.p`, `init.theta`: length-`np`/`ntheta` vectors (time-invariant). `init.theta` is the `add_par` VALUE.
- `init.u`, `init.z`, `init.y`: a scalar / length-`n` vector (broadcast to all collocation points)
  or a callable `t -> vector` (time-varying guess, sampled at collocation points).
- **Empty/absent ⇒ that class is absent** (`init.p=[]` ⇒ `np=0`, etc.); likewise `u=[]`/omitted ⇒
  no fixed profile. Each entry ↦ ExaModels `start`. Dimensions come from `init`/`u`/function
  outputs, so no explicit `nz/ny/nu/np/ntheta` kwargs are needed.

Settled conventions:
- **`p` vs `theta`.** `p` = NLP unknowns (`add_var`, bounded/optimized). `theta` = ExaModels
  parameters (`add_par`, fixed per solve, mutable via `set_parameter!`).
- **Function arity.** `f(z,y,u,p,theta,t)`; `g/c(z,y,u,p,theta,t)` (`t` last, after `theta`).
  `z0(y,u,p,theta)` and `hE(zf,p,theta)` **take no `t`** — by definition they are the `t₀` and
  `t_f` points respectively.
- **`u`.** Providing the `u` profile makes `u` a fixed input; omitting it makes `u` a
  decision variable whose initial guess comes from `init.u`.
- **`init` = `start`.** Each `init.X` maps to ExaModels `add_var`'s `start` for block `X`
  (initial guess). `init.theta` is the `add_par` value. `init` also supplies the concrete
  values that drive simulation-based initialization.
- **Return form.** Full vector (required — the `length`-based dimension probe and the
  simulator both need it).

## 4. Initialization, meshing & dimension inference

The forward DAE solve is a **convenience layer**, not a requirement. Its three products —
dimensions, warm-start, mesh — degrade independently.

### 4.1 What determines each dimension

| Dim | Source | Needs simulation? |
|-----|--------|-------------------|
| `np` | `length(init.p)` | no |
| `ntheta` | `length(init.theta)` | no |
| `nu` | `length(u(t0))` if `u` given, else `length(init.u)` | no |
| `nz` | `length` of `f`/`z0` output | no (single evaluation) |
| `ny` | `length` of `g` output | no (single evaluation) |
| `nc`, `nhE` | `length` of `c` / `hE` output | no (single evaluation) |

Concrete `init.p/init.theta/u` carry the input-only dims (`np/ntheta/nu`) directly. `nz/ny/nc/nhE` are
output cardinalities, read from one evaluation of each function. The `z`/`y` inputs needed to
perform that evaluation are supplied as a **count-only probe** `_Probe`: a singleton scalar whose
`getindex` returns another `_Probe`, and whose arithmetic and `Base` math ops all return `_Probe`.
A type-generic user function fed `_Probe` for the unknown-length vectors returns a `Vector` whose
**length is the dimension**, with no numeric evaluation, so no size need be known in advance and no
domain error (`sqrt`, `log`, `/`) can fire. This is why user functions must be index-based and
type-generic (§2).

Extraction order, where each step makes the next argument concrete so probes are used only for
genuinely unknown lengths:
1. `np, ntheta, nu` from `init`/`u` (no eval).
2. `nz = length(z0(_Probe, u, p, theta))`.
3. `ny = length(g(_Probe, _Probe, u, p, theta, t0))`  (`0` if `g === nothing`).
4. `nc = length(c(...))`, `nhE = length(hE(_Probe, p, theta))`  (`0` when absent).

**All dimensions are obtainable without running the integrator** — the integrator is only for
meshing (§4.2).

### 4.2 When the forward solver (OrdinaryDiffEq extension) is loaded

The integrator runs in **exactly one case**: `nodes === nothing`. Providing `nodes` fixes the mesh
and the integrator is never loaded; omitting `nodes` means the mesh comes from a forward solve, so
a simulatable setup (a `u` profile and the extension) must be present.

| `nodes` | Behavior |
|---------|----------|
| **given** | **Integrator NOT loaded.** Mesh = user's (`N = length(nodes) - 1`). Dims from §4.1. Warm-start integrator-free (§4.3). |
| `nothing` | Integrator runs: adaptive steps → mesh nodes; trajectory → `:simulate` warm-start (§4.3). Requires `u` profile + extension. |

This satisfies the requirement: **providing `nodes` guarantees OrdinaryDiffEq is never
loaded.** The forward-solve mesh+warmstart lives in a package extension
(`…OrdinaryDiffEqExt`); the core `add_dae` path is dependency-free. There is no uniform-`N`
fallback: with no `nodes` the mesh is defined by the simulation, so no explicit element count is
ever required.

### 4.3 Initialization (inferred, no flag)

There is no warm-start kwarg. `add_dae` seeds the collocation starts from what the user already
provided, applied after the mesh is fixed. The baseline start of every block is its `init.X`
(scalar broadcast, per-component vector, or callable sampled at `t[i,j]`), defaulting to the
ExaModels `0` start where no `init.X` is given. The differential/algebraic state starts are then
refined by the best information available:

1. **Simulate** — if the integrator ran (`nodes === nothing`), sample its trajectory at the
   collocation times `t[i,k]` for `z`/`y` starts, junction times for `zb`, and `u(t[i,k])` for `u`.
2. **Consistent IC + hold** (integrator-free) — else if a `u` profile and `init.p/init.theta` are
   present, compute the consistent initial state at `t₀` (`zb₀ = z0(y₀, u(t₀), init.p, init.theta)`,
   with `y₀` from a small internal Newton solve of `g(zb₀, y₀, …) = 0` when `ny > 0`) and hold it
   flat: every `z_{i,k} = zb₀`, `y_{i,k} = y₀`, `u_{i,k} = u(t[i,k])`.
3. **`init` / zero** (integrator-free) — else the baseline `init.X` starts stand (or `0`).

Each tier degrades independently to the next when its inputs are absent, so the user controls
initialization purely by what they pass (`nodes`, `u`, `init`), not by a mode flag. Precedence
between an explicit `init.X` and an auto-seed for the same block is a later detail.

### 4.4 Fallback contract

Because `init` is required, `np`/`ntheta` (and `nu` via `init.u`) are always available, and
`nz/ny/nc/nhE` come from function outputs, so dimensions never need explicit kwargs. The mesh has
exactly two sources: the user's `nodes`, or a forward solve when `nodes === nothing`. Starts follow
the §4.3 ladder: simulate if the integrator ran, else consistent-IC hold, else `init`/zero.

## 5. Returned metadata (`dae`)

Sufficient to (a) build the objective, (b) add further constraints, (c) recover
trajectories from a solution.

```julia
struct DAEta
    # variable handles (ExaModels Variable / Parameter objects)
    z          # differential collocation states z_{i,j}
    zb         # boundary states zb_i (i=0..N);  zb[end] == zf
    y          # algebraic states y_{i,j}
    u          # controls u_{i,j}
    p          # decision parameters (Variable)
    theta          # ExaModels parameters (Parameter)

    # dimensions
    nz; ny; nu; np; ntheta
    N          # finite elements
    K          # collocation points per element (degree)

    # mesh & collocation layout
    nodes      # element boundaries τ̂₀..τ̂_N          (length N+1)
    h          # element lengths hᵢ                   (length N)
    tau        # collocation roots in (0,1]           (length K)
    t          # collocation times t_{i,j}            (N × K)
    A          # general collocation weights a_{jk} (Lagrange diff. matrix / RK Butcher A)
    b          # general final weights b_k (interpolation weights ℓ_k(1) / RK Butcher b)

    # convenience
    zf        # handle for terminal state zb_N
    method     # (basis, polynomial, roots) used
    con        # NamedTuple of constraint handles (collocation, continuity, initial, algebraic, path, terminal)
end
```

### Building the objective from `dae` (user side)

```julia
core, dae = EMD.add_dae(core, f, z0; hE=hE, degree=3, roots=EMD.GaussRadau(),
                         u=u_nominal, init=(p=p0, theta=theta0))

# Mayer term φ(zf):
@add_obj(core, (dae.zf[k] - z_ref[k])^2 for k in 1:dae.nz)

# Lagrange (integral) term via quadrature weights in dae:
@add_obj(core, dae.h[i] * w[j] * L(dae.z[k,i,j], dae.u[l,i,j]) for ...)
```

## 6. Internal module mapping and extraction pipeline

The strategy inputs `roots`, `polynomial`, `basis` are **singleton tag structs with no fields**.
They carry no data; all information is produced by helpers dispatching on their type. The extraction
is a chain on the reference element `[0,1]`, depending only on `degree = K`, so it is computed
**once** and reused across all `N` elements (and all conditions). Only `h[i]` and `t[i,j]` are
per-element.

```
roots, K    ──_get_roots(roots, K)───►  τ = [0, τ₁..τ_K], quadrature weights w
{0}∪τ       ──polynomial (Lagrange)──►  Lagrange basis ℓ_k over the K+1 node set
poly, basis ──_get_weights(basis, …)─►  (A, b)
```

**Node convention.** `τ₀ = 0` is universal across every family, so the node set is always
`{τ₀ = 0, τ₁..τ_K}` (`K+1` points) with identical structure. `τ₀` is the anchor tied to `zb_{i-1}`;
the ODE collocation residual is enforced at `τ₁..τ_K` (the `K` roots), giving C1's `nz·N·K` rows.
The families differ only in where `τ₁..τ_K` sit: Radau `τ_K = 1`; Legendre `τ₁..τ_K` interior to
`(0,1)`; Lobatto `τ_K = 1` as well (its left endpoint coincides with `τ₀`, not double-counted).

Helper contracts:
- `taus.jl` — `_get_roots(r::AbstractRoots, K)` dispatches on `GaussRadau/GaussLegendre/GaussLobatto`
  and returns `(τ = [0, τ₁..τ_K], w)`: the collocation points (roots of the shifted Gauss-Jacobi
  polynomial) with `0` prepended, and the `K` quadrature weights over `τ₁..τ_K` for user-side
  integral objective terms.
- `polynomial.jl` — `Lagrange` builds the interpolating-basis machinery over `{0, τ₁..τ_K}` and
  exposes `ℓ_k(1)` (endpoint value), `dℓ_k/dτ(τ_j)` (derivative at the collocation points), and
  `ω_k(τ)` (integrated basis) with `ω_k(τ_j)`, `ω_k(1)`.
- `basis.jl` — `_get_weights(basis, polynomial, τ)` dispatches on `basis` to select which
  combination becomes `(A, b)`; `_create_collocation` dispatches on `basis` for the C1 form:

  | `basis` | node set | `A` | `b` | C1 form |
  |---------|----------|-----|-----|---------|
  | `StateForm` | `{0, τ₁..τ_K}` (`K+1`) | diff. matrix `[dℓ_k/dτ(τ_j)]` | `[ℓ_k(1)]` | `Σ_k z_{i,k} ℓ̇_k(τ_j) = h_i f(…)` (Eq. 10.19b) |
  | `DerivativeForm` | `{τ₁..τ_K}` (`K`) | Butcher `[ω_k(τ_j)]` | `[ω_k(1)]` | `z_{i,j} = zb_{i-1} + h_i Σ_k a_{jk} f(…)` |

  `StateForm` anchors the polynomial at `0` (the boundary state); `DerivativeForm` represents the
  derivative over just the collocation points. Same math from both directions, which is why they
  share `roots`.
- `mesh.jl` — `_create_mesh(nodes, tspan, τ)` returns `(nodes, h, t)`. Two paths: user `nodes`
  (`N = length(nodes) - 1`, boundaries as given), or `nodes === nothing` routing to the
  OrdinaryDiffEq extension whose adaptive steps become the boundaries. `h[i] = nodes[i+1] - nodes[i]`,
  `t[i,j] = nodes[i] + h[i]·τ_j`.
- `initialize.jl` — dimension probing (§4.1), consistent-IC / constant warm-start, block
  allocation with bounds/starts, assembly of C1–C6 into `DAEta`.
- `daeta.jl` — the `DAEta` struct.
- **`ext/…OrdinaryDiffEqExt.jl`** — forward-solve mesh + `:simulate` warm-start; loaded only when
  OrdinaryDiffEq is available and `nodes === nothing`.

`_get_roots`, `_get_weights`, and the reference-element part of `_create_mesh` assemble one
resolved-spec object (dims, presence flags, reference `τ/w/A/b`, per-element `nodes/h/t`, resolved
init/bounds, `method = (basis, polynomial, roots)`) that flows into every `_create_*` generator and
largely populates `DAEta`. Under `adaptive = true` the reference/mesh arrays are realized as
ExaModels parameters (mutable for refinement); under `adaptive = false` they stay constants. This is
the immutable-weights vs AMR-mutable split noted in `daeta.jl`.

## 7. What "complete" requires (checklist)

- [ ] Dimensions inferred from `init`/`u` + function outputs (no explicit dim kwargs).
- [ ] `init`/`u` empty-or-absent ⇒ that class is absent (dimension 0).
- [ ] `theta` threaded into `f, z0, g, c, hE`; realized as an `@add_par` block from `init.theta`.
- [ ] `bounds`/`init` NamedTuples map to `lvar/uvar`/`start` on `z, y, u, p`; `theta` value
      from `init.theta`. `u` given ⇒ fixed input; omitted ⇒ decision variable.
- [ ] General initial-condition constraint `z0(y,u,p,theta)` (C3).
- [ ] Terminal constraint `hE(zf, p, theta)` (C6, Eq. 10.19h).
- [ ] `basis` dispatch producing C1 in Lagrange *or* Runge–Kutta form from shared roots.
- [ ] Forward-solve init (dims + simulate-trajectory starts + adaptive mesh) behind an
      OrdinaryDiffEq extension; **never loaded when `nodes` is provided**.
- [ ] Integrator-free initialization (consistent-IC hold, else `init`/zero), inferred not flagged (§4.3).
- [ ] `dae` exposes all variable handles, mesh/collocation layout, and `zf`.
- [ ] Solution recovery helper: `dae` → trajectories `z(t), y(t), u(t)`.

## 8. Open decisions

- **Auto-mesh step-size cap.** When the mesh is auto-generated from the integrator's steps,
  a cap / resample heuristic to a target `N` is needed. *User will specify this heuristic
  later* — placeholder until then.
- **Integrator backend** for the extension: OrdinaryDiffEq vs Sundials IDA for genuine
  index-1 DAEs.
- **Control parametrization (stepping).** By default a decision control is free at every
  collocation point (`u_{i,k}`). For controls that may only change on a coarser grid
  (control-vector parametrization / zero-order hold), a spec is needed — e.g. `control =
  PiecewiseConstant(u_nodes)` giving one DOF per control interval, held across the
  collocation points inside it (`PiecewiseLinear` later). Kept separate from `u` (fixed
  profile) and `init.u` (guess). Needs a decision on syntax + default hold.

Resolved:
- **Constraint naming:** `g` = algebraic equality (`=0`), `c` = path inequality (`≤0`),
  `hE` = terminal equality. README aligned to this.
- `u`'s role — providing the `u` profile fixes the control; omitting makes it a decision
  variable (guess via `init.u`).
- `z0` and `hE` take **no `t`** — they are by definition the `t₀` and `t_f` points.
- **Mesh sources.** Exactly two: user `nodes`, or a forward solve when `nodes === nothing`. No
  uniform-`N` fallback, so no explicit element count is ever required (§4.2).
- **Collocation node convention.** `τ₀ = 0` is universal across Radau/Legendre/Lobatto; the node
  set is always `{0, τ₁..τ_K}` and the ODE is enforced at `τ₁..τ_K`. Lobatto's left endpoint
  coincides with `τ₀` (not double-counted) (§6).
- **Dimension probing.** `nz/ny/nc/nhE` from one evaluation per function using the count-only
  `_Probe`; no integrator, no domain errors (§4.1).
