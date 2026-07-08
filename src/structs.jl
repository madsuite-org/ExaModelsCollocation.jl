# Metadata returned by `add_dae`. Holds the variable/parameter handles created on the
# ExaCore, the discretization dimensions, the mesh/collocation layout, and the appended
# constraint handles — everything the user needs to build an objective, add further
# constraints, or recover trajectories from a solution. See `docs/api_design.md` §5.

"""
    DAEta

Discretization metadata returned by [`add_dae`](@ref) (bound as `dae`). Carries the
ExaModels variable/parameter handles, the problem dimensions, the mesh and collocation
layout, and the handles of the appended structural constraints. `add_dae` transcribes the
DAE onto the `ExaCore`; the objective and any extra constraints are written separately by
the user against this object.

# Fields

Variable / parameter handles (`nothing` when the corresponding class is absent):
- `z`  — differential collocation states `z_{i,j}`
- `zb` — element-boundary states `zb_i` (`i = 0..N`); `zb[end] == zf`
- `y`  — algebraic collocation states `y_{i,j}`
- `u`  — controls `u_{i,j}`
- `p`  — decision parameters (ExaModels `Variable`)
- `theta` — mutable parameters (ExaModels `Parameter`)
- `zf` — terminal boundary state `zb_N`

Dimensions: `nz`, `ny`, `nu`, `np`, `ntheta`; `N` finite elements; `K` collocation points
per element (`= degree`).

Mesh & collocation layout:
- `nodes` — element boundaries `τ̂₀..τ̂_N` (length `N+1`)
- `h`  — element lengths `hᵢ` (length `N`)
- `tau` — collocation roots in `(0,1]` (length `K`)
- `t`  — collocation times `t_{i,j}` (`N × K`)
- `A`  — general collocation weights `a_{jk}` (Lagrange differentiation matrix, or Runge–Kutta Butcher `A`)
- `b`  — general final weights `b_k` (interpolation weights `ℓ_k(1)`, or Runge–Kutta Butcher `b`)

Convenience:
- `method` — NamedTuple `(basis, polynomial, roots)` of the strategy objects used
- `con` — NamedTuple of appended constraint handles (collocation, continuity, initial,
  algebraic, path, terminal)
"""
struct DAEta{Z,ZB,Y,U,P,Theta,ZF,T,MA,MTH,CON}
    # variable / parameter handles
    z::Z
    zb::ZB
    y::Y
    u::U
    p::P
    theta::Theta
    zf::ZF

    # dimensions
    nz::Int
    ny::Int
    nu::Int
    np::Int
    ntheta::Int
    N::Int
    K::Int

    # mesh & collocation layout
    nodes::Vector{T}
    h::Vector{T}
    tau::Vector{T}
    t::Matrix{T}
    A::MA
    b::Vector{T}

    # convenience
    method::MTH
    con::CON
end

function Base.show(io::IO, dae::DAEta)
    print(
        io,
        """
        DAEta

          states     nz = $(dae.nz),  ny = $(dae.ny),  controls nu = $(dae.nu)
          parameters np = $(dae.np),  ntheta = $(dae.ntheta)
          mesh       N  = $(dae.N) elements × K = $(dae.K) collocation points
          method     $(dae.method.basis), $(dae.method.polynomial), $(dae.method.roots)
        """,
    )
end
