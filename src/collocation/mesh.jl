# Step 3 of the discretization: place the reference interval along physical time.
# Given the N+1 interval boundaries and the K collocation points taus, the mesh holds
# t[i,j] = nodes[i] + h[i]*tau_j, the time each collocation point falls on.
#
# h and t are the only t-space data the residuals read -- A, b and taus are tau-space and do
# not move with the mesh -- so they are the only two an adaptive mesh has to make mutable.
# Under `adaptive`, each is additionally allocated as an ExaModels parameter block, and the
# residuals index the handle instead of the array. The numeric arrays are kept either way:
# bounds, start values and any initialization by integration need plain numbers.
"""
    CollocationMesh{TB,TH,TT,PH,PT}

Physical mesh geometry of the discretized horizon.

# Fields
- `nodes` : interval boundaries `nodes[i]`, i=1,...,N+1
- `h`     : interval lengths `h[i] = nodes[i+1] - nodes[i]`, i=1,...,N
- `t`     : collocation times `t[i,j] = nodes[i] + h[i]*tau_j`, size N x K
- `hpar`  : `h` as an `ExaModels.Parameter` under an adaptive mesh, `nothing` otherwise
- `tpar`  : `t` as an `ExaModels.Parameter` under an adaptive mesh, `nothing` otherwise
"""
struct CollocationMesh{TB,TH,TT,PH,PT}
    nodes::TB
    h::TH
    t::TT
    hpar::PH
    tpar::PT
end

# Parse resolved boundaries and reference points into a CollocationMesh
function _get_mesh(nodes::AbstractVector, taus::AbstractVector)
    # Interval lengths h[i]
    h = diff(nodes)
    N = length(h)

    # Collocation points tau_1..tau_K
    K = length(taus)

    # Collocation times t[i,j], following Biegler's indexing
    t = Matrix{eltype(h)}(undef, N, K)
    for i in 1:N, j in 1:K
        t[i, j] = nodes[i] + h[i] * taus[j]
    end

    return CollocationMesh(nodes, h, t, nothing, nothing)
end

# Attach the parameter handles an adaptive mesh indexes its residuals off, keeping the
# numeric arrays exactly as they were.
_with_parameters(mesh::CollocationMesh, hpar, tpar) =
    CollocationMesh(mesh.nodes, mesh.h, mesh.t, hpar, tpar)
