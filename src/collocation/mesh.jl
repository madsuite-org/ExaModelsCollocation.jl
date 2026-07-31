# Collocation mesh data structure
"""
    CollocationMesh{TB,TH,TT,PH,PT}

Contains collocation mesh details.

# Fields
- `nodes` : interval boundaries `nodes[i]`, i=1,...,N+1, for N intervals
- `h`     : interval lengths `h[i] = nodes[i+1] - nodes[i]`, for i=1,...,N
- `t`     : collocation times `t[i,j] = nodes[i] + h[i]*tau_j`, for i=1,...,N, k=0,...,K
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

    # Collocation points tau[j], j=1,...,K
    K = length(taus)

    # Collocation times t[i,j], i=1,...,N, j=0,...,K
    t = Matrix{eltype(h)}(undef, N, K)
    for i in 1:N, j in 1:K
        t[i, j] = nodes[i] + h[i] * taus[j]
    end

    return CollocationMesh(nodes, h, t, nothing, nothing)
end

# If adaptive = true, append parameter handles for h[i], t[i,j]
_with_parameters(mesh::CollocationMesh, hpar, tpar) =
    CollocationMesh(mesh.nodes, mesh.h, mesh.t, hpar, tpar)
