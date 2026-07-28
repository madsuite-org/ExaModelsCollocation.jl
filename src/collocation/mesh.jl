# Step 3 of the discretization: place the reference interval along physical time.
# Given the N+1 interval boundaries and the K collocation points taus, the mesh holds
# t[i,j] = nodes[i] + h[i]*tau_j, the time each collocation point falls on.
"""
    CollocationMesh{TB,TH,TT}

Physical mesh geometry of the discretized horizon.

# Fields
- `nodes` : interval boundaries `nodes[i]`, i=1,...,N+1
- `h`     : interval lengths `h[i] = nodes[i+1] - nodes[i]`, i=1,...,N
- `t`     : collocation times `t[i,j] = nodes[i] + h[i]*tau_j`, size N x K
"""
struct CollocationMesh{TB,TH,TT}
    nodes::TB
    h::TH
    t::TT
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

    return CollocationMesh(nodes, h, t)
end
