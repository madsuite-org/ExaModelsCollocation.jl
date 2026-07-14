# Construct collocation mesh
"""
    CollocationMesh{T}
"""
struct CollocationMesh{TTA,TT,TH}
    taus::TTA
    t::AbstractMatrix{TT}
    h::AbstractVector{TH}
end