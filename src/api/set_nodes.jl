# Move the mesh a core or model was built on, keeping the interval count
"""
    set_nodes!(model, nodes)

Relocates the placement of `nodes` of a `CollocationExaModel`, given `adaptive = true`.
"""
function set_nodes!(c::Union{CollocationExaCore, CollocationExaModel}, nodes::AbstractVector)
    mesh = _mesh(c)
    _hpar(mesh) === nothing && throw(ArgumentError(
        "set_nodes!: this mesh is not adaptive; build it with `adaptive = true`"
    ))
    length(nodes) == length(mesh.nodes) || throw(DimensionMismatch(
        "set_nodes!: expected $(length(mesh.nodes)) boundaries, got $(length(nodes)); " *
        "changing the number of intervals needs a rebuild"
    ))
    issorted(nodes; lt = <=) ||
        throw(ArgumentError("set_nodes!: nodes must be strictly increasing along t"))

    taus = _weights(c).taus
    h, t = _hval(mesh), _tval(mesh)
    copyto!(mesh.nodes, nodes)
    h .= diff(mesh.nodes)
    for i in axes(t, 1), j in axes(t, 2)
        t[i, j] = mesh.nodes[i] + h[i] * taus[j]
    end

    ExaModels.set_value!(c, _hpar(mesh), h)
    ExaModels.set_value!(c, _tpar(mesh), t)
    return nothing
end
