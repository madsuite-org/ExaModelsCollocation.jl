# ExaModels indexes generically over AbstractVariable (.size, .offset, .length), so a subtype
# carrying the collocation layout indexes like a Variable and needs no side table.

"""
    CollocationVariable

A variable block over the collocation mesh, from [`add_var_collocation`](@ref). Indexes like
an `ExaModels.Variable`: `z[v, c, i, k]`.

# Fields
- `dims`   : declared dimensions, the shape at one collocation point
- `krange` : collocation index range, `0:K` with the interval-left boundary node, else `1:K`
"""
struct CollocationVariable{S, O, T, D} <: ExaModels.AbstractVariable
    size::S
    length::O
    offset::O
    name::Symbol
    tag::T
    dims::D
    krange::UnitRange{Int}
end

CollocationVariable(v::ExaModels.Variable, dims, krange::UnitRange{Int}) =
    CollocationVariable(v.size, v.length, v.offset, v.name, v.tag, dims, krange)

# Leading dimensions the caller declared; the mesh indices (i, k) are the rest.
_nleading(z::CollocationVariable) = length(z.dims)

function Base.show(io::IO, z::CollocationVariable)
    print(
        io,
        """
        CollocationVariable

          $(z.name)[$(join(z.dims, ", ")), i, k] ∈ R^{$(join(ExaModels.size(z.size), " × "))}, k = $(z.krange)
        """,
    )
end

# ExaModels' bound accessors dispatch on ::Variable, so they need methods here. Indexing and
# `solution` are generic or duck-typed, and need none.
_var_range(z::CollocationVariable) = (z.offset + 1):(z.offset + z.length)

ExaModels.get_start(m::ExaModels.ExaModel, z::CollocationVariable) =
    view(m.meta.x0, _var_range(z))
ExaModels.get_lvar(m::ExaModels.ExaModel, z::CollocationVariable) =
    view(m.meta.lvar, _var_range(z))
ExaModels.get_uvar(m::ExaModels.ExaModel, z::CollocationVariable) =
    view(m.meta.uvar, _var_range(z))

ExaModels.set_start!(m::ExaModels.ExaModel, z::CollocationVariable, values) =
    _copy_into!(view(m.meta.x0, _var_range(z)), values, z, :set_start!)
ExaModels.set_lvar!(m::ExaModels.ExaModel, z::CollocationVariable, values) =
    _copy_into!(view(m.meta.lvar, _var_range(z)), values, z, :set_lvar!)
ExaModels.set_uvar!(m::ExaModels.ExaModel, z::CollocationVariable, values) =
    _copy_into!(view(m.meta.uvar, _var_range(z)), values, z, :set_uvar!)

function _copy_into!(dest, values, z::CollocationVariable, who::Symbol)
    length(values) == z.length || throw(DimensionMismatch(
        "$who: expected $(z.length) elements, got $(length(values))"
    ))
    return copyto!(dest, values)
end
