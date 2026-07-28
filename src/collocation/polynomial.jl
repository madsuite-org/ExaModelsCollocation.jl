# The interpolating polynomial family, and the basis evaluations the weights are built from.
# Lagrange is preferred because its coefficients are the profile values themselves, so they
# inherit the same variable bounds (L. T. Biegler, Nonlinear Programming, Chapter 10.2.1).
#
# Naming follows Biegler: ell/dell are the basis and its derivative, Omega its integral.
"""
    AbstractPolynomial

Abstract type for the interpolating polynomial family used within each interval.
A concrete subtype supplies the basis functions and the weights for the
colocation and continuity constraints.
"""
abstract type AbstractPolynomial end

"""
    Lagrange()

Lagrange interpolation polynomials.
"""
struct Lagrange <: AbstractPolynomial end

# Every weight below is `basis function j, evaluated somehow at point k`, laid out as
# `[j, k]` so the constraint helpers can read a column per collocation point. `nodes` are
# the interpolation nodes the basis is built over; `at` are the points to evaluate at.

"""
    delljk(nodes, at) -> Matrix

Derivative weights `A[j,k] = ℓⱼ'(at[k])` for the Lagrange basis over `nodes`.
Size `length(nodes) × length(at)`.

Uses the barycentric differentiation matrix, which is exact only at the nodes themselves,
so every entry of `at` must be one of `nodes`.
"""
function delljk(nodes, at)
    D = _diffmatrix(nodes)
    rows = map(x -> _nodeindex(nodes, x), at)
    return permutedims(D[rows, :])
end

"""
    ell1j(nodes) -> Vector

Continuity weights `b[j] = ℓⱼ(1)`, each basis polynomial over `nodes` at the interval's right
endpoint `τ = 1`. Length `length(nodes)`.
"""
ell1j(nodes) = _lagrange(nodes, _baryweights(nodes), one(eltype(nodes)))

"""
    Omegajk(nodes, at) -> Matrix

Integrated weights `A[j,k] = Ωⱼ(at[k])` with `Ωⱼ(τ) = ∫₀^τ ℓⱼ`, the Lagrange basis over
`nodes`. Size `length(nodes) × length(at)`. `at` is unrestricted here.
"""
function Omegajk(nodes, at)
    w = _baryweights(nodes)
    A = zeros(eltype(nodes), length(nodes), length(at))
    for k in eachindex(at)
        A[:, k] = _integrate_basis(nodes, w, at[k])
    end
    return A
end

"""
    Omega1j(nodes) -> Vector

Integrated weights `b[j] = Ωⱼ(1) = ∫₀^1 ℓⱼ` over `nodes`. Length `length(nodes)`.
"""
Omega1j(nodes) = _integrate_basis(nodes, _baryweights(nodes), one(eltype(nodes)))

# Barycentric differentiation matrix D[k,i] = ℓᵢ'(nodes[k])
function _diffmatrix(nodes)
    n = length(nodes)
    w = _baryweights(nodes)

    D = zeros(eltype(nodes), n, n)
    for k in 1:n, i in 1:n
        i == k && continue
        D[k, i] = (w[i] / w[k]) / (nodes[k] - nodes[i])
    end
    for k in 1:n
        # init covers the single-node basis, where the off-diagonal sum is empty
        D[k, k] = -sum((D[k, i] for i in 1:n if i != k); init = zero(eltype(D)))
    end
    return D
end

function _nodeindex(nodes, x)
    i = findfirst(==(x), nodes)
    i === nothing &&
        throw(ArgumentError("delljk evaluates only at its own nodes; $x is not one of them"))
    return i
end

# Barycentric weights wᵢ = 1 / ∏_{j≠i}(xᵢ - xⱼ) for distinct nodes x
function _baryweights(x)
    n = length(x)
    w = ones(eltype(x), n)
    for i in 1:n, j in 1:n
        i == j && continue
        w[i] /= (x[i] - x[j])
    end
    return w
end

# Lagrange basis [ℓⱼ(t)] at t via the barycentric form, exact at the nodes
function _lagrange(x, w, t)
    hit = findfirst(xi -> xi == t, x)
    hit === nothing || return [j == hit ? one(t) : zero(t) for j in eachindex(x)]

    terms = w ./ (t .- x)
    return terms ./ sum(terms)
end

# Integrated basis [∫₀^b ℓⱼ] via Gauss-Legendre quadrature, exact for the degree n-1 basis
function _integrate_basis(x, w, b)
    n = length(x)
    gx, gw = FastGaussQuadrature.gausslegendre(n)

    acc = zeros(eltype(x), n)
    for q in 1:n
        s = b / 2 * (gx[q] + 1)               # map [-1,1] to [0, b]
        acc .+= (b / 2 * gw[q]) .* _lagrange(x, w, s)
    end
    return acc
end
