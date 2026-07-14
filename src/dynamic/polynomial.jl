# Interpolating polynomial used to represent the differential state
# across each interval, evaluated at the collocation points.
# default = ExaModelsDAE.Lagrange()
#   ExaModelsDAE.Lagrange() : Lagrange interpolation polynomials
# NOTE: Largrange interpolation polynomials are preffered because polynomial
# coefficients have the same variable bounds as the profiles themselves
# (L. T. Biegler, Nonlinear Programming, Chapter 10.2.1)
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

"""
    delljk(taus) -> Matrix

Derivative interpolation weights `A[j,k] = ℓⱼ'(τₖ)` for the Lagrange basis over the
nodes `taus = [0, τ₁..τ_K]`, evaluated at the collocation points `τ₁..τ_K`.
Rows `j = 0..K` (basis), columns `k = 1..K` (collocation); size `(K+1) × K`.
"""
function delljk(taus)
    n = length(taus)                      # n = K+1
    w = _baryweights(taus)

    # Barycentric differentiation matrix D[k,i] = ℓᵢ'(taus[k])
    D = zeros(eltype(taus), n, n)
    for k in 1:n, i in 1:n
        i == k && continue
        D[k, i] = (w[i] / w[k]) / (taus[k] - taus[i])
    end
    for k in 1:n
        D[k, k] = -sum(D[k, i] for i in 1:n if i != k)
    end

    # Keep the K collocation rows (drop tau0 = 0) and orient as A[j,k]
    return permutedims(D[2:n, :])
end

"""
    ell1j(taus) -> Vector

Continuity weights `b[j] = ℓⱼ(1)` for the Lagrange basis over `taus = [0, τ₁..τ_K]`,
each basis polynomial at the right endpoint `τ = 1`. Length `K+1`.
"""
function ell1j(taus)
    w = _baryweights(taus)
    return _lagrange(taus, w, one(eltype(taus)))
end

"""
    Omegajk(taus) -> Matrix

Collocation weights `A[j,k] = Ωⱼ(τₖ)` with `Ωⱼ(τ) = ∫₀^τ ℓ̄ⱼ`, the integrated Lagrange
basis over the collocation points `roots = τ₁..τ_K`. Rows `j = 1..K` (basis),
columns `k = 1..K` (collocation); size `K × K`.
"""
function Omegajk(taus)
    roots = taus[2:end]
    K = length(roots)
    w = _baryweights(roots)

    # Column k holds the basis integrated from 0 to the kth collocation point
    A = zeros(eltype(roots), K, K)
    for k in 1:K
        A[:, k] = _integrate_basis(roots, w, roots[k])
    end
    return A
end

"""
    Omega1j(taus) -> Vector

Continuity weights `b[j] = Ωⱼ(1) = ∫₀^1 ℓ̄ⱼ`, the integrated Lagrange basis over the
collocation points `roots = τ₁..τ_K` at the right endpoint. Length `K`.
"""
function Omega1j(taus)
    roots = taus[2:end]
    w = _baryweights(roots)
    return _integrate_basis(roots, w, one(eltype(roots)))
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

# Integrated basis [∫₀^b ℓⱼ] via Gauss-Legendre quadrature, exact for degree K-1
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
