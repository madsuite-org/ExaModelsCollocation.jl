# r-refinement: solve, estimate the error per interval, move a fixed number of nodes to
# equidistribute it, repeat. set_nodes! is what makes the mesh move without a rebuild.
#
#   dz/dt = a / (1 + a^2 (t - t0)^2),   z(0) = atan(-a t0)   =>   z(t) = atan(a (t - t0))
#
# The rise spans ~2/a = 0.02, so N uniform intervals put the peak inside one of them and the
# quadrature barely sees it. Nothing in the loop knows zexact: it estimates its own error.
#
#   julia --project=examples examples/r_refinement.jl

ENV["GKSwstype"] = "100"                    # GR writes the png without a display

using ExaModels
using ExaModelsCollocation
using MadNLP
using Plots

# ----- Define problem -----

const A, T0, TEND = 100.0, 0.5, 1.0

# ODE right-hand side function
rhs(z, t) = A / (1 + A^2 * (t - T0)^2)

# Exact solution for comparison
zexact(t) = atan(A * (t - T0))

# ----- Create ExaModel -----

function example_model(nodes; K = 3)
    # Create CollocationExaCore with adaptive mesh (t is an ExaModels parameter)
    core = CollocationExaCore(nodes, K; adaptive = true)
    t = core.mesh.tpar

    # Create CollocationVariable
    @add_var_collocation(core, z)

    # Create collocation constraints    
    itr = [(i, k) for i in 1:core.N, k in 1:core.K]
    @add_con_collocation(core, coll, z, rhs(z[i, k], t[i, k]) for (i, k) in itr)

    # Create continuity constraints
    @add_con_continuity(core, cont, z)

    # Create objective function
    @add_con(core, ic, z[1, 0] - zexact(0.0) for _ in 1:1)

    return ExaModel(core), z
end

solve(model, z) = solution(madnlp(model; print_level = MadNLP.ERROR, tol = 1e-8), z)

# ----- Error estimate -----

function estimate_error_phr(model, zsol, f)
    nodes, h = model.nodes, model.mesh.h

    # The degree K Lagrange polynomial our collocation equations use
    x = [zero(eltype(model.weights.taus)); model.weights.taus]
    interp(zi, tau) = sum(
        prod((tau - x[m]) / (x[j] - x[m]) for m in eachindex(x) if m != j) * zi[j]
        for j in eachindex(x)
    )

    # The K+1 roots (non-collocation points to compare)
    fine = Collocation([0.0, 1.0], model.K + 1;
        roots = model.roots, 
        basis = DerivativeForm(),
        polynomial = model.polynomial
    )
    tau, Omega = fine.mode.weights.taus, fine.mode.weights.A

    # Calculate max error estimate
    return [
        # Compare:
        #   Interpolated states of the degree K Lagrange polynomial at the K+1 roots
        #   Integrated states as if we had used K+1 Lagrange polynomial
        let zi = view(zsol, i, :),
            # Interpolated states of the degree K Lagrange polynomial at the K+1 roots
            zf = [interp(zi, s) for s in tau],

            # the RHS function evaluated at the K+1 roots
            ff = [f(zf[m], nodes[i] + h[i] * tau[m]) for m in eachindex(tau)]

            maximum(
                # Integrated states as if we had used K+1 Lagrange polynomial
                abs(zi[1] + h[i] * sum(Omega[j, m] * ff[j] for j in eachindex(tau)) - zf[m])
                for m in eachindex(tau)
            ) / (1 + maximum(abs, zsol))
        end
        for i in eachindex(h)
    ]
end
# NOTE: this error estimation is cheap because we already obtained the 
# polynomial coefficients (the discretized states) from solving the NLP
# so this is just an O(N) calculation, N = num. intervals


# ----- Mesh update -----

function find_new_nodes(model, err; floor_frac = 0.1, passes = 2)
    nodes, h, N = model.nodes, model.mesh.h, model.N

    # de Boor density (?)
    rho = err .^ (1 / (model.K + 1)) ./ h
    rho = rho .+ floor_frac * (sum(rho) / N)
    for _ in 1:passes
        rho = [(rho[max(i - 1, 1)] + 2rho[i] + rho[min(i + 1, N)]) / 4 for i in 1:N]
    end

    # invert cumulative mass to convert density to new node placement (?)
    W = cumsum([0.0; rho .* h])                # W[i]: monitor mass left of nodes[i]
    new = collect(float.(nodes))               # the horizon ends stay put
    for m in 2:N
        target = W[end] * (m - 1) / N
        i = clamp(searchsortedlast(W, target), 1, N)
        new[m] = nodes[i] + (target - W[i]) / rho[i]
    end

    return new
end

# ----- r-refinement algorithm -----

# Keep track of how much the mesh moved for convergence criteria purposes
movement(new, old, h) = maximum(
    abs(new[j] - old[j]) / min(h[max(j - 1, 1)], h[min(j, length(h))])
    for j in eachindex(new)
)


# Re-interpolated states at interpolation points based on new node locations for better initial guess
function reinterpolate(model, zsol, old, new)
    # evaluate the degree K Lagrange polynomial through the solved coefficients
    x = [zero(eltype(model.weights.taus)); model.weights.taus]
    interp(zi, tau) = sum(
        prod((tau - x[m]) / (x[j] - x[m]) for m in eachindex(x) if m != j) * zi[j]
        for j in eachindex(x)
    )
    hold, hnew = diff(old), diff(new)
    start = similar(zsol, size(zsol))
    for i in eachindex(hnew), (k, tau) in enumerate(x)
        t = new[i] + hnew[i] * tau
        ii = clamp(searchsortedlast(old, t), 1, length(hold))
        s = (t - old[ii]) / hold[ii]
        start[i, k] = interp(view(zsol, ii, :), s)
    end
    return start
end

function solve_adaptively(
        model, z, f;
        tol = 1e-8,
        movetol = 1e-2,
        maxiters = 20,
        verbose = true,
    )
    history = []

    # Solve model
    zsol = solve(model, z)

    # while (max error < tol OR movement < movement tol)
    for it in 0:maxiters
        nodes = copy(model.nodes)

        # calculate error using Patterson-Hager-Rao
        err = estimate_error_phr(model, zsol, f)

        push!(history, (; nodes, zsol, err)) # tracking error histroy for plot

        # find new node placements based on error + de Boor equidistributino
        new = find_new_nodes(model, err)
        moved = movement(new, nodes, model.mesh.h)
        verbose && println("iteration $it: max error = $(maximum(err)), movement = $moved")

        # convergence criteria
        (maximum(err) < tol || moved < movetol) && break

        # if criteria not satisfied, set new nodes and re-solve
        set_nodes!(model, new) # <--- ExaModelsCollocation.jl feature with adaptive = true
        set_start!(model, z, reinterpolate(model, zsol, nodes, new))
        zsol = solve(model, z)
    end

    return history
end


# ----- Solve -----

const N = 30

# Create ExaModel
model, z = example_model(range(0.0, TEND; length = N + 1))

# Solve with adaptive mesh refinement
history = solve_adaptively(model, z, rhs)


# ----- Plot -----

# Display and plot
function plot_mesh_history(history, exact; floor = 1.0e-9)
    atnodes(s) = [s.zsol[:, 1]; s.zsol[end, end]]

    t = reduce(vcat, s.nodes for s in history)
    iteration = reduce(vcat, fill(i - 1, length(s.nodes)) for (i, s) in enumerate(history))

    err = log10.(
        max.(reduce(vcat, [abs.(atnodes(s) .- exact.(s.nodes)) for s in history]), floor)
    )
    err = (err .- minimum(err)) ./ (maximum(err) - minimum(err))

    return scatter(
        t, iteration;
        marker_z = err, c = :jet, clims = (0, 1),
        markersize = 5, markerstrokewidth = 0, legend = false, colorbar = true,
        colorbar_title = "\nerror vs exact solution",
        colorbar_ticks = 0:0.2:1,
        xlabel = "t", ylabel = "refinement iteration",
        xlims = (0.0, TEND), ylims = (-0.5, length(history) - 0.5),
        yticks = 0:(length(history) - 1),
        title = "r-refinement: node placement per iteration",
        right_margin = 5Plots.mm,
    )
end
nodes = last(history).nodes
println("h ranges over [$(minimum(diff(nodes))), $(maximum(diff(nodes)))]")
for (it, s) in enumerate(history)
    znode = [s.zsol[:, 1]; s.zsol[end, end]]
    println("iteration $(it - 1): |z(T) - exact| = $(abs(znode[end] - zexact(TEND)))")
end
png = joinpath(@__DIR__, "r_refinement.png")
savefig(plot_mesh_history(history, zexact), png)
println("wrote $png")
