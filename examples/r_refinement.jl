# toy example of r-refinement:
#   - solve NLP
#       - while (err > tol) or (mesh movement > movement_tol)
#           - estimate error
#           - relocate mesh
#           - check convergence criteria
#           - resolve NLP
#

ENV["GKSwstype"] = "100"

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

function example_model(nodes, K = 3)
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

    return ExaModel(core)
end

solve(model) = solution(madnlp(model; print_level = MadNLP.ERROR, tol = 1e-8), model.z)

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

    # Calculate error estimate
    return [
        # Compare:
        #   Interpolated states of the degree K Lagrange polynomial at the K+1 roots
        #   Integrated states to K+1 degree using K+1 interpolated state points
        let zi = view(zsol, i, :),
            # znew = Interpolated states of the degree K Lagrange polynomial at the K+1 roots
            znew = [interp(zi, s) for s in tau],

            # the RHS function evaluated at the K+1 roots
            fnew = [f(znew[m], nodes[i] + h[i]*tau[m]) for m in eachindex(tau)]

            maximum(
                # (Integrated states to K+1 degree using K+1 interpolated state points) - znew
                abs(zi[1] + h[i]*sum(Omega[j,m]*fnew[j] for j in eachindex(tau)) - znew[m])
                for m in eachindex(tau)
            ) / (1 + maximum(abs, zi))
        end
        for i in eachindex(h)
    ]
end
# NOTE: this error estimation is cheap because we already obtained the 
# polynomial coefficients (the discretized states) from solving the NLP


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
    W = cumsum([0.0; rho .* h])
    new = collect(float.(nodes))
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
        model, f;
        tol = 1e-6,
        movetol = 1e-2,
        maxiters = 20,
        verbose = true,
    )
    history = []

    # Solve model
    zsol = solve(model)

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
        # set_start!(model, model.z, reinterpolate(model, zsol, nodes, new))
        zsol = solve(model)
    end

    return history
end


# ----- Solve -----

const N = 50
const K = 3

# Create ExaModel
model = example_model(range(0.0, TEND; length = N + 1), K)

# Solve with adaptive mesh refinement
history = solve_adaptively(model, rhs)


# ----- Plot -----

default(
    titlefontsize = 16, guidefontsize = 13, tickfontsize = 10,
    legendfontsize = 10, colorbar_titlefontsize = 12, plot_titlefontsize = 18,
)

atnodes(s) = [s.zsol[:, 1]; s.zsol[end, end]]

function error_color(history, exact; floor = 1.0e-9)
    err = [log10.(max.(abs.(atnodes(s) .- exact.(s.nodes)), floor)) for s in history]
    clims = (log10(floor), ceil(maximum(maximum, err)))

    return err, clims
end

function colorbar_strip(clims; rows = 256)
    decades = first(clims):last(clims)
    ramp = collect(range(first(clims), last(clims); length = rows))

    return heatmap(
        [0.0], ramp, reshape(ramp, :, 1);
        c = :jet, clims = clims, legend = false, colorbar = false, grid = false,
        xticks = false, ymirror = true, tickfontsize = 11,
        yticks = (decades, ["1e$(Int(d))" for d in decades]),
        title = "|z_exact - z|", titlefontsize = 12,
        right_margin = 4Plots.mm,
    )
end

# Display and plot
function plot_mesh_history(history, exact)
    t = reduce(vcat, s.nodes for s in history)
    iteration = reduce(vcat, fill(i - 1, length(s.nodes)) for (i, s) in enumerate(history))
    err, clims = error_color(history, exact)

    return scatter(
        t, iteration;
        marker_z = reduce(vcat, err), c = :jet, clims = clims,
        markersize = 7, markerstrokewidth = 0, legend = false, colorbar = false,
        xlabel = "t", ylabel = "r-iter", guidefontsize = 16, yguidefontsize = 18,
        tickfontsize = 12,
        xlims = (0.0, TEND), ylims = (-0.5, length(history) - 0.5),
        yticks = 0:(length(history) - 1),
        title = "r-refinement: node placement per iteration", titlefontsize = 18,
        right_margin = 5Plots.mm,
    )
end

function plot_solution_3d(history, exact)
    zall = reduce(vcat, atnodes(s) for s in history)
    zfloor = minimum(zall) - 0.18 * (maximum(zall) - minimum(zall))
    errs, clims = error_color(history, exact)

    p = plot3d(;
        xlabel = "t", ylabel = "r-iter", zlabel = "z(t)",
        xlims = (0.0, TEND), ylims = (-0.5, length(history) - 0.5),
        zlims = (zfloor, maximum(zall)),
        yticks = 0:(length(history) - 1),
        legend = false, camera = (17.5, 15), colorbar = false,
        projection_type = :perspective,
        title = "r-refinement: state trajectory per iteration",
        left_margin = -14Plots.mm, right_margin = -6Plots.mm,
        top_margin = -14Plots.mm, bottom_margin = -14Plots.mm,
    )

    for (it, (s, err)) in enumerate(zip(history, errs))
        iteration = fill(it - 1, length(s.nodes))
        scatter3d!(
            p, s.nodes, iteration, fill(zfloor, length(s.nodes));
            marker_z = err, c = :jet, clims = clims,
            markersize = 3.5, markerstrokewidth = 0,
        )
        plot3d!(
            p, s.nodes, iteration, atnodes(s);
            line_z = err, c = :jet, clims = clims, linewidth = 2.5,
        )
    end
    return p
end

nodes = last(history).nodes
println("h ranges over [$(minimum(diff(nodes))), $(maximum(diff(nodes)))]")
for (it, s) in enumerate(history)
    znode = [s.zsol[:, 1]; s.zsol[end, end]]
    println("iteration $(it - 1): |z(T) - exact| = $(abs(znode[end] - zexact(TEND)))")
end
png = joinpath(@__DIR__, "r_refinement.png")
savefig(
    plot(
        plot_mesh_history(history, zexact),
        plot_solution_3d(history, zexact),
        colorbar_strip(last(error_color(history, zexact)));
        layout = grid(1, 3; widths = [0.48, 0.48, 0.04]), size = (1500, 720),
        left_margin = 8Plots.mm, bottom_margin = 5Plots.mm,
    ),
    png,
)
println("wrote $png")
