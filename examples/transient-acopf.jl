# Transient power systems 

ENV["GKSwstype"] = "100"

using ExaModels
using ExaModelsCollocation
using MadNLP
using Plots

# ----- Problem data -----



# ----- Create ExaModel -----

function transient_acopf()
    # Create CollocationExaCore
    core = CollocationExaCore(nodes, K;
        
    )
end

# ----- Solve -----