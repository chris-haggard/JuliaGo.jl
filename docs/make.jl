using Documenter
using JuliaGo

makedocs(
    sitename = "JuliaGo",
    format = Documenter.HTML(prettyurls = false),
    modules = [JuliaGo],
    pages = ["Home" => "index.md", "Documentation" => "Documentation.md"],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
