using Documenter
using TransformerXAI

DocMeta.setdocmeta!(TransformerXAI, :DocTestSetup, :(using TransformerXAI); recursive=true)

makedocs(;
    modules=[TransformerXAI],
    sitename="TransformerXAI.jl",
    authors="Charansurya Udaysingh Jhurree",
    checkdocs=:exports, #tells Documenter to only verify that exported functions are documented in the manual, leaving forward_changed! and talktollm_changed alone since they're intentionally internal.
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nirvanjhurreepriv.github.io/TransformerXAI.jl",
        edit_link="main",
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "gettingStarted.md",
        "Functionality" => "functionality.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/nirvanjhurreepriv/TransformerXAI.jl",
    devbranch="main",
)