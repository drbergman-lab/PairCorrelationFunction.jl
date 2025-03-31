using PairCorrelationFunction
using Documenter

DocMeta.setdocmeta!(PairCorrelationFunction, :DocTestSetup, :(using PairCorrelationFunction); recursive=true)

makedocs(;
    modules=[PairCorrelationFunction],
    authors="Daniel Bergman <danielrbergman@gmail.com> and contributors",
    sitename="PairCorrelationFunction.jl",
    format=Documenter.HTML(;
        canonical="https://drbergman-lab.github.io/PairCorrelationFunction.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/drbergman-lab/PairCorrelationFunction.jl",
    devbranch="development",
    push_preview=true,
)
