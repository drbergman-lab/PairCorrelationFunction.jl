using PairCorrelationFunction
using Documenter

DocMeta.setdocmeta!(PairCorrelationFunction, :DocTestSetup, :(using PairCorrelationFunction); recursive=true)

makedocs(;
    modules=[PairCorrelationFunction],
    authors="Daniel Bergman <danielrbergman@gmail.com> and contributors",
    sitename="PairCorrelationFunction.jl",
    format=Documenter.HTML(;
        canonical="https://Daniel Bergman.github.io/PairCorrelationFunction.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Daniel Bergman/PairCorrelationFunction.jl",
    devbranch="main",
)
