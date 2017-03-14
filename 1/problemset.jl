using Distributions, Gadfly, DataFrames, Iterators

function nfolds(x::AbstractArray, n::Int)
    l = length(x)
    [ x[round(Int64, (i-1)*l/n) + 1 : min(l, round(Int64, i*l/n))] for i in 1:n ]
end

mom(X, k) = median([mean(i) for i in nfolds(X, k)])

function generate_means(D::Sampleable, N::Int, M::Int, K::Int)
    [(mean(r), mom(r,K)) for r in [ rand(D, N) for i in 1:M ]]
end

function compare_means(means::Array{Tuple{Float64,Float64}})
    geom = Geom.density(bandwidth=5)
    plot(
        layer(x = [i for (i,_) in means], geom,
              Theme(default_color=colorant"rgba(50,220,220,0.5)")),
        layer(x = [j for (_,j) in means], geom,
              Theme(default_color=colorant"rgba(220,130,50,1)")),
        Coord.cartesian(xmin=-25, xmax=25)
    )
end

function compare_tails(means::Array{Tuple{Float64,Float64}}, tail = 1)
    geom = Geom.histogram(density=true)
    f = filter(t -> all(x->(x>tail), t), means)
    plot(
        layer(x = [i for (i,_) in f], geom,
              Theme(default_color=colorant"rgba(50,220,220,0.5)")),
        layer(x = [j for (_,j) in f], geom,
              Theme(default_color=colorant"rgba(220,130,50,1)")),
        Coord.cartesian(xmin=tail, xmax=3)
    )
end

worst_case(means::Array{Tuple{Float64,Float64}}) =
    maximum([map(abs, a) for a in means])

function plot_worst_cases(D::Sampleable, M::Int, K::Int, sizes::AbstractArray{Int})
    cases = [worst_case(generate_means(D, N, M, K)) for N in sizes]
    geom = Geom.smooth()
    plot(
        layer(y = [i for (i,_) in cases], x = sizes, geom, Theme(default_color=colorant"blue")),
        layer(y = [j for (_,j) in cases], x = sizes, geom, Theme(default_color=colorant"orange"))
    )
end

##########################
# Basis Projections
##########################

scale_and_center(p) = (p - mean(p))/var(p)
project(n::Int, m = eye(n), d = 2) = rand(Normal(), d, n) * m

function compare_plots(n::Int)
    s = scale_and_center(project(n))
    r = rand(Normal(), 2, n)
    DataFrame([
        s' fill("projected", n) fill(n, n);
        r' fill("random", n) fill(n, n)
    ])
end

function compare_all_plots(nums = [20,500])
    c = vcat([compare_plots(n) for n in nums])
    plot(c, x = "x1", y = "x2", xgroup="x3", ygroup="x4",
         Geom.subplot_grid(Geom.point),
         Guide.xlabel = "Projected vs. Random",
         Guide.xlabel = "Sample Size")
end

##########################
# Boxes Projections
##########################

function boxes(n::Int)
    c = collect(product(repeat([[-1,1]], outer=n)...))
    reinterpret(Int, c, (n, 2^n))
end

project_boxes(n::Int) = project(n, boxes(n))'

function make_df_from_calls(fn::Function, arr)
    m = reduce((a,b) -> [a; fn(b) fill(b, 2^b)], Array{Any}(0,3), arr)
    DataFrame(m)
end

function plot_projections(arr::Vector{Int})
    df = make_df_from_calls(project_boxes, arr)
    plot(df, x = "x1", y="x2", xgroup="x3", Geom.subplot_grid(Geom.point))
end
