 using Gadfly
using DataFrames
using Distances

####################
# 5
d = DataFrame(min = [0, 1],
              max = [2, 4],
              y = [1/2, 1/3],
              class = ["Class 0", "Class 1"])

plot(d, xmin="min", xmax="max", y = "y", label = "class",
     Geom.bar,
     Theme(default_color=colorant"rgba(220,130,50,0.5)"),
     Scale.x_continuous(minvalue = -1, maxvalue = 5),
     Scale.y_continuous(minvalue = 0, maxvalue = 1),
     Guide.xlabel(nothing),
     Guide.ylabel(nothing),
     Geom.label)


###################
# 8
function draw_sets(n, d)
    X = rand(n,d)
    y = [i < j ? 1 : 0 for (i,j) in zip(rand(n), X[:,1])]
    X, y
end

get_neighbors(x_distances, k) = sortperm(x_distances)[2:k+1]
get_score(neighbors, y) = round(mean(y))
compute_risk(y_hat, y) = sum([x == true ? 0 : 1 for x in (y_hat .== y)])/length(y)

function classify(X, y, k)
    dist = pairwise(Euclidean(), X')
    l = size(dist)[1]
    labels = [get_score(getindex(y, get_neighbors(dist[i,:], k))) for i in 1:l]
    compute_risk(labels, y)
end

function runs(m, n, d, k)
    [classify(X, y, k) for (X,y) in [(X,y) = draw_sets(n,d) for i in 1:m]]
end

function plotter(K, m, n = 500, d = 3)
    losses = reduce(vcat, [[runs(m, n, d, k) fill(k, m)] for k in K])
    plot(DataFrame(losses), x="x1", color="x2",
         Geom.density(bandwidth = .03),
         Scale.x_continuous(minvalue=0.25, maxvalue=0.5),
         Scale.color_discrete)
end

plotter([1,3,5,7,9], 50, 1000, 3)
