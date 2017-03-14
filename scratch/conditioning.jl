using Distributions
using Gadfly

function get_conditions(K = 10, N = 10, P = 10, dist = Normal())
    [cond(rand(dist, N, P)) for i in 1:K]
end

get_max(range = 10:2:50, dist = Normal(), K=1000, P = 10) = [quantile(get_conditions(1000, i, P, dist), .95) for i in range]

##############################
using Distances

p = pdf(TDist(2), rand(TDist(2), 1000))
q = pdf(TDist(1), rand(TDist(1), 1000))


kl_divergence(p,q)

kl_divergence(q,p)

plot(layer(x -> pdf(TDist(2), x), -5, 5),
     layer(x -> pdf(TDist(1), x), -5, 5)
     )
