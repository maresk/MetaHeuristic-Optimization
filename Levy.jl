## Implementing MetaHeuristic Optimization in Julia
## Cuckoo Search Algorithm: Global Convergence, MultiModal optimization 


## This script implements CS using the Levy distribution 
## Use a simple objective such as a circle in n dimensions to test

using Distributions
using Bokeh


# Get new Cuckoos

function get_cuckoos(nests,best,Lb,Ub)

	n = size(nests,2)

	# Levy Exponent and Coefficient
	
	beta = 3/2
	sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);

	for j in 1:n

		# Following Mantegna to generate Levy Flights 
		u = rand(GD)*sigma
		v = rand(GD)
		step = u/abs(v).^(1/beta)

		s = nests[:,j]
		stepsize = 0.01*step.*(s-best)
		s += stepsize.*rand(GD,size(best))

		# Update new move
		nests[:,j] = applybounds!(s,Lb,Ub)
				
	end

end


# Find the current best nest

function get_best_nest(nests,newnests,fitness)

	for j in 1:size(nests,2)
		fnew = objective(newnests[:,j])

		if fnew <= fitness[j]
			fitness[j] = fnew
			nests[:,j] = newnests[:,j]
		end

	end

	(fmin, K) = findmin(fitness)
	best = nests[:,K]

	return (fmin,K,best) 

end


function empty_nests(nest,Lb,Ub,pa)

	n = size(nest,2)
	K = rand(size(nest)) .> pa
	#println(K)

	# New solution by biased/selective Random Walk 

	stepsize = rand()*(nest[:,randperm(n)] - nest[:,randperm(n)])
	new_nest = nest + stepsize.*K

	for j in 1:size(new_nest,2)
		s = new_nest[:,j]
		new_nest[:,j] = applybounds!(s,Lb,Ub)

	end


	return new_nest 
end


# Keeping solutions within bounds
function applybounds!(s,Lb,Ub)
	s_temp = s
	s_temp[s_temp .< Lb] = Lb[s_temp .< Lb]
	s_temp[s_temp .> Ub] = Ub[s_temp .> Ub]
	
	# Update new move
	return s_temp

end


# General objective function to minimize
function objective(u::Array{Float64,1}) # Simple objective function
	# d-dimensional sphere with minimum at (1,1,1...)
	z = sum((u-1).^2)
end


GD = Normal()
N_nests = 50			# Number of nests
pa = 0.25			# Discovery rate of solutions
N_IterTotal = 1000	# Number of iterationss
Tol = 0.01			# Tolerance
nd = 15				# Dimensionality of the input space

# Bounds
Lb = -5*ones(nd)
Ub = 5*ones(nd)


# Randomized initital Nests
nests = zeros(nd,N_nests)
for i in 1:N_nests
	nests[:,i] = Lb + (Ub-Lb).*rand(GD,size(Lb)) 
end


# Get the current best
fitness = 10^10*ones(N_nests,1)

(fmin,K,bestnest) = get_best_nest(nests,nests,fitness)

N_iter = 0

while fmin > Tol

for iter in 1:N_IterTotal
	# Generating new solutions keeping the current best
	new_nest = get_cuckoos(nests,bestnest,Lb,Ub)

	(fnew,K,best) = get_best_nest(nests,nests,fitness)

	N_iter += N_nests

	# Discover New Nests

	new_nest = empty_nests(nests,Lb,Ub,pa)

	if fnew < fmin
		fmin = fnew
		bestnest = best
	end

end

println("Current minimum fitness : ", fmin)

end

println("Number of Iterations : ", N_iter)
println("Minimum fitness : ", fmin)
println("Best Nest")
println(bestnest)


#get_cuckoos(nests,nests[:,1],Lb,Ub)



