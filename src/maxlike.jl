module maxlike

	using Distributions, Optim, PyPlot, DataFrames, Debug

	"""
    `input(prompt::AbstractString="")`

    Read a string from STDIN. The trailing newline is stripped.

    The prompt string, if given, is printed to standard output without a
    trailing newline before reading input.
    """
    function input(prompt::AbstractString="")
        print(prompt)
        return chomp(readline())
    end

    export runAll, makeData



	# methods/functions
	# -----------------

	# data creator
	# should/could return a dict with beta,numobs,X,y,norm
	# true coeff vector, number of obs, data matrix X (Nxk), response vector y (binary), and a type of parametric distribution; i.e. the standard normal in our case.
	function makeData(n=10000, latent = true)
		beta = [ 1.0; 1.5; -0.5 ]
		#Here we assume that the x follow a normal distribution of mean [1,2,3] and variance [0.0,1,1]
		mu = [1.0,2.0]
		sig = diagm([1.0;1.0])
		#We set up the generator
		law_x=MvNormal(mu,sig)
		#and generate the sample of explanatory variables
		sample_x=ones(n,3)
		sample_x[:,2:3]=transpose(rand(law_x,n))
		norm = Normal()
		if latent
			law_eps=Normal()
			eps=rand(law_eps,n)
			y_star=sample_x*beta + eps
			sample_y=zeros(n)
			for i in 1:n
				if y_star[i] > 0.0
					sample_y[i]=1.0
				end
			end
		else
			#We compute the probability for the observations of y

			proba_vect = cdf(norm,sample_x*beta)
			#And finally compute the final observations of y
			#probably not the best solution
			sample_y=zeros(n)
			for i in 1:n
				 sample_y[i]=rand(Bernoulli(proba_vect[i]),1)[1]
			end
		end
		return Dict("y"=> sample_y,"data"=>sample_x, "n"=>n, "coeff" => beta,"distribution" => norm)
	end
	srand(20160324)
	d=makeData()
	# log likelihood function at x
	# function loglik(betas::Vector,d::Dict)
	function loglik(betas::Vector,d=d)
		#One will only compute the good side.
		result=0
		#Here we have to go with a for loop to avoid the cases were the log will not be defined even though it is multiplied by 0.
		for i in 1:length(d["y"])
			if d["y"][i] == 1.0
				result+=log(cdf(d["distribution"],d["data"][i,:]*betas))
			else
				result+=log(1-cdf(d["distribution"],d["data"][i,:]*betas))
			end
		end
		return sum(result)
	end
	#Here we define an additional function Likeli that returns the inverse value of the likelihood for some beta
	function Likeli(betas::Vector)
		return -loglik(betas)
	end


	# gradient of the likelihood at x
	function grad!(betas::Vector,storage::Vector,d=d)
		for i in 1:length(storage)
			result=0
			for j in 1:length(d["y"])
				if d["y"][j] == 1.0
					result+=d["data"][j,i]*pdf(d["distribution"],d["data"][j,:]*betas)/cdf(d["distribution"],d["data"][j,:]*betas)
				else
					result-=d["data"][j,i]*pdf(d["distribution"],d["data"][j,:]*betas)/(1-cdf(d["distribution"],d["data"][j,:]*betas))
				end
			end
			storage[i]=-sum(result)
		end
	end


	# hessian of the likelihood at x
	function hessian!(betas::Vector,storage::Matrix,d=d)
		#Here it is possible to make this function greater by using three for loops instead of one
		storage[:,:]=0.0
		for i in 1:length(d["y"])
			#For clarity we define the value of the density and the cdf outside the calculation
			density=pdf(d["distribution"],d["data"][i,:]*betas)[1]
			cumulative=cdf(d["distribution"],d["data"][i,:]*betas)[1]
			#As always we compute the good side of the equation, to avoid NaN values
			if d["y"][i]==1.0
				storage[1,1]+=density*d["data"][i,1]*d["data"][i,1]*((density.+(d["data"][i,:]*betas)[1].*(cumulative))/(cumulative^2))
				storage[2,1]+=density*d["data"][i,2]*d["data"][i,1]*((density.+(d["data"][i,:]*betas)[1].*(cumulative))/(cumulative^2))
				storage[3,1]+=density*d["data"][i,3]*d["data"][i,1]*((density.+(d["data"][i,:]*betas)[1].*(cumulative))/(cumulative^2))
				storage[2,2]+=density*d["data"][i,2]*d["data"][i,2]*((density.+(d["data"][i,:]*betas)[1].*(cumulative))/(cumulative^2))
				storage[3,2]+=density*d["data"][i,3]*d["data"][i,2]*((density.+(d["data"][i,:]*betas)[1].*(cumulative))/(cumulative^2))
				storage[3,3]+=density*d["data"][i,3]*d["data"][i,3]*((density.+(d["data"][i,:]*betas)[1].*(cumulative))/(cumulative^2))
			else
				storage[1,1]+=density*d["data"][i,1]*d["data"][i,1]*((density-(d["data"][i,:]*betas)[1].*(1-cumulative))/((1-cumulative)^2))
				storage[2,1]+=density*d["data"][i,2]*d["data"][i,1]*((density-(d["data"][i,:]*betas)[1].*(1-cumulative))/((1-cumulative)^2))
				storage[3,1]+=density*d["data"][i,3]*d["data"][i,1]*((density-(d["data"][i,:]*betas)[1].*(1-cumulative))/((1-cumulative)^2))
				storage[2,2]+=density*d["data"][i,2]*d["data"][i,2]*((density-(d["data"][i,:]*betas)[1].*(1-cumulative))/((1-cumulative)^2))
				storage[3,2]+=density*d["data"][i,3]*d["data"][i,2]*((density-(d["data"][i,:]*betas)[1].*(1-cumulative))/((1-cumulative)^2))
				storage[3,3]+=density*d["data"][i,3]*d["data"][i,3]*((density-(d["data"][i,:]*betas)[1].*(1-cumulative))/((1-cumulative)^2))
			end
		end
		#By symetry one fills out the rest of the matrix
		storage[1,2]=storage[2,1]
		storage[2,3]=storage[3,2]
		storage[1,3]=storage[3,1]
	end


	"""
	inverse of observed information matrix
	"""
	function inv_observedInfo(betas::Vector,d)
		storage=zeros(3,3)
		hessian!(betas,storage,d)
		return storage^(-1)
	end

	"""
	standard errors
	"""
	function se(betas::Vector,d=d)
		return sqrt(diag(inv_observedInfo(betas,d)))
		#return [0,0,0]
	end

	# function that maximizes the log likelihood without the gradient
	# with a call to `optimize` and returns the result
	function maximize_like(x0=[0.8,1.0,-0.1],meth=:nelder_mead)
	#maximizing is minimizing the inverse
		return optimize(Likeli,x0, method= :nelder_mead)
	end



	# function that maximizes the log likelihood with the gradient
	# with a call to `optimize` and returns the result
	function maximize_like_grad(x0=[0.8,1.0,-0.1],meth=:bfgs)
		return optimize(Likeli,grad!,x0, method= :bfgs)
	end

	# function that maximizes the log likelihood with the gradient
	# and hessian with a call to `optimize` and returns the result
	function maximize_like_grad_hess(x0=[0.8,1.0,-0.1],meth=:newton)
		return optimize(Likeli,grad!,hessian!,x0, method= :newton)
	end

	# function that maximizes the log likelihood with the gradient
	# and computes the standard errors for the estimates
	# should return a dataframe with 3 rows
	# first column should be parameter names
	# second column "Estimates"
	# third column "StandardErrors"
	function maximize_like_grad_se(x0=[0.8,1.0,-0.1],meth=:bfgs)
		result=optimize(Likeli,grad!,x0, method= :bfgs)
		return DataFrame(Names=["beta 1","beta 2", "beta 3"],Estimates=result.minimum,StandardErrors=se(result.minimum))
	end


	# visual diagnostics
	# ------------------

	# function that plots the likelihood
	# we are looking for a figure with 3 subplots, where each subplot
	# varies one of the parameters, holding the others fixed at the true value
	# we want to see whether there is a global minimum of the likelihood at the true value.
	function plotLike(data=d)
#Lets call the parameters x,y,z
		x=linspace(0,2,200)
		y=linspace(0.5,2.5,200)
		z=linspace(-1.5,0.5,200)
		l_x=zeros(200)
		l_y=zeros(200)
		l_z=zeros(200)
		for i in 1:200
			l_x[i]=loglik([x[i];1.5;-0.5],data)
			l_y[i]=loglik([1.0;y[i];-0.5],data)
			l_z[i]=loglik([1.0;1.5;z[i]],data)
		end
		figure("Log likelihood with respect to the 3 parameters")
		subplot(311)
		plot(collect(x),l_x,color="green")
		title("WRT x")
		subplot(312)
		plot(collect(y),l_y,color="green")
		title("WRT y")
		subplot(313)
		plot(collect(z),l_z,color="green")
		title("WRT z")

	end




	function runAll()
		plotLike()
		m1 = maximize_like()
		m2 = maximize_like_grad()
		m3 = maximize_like_grad_hess()
		m4 = maximize_like_grad_se()
		println("results are:")
		println("maximize_like: $(m1.minimum)")
		println("maximize_like_grad: $(m2.minimum)")
		println("maximize_like_grad_hess: $(m3.minimum)")
		println("maximize_like_grad_se: $m4")
		println("")
		println("running tests:")
		include("test/runtests.jl")
		println("")
		ok = input("enter y to close this session.")
		if ok == "y"
			quit()
		end
	end


end
