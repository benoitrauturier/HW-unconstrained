
using FactCheck
using GLM
using maxlike
using DataFrames

context("basics") do
	d=maxlike.makeData()
	facts("Test Data Construction") do
		@fact typeof(d) --> Dict{ASCIIString,Any}
		@fact typeof(d["data"])--> Array{Float64,2}
		@fact maximum(d["data"][:,1]) --> 1.0
		@fact minimum(d["data"][:,1]) --> 1.0
	end

	facts("Test Return value of likelihood") do
		@fact maxlike.loglik([100.0,100.0,100.0],d) <1e-8 -->true
		#we check that the loglike is decreasing around the true value
		@fact maxlike.loglik(d["coeff"],d) - maxlike.loglik([randn()+d["coeff"][1];randn()+d["coeff"][2];randn()+d["coeff"][3]],d) >=0 -->true

	end

	facts("Test return value of gradient") do
		# gradient should not return anything,
		# but modify a vector in place.
		storage=zeros(3)
		#check nothing is returned by the function
		@fact maxlike.grad!([randn()+d["coeff"][1];randn()+d["coeff"][2];randn()+d["coeff"][3]],storage,d) --> nothing
		#chek vector storage is somewhat modified
		@fact storage != zeros(3) -->true
	end
end

context("test maximization results") do
	d=maxlike.makeData()
	facts("maximize returns approximate result") do
		#we test that the true value is in the confidence intervall
		result=maxlike.maximize_like()
		for i in 1:length(d["coeff"])
			@fact abs(result.minimum[i]-d["coeff"][i])<0.1-->true
		end

	end

	facts("maximize_grad returns accurate result") do
		result=maxlike.maximize_like_grad()
		for i in 1:length(d["coeff"])
			@fact abs(result.minimum[i]-d["coeff"][i])<0.1-->true
		end
	end

	facts("maximize_grad_hess returns accurate result") do
		result=maxlike.maximize_like_grad_hess()
		for i in 1:length(d["coeff"])
			@fact abs(result.minimum[i]-d["coeff"][i])<0.1-->true
		end
	end

end

context("test against GLM") do
	# create data and use the GLM package
	# probit example is on the github page
	#here we ensure that we have the same data as the default values in maxlike
	srand(20160324)
	d=maxlike.makeData()
	estim=maxlike.maximize_like_grad_se()
	d_df=DataFrame(y=d["y"],d1=d["data"][:,1],d2=d["data"][:,2],d3=d["data"][:,3])
	Probit = glm(y~d2+d3,d_df,Binomial(),ProbitLink())
	facts("estimates vs GLM") do
		diff_coeff=abs(coef(Probit)-estim[:Estimates])
		for i in 1:length(diff_coeff)
			@fact diff_coeff[i] < 1e-5 -->true
		end


	end

	facts("standard errors vs GLM") do
		diff_se=abs(stderr(Probit)-estim[:StandardErrors])
		for i in 1:length(diff_se)
			@fact diff_se[i]<1e-4 -->true
		end
	end

end
