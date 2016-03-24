


[![Build Status](https://travis-ci.org/benoitrauturier/HW-unconstrained.svg?branch=master)](https://travis-ci.org/benoitrauturier/HW-unconstrained.svg?branch=master)


## Questions branch

[![Join the chat at https://gitter.im/ScPo-CompEcon/HW-unconstrained](https://badges.gitter.im/ScPo-CompEcon/HW-unconstrained.svg)](https://gitter.im/ScPo-CompEcon/HW-unconstrained?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

* In this homework you will implement the maximum likelihood estimator for the probit model.
* Clearly there are many packaged solutions for this problem out there, which are very robust and efficient. Nevertheless it is a good exercise to try maximum likelihood out, precisely because we can compare the results to packaged solutions.
* any non-standard maximum likelihood problem you might encounter will be very similar to the steps involved here, so there is something to learn.

### testing

1. You will complete some unit tests in this homework.
1. you should then enable your forked repo for unit testing on `Travis`.
1. You see the (red) testing badge on top of this file? This should be green when you submit the homework.
1. To enable travis, go to *settings* in your fork.
	1. on the left, click on *Webhooks and services*
	1. Under *Services*, click on *add service*
	1. Type **Travis** and select **Travis CI**.
	1. Enter your password, don't change any settings and click on create service.
	1. Finally, go to [travis and click login with github](https://travis-ci.org).
	1. on the left, click on your repo that you want to test
	1. click on the status build image, and copy the Markdown code from the popup.
	1. Paste that code right at the top of *this* file, instead of the code you see there now. 
	1. Now, each time you push to github, or you submit a pull request, the tests will run.
1. Notice the config file `.travis.yml` in this repo. Don't change it.


### questions

1. Please follow the questions in the accompanying IJulia notebook.


## License

Please observe that this repo is part of the [Sciences Po CompEcon Organisation](https://github.com/ScPo-CompEcon) and therefore subject to the license detailed at the bottom of [The Syllabus repo](https://github.com/ScPo-CompEcon/Syllabus).
