---
layout: post
title:  "Guest Post: GSoC Hyperparameter Tuning Visualization"
author: mason
date:   2016-08-21 12:10:44 -0400
categories: machine learning, R
---
Today I am wrapping up my GSoC project on hyperparameter tuning visualization for the R package, mlr. I wanted to give
an overview of the types of problems the project solves and how to use it within the mlr API. A cross post is available
[here](here) on the official mlr blog.


Background: Hyperparameter tuning visualization provides a way of visualizing what happens during the tuning process that identifies the best hyperparameters for given data.

Typical usecases:

- How does varying the value of a hyperparameter change the performance of the machine learning algorithm?
- On a related note: where's an ideal range to search for optimal hyperparameters?
- How did the optimization algorithm (prematurely) converge?
- What's the relative importance of each hyperparameter?

Typical users:

- researchers
- engineers
- teachers

In action:

For the examples, we will use the Pima Indians dataset. This means we will be performing classification, and we will use SVM to demonstrate.
Of course, the examples would also work for regression and any custom learner the user wishes to call.

Let's use SVM and tune the `C` parameter:

{% highlight R %}

library(mlr)
# create the C parameter in continuous space: 2^-5 : 2^5
ps = makeParamSet(
  makeNumericParam("C", lower = -5, upper = 5, trafo = function(x) 2^x)
)
# random search in the space with 100 iterations
ctrl = makeTuneControlRandom(maxit = 100L)
# 2-fold CV
rdesc = makeResampleDesc("CV", iters = 2L)
# run the tuning process
res = tuneParams("classif.ksvm", task = pid.task, control = ctrl,
  measures = list(acc, mmce), resampling = rdesc, par.set = ps, show.info = FALSE)
# generate the hyperparameter tuning data, accounting for the transformation 2^x
generateHyperParsEffectData(res, trafo = T)
#> HyperParsEffectData:
#> Hyperparameters: C
#> Measures: acc.test.mean,mmce.test.mean
#> Optimizer: TuneControlRandom
#> Nested CV Used: FALSE
#> Snapshot of $data:
#>           C acc.test.mean mmce.test.mean iteration exec.time
#> 1  0.4883018     0.7656250      0.2343750         1     0.127
#> 2  1.6792996     0.7343750      0.2656250         2     0.049
#> 3 25.5804975     0.7044271      0.2955729         3     0.058
#> 4  0.1226795     0.7122396      0.2877604         4     0.052
#> 5  1.2203486     0.7382812      0.2617188         5     0.053
#> 6 31.4949746     0.7057292      0.2942708         6     0.063

{% endhighlight %}

Now we can easily plot `C` against the accuracy:

{% highlight R %}

data = generateHyperParsEffectData(res, trafo = T)
plotHyperParsEffect(data, x="C", y="acc.test.mean", plot.type="line")

{% endhighlight %}

![c_plot](/assets/plotC.png)

Maybe we then further search in the region between C=0 and C=3 to see if we can find better performance.

Let's try a more complicated, real-world scenario. We want to:

- Tune both `C` and `sigma` simultaneously
- Construct a heatmap, interpolating points in the interval automatically
- Mark the points that were in-experiment to distinguish from interpolated points
- Use nested cross validation to get an unbiased measure of performance


{% highlight R %}

ps = makeParamSet(
  makeNumericParam("C", lower = -5, upper = 5, trafo = function(x) 2^x),
  makeNumericParam("sigma", lower = -5, upper = 5, trafo = function(x) 2^x))
ctrl = makeTuneControlRandom(maxit = 100)
rdesc = makeResampleDesc("Holdout")
learn = makeLearner("classif.ksvm", par.vals = list(kernel = "rbfdot"))
lrn = makeTuneWrapper(learn, control = ctrl, measures = list(acc, mmce),
  resampling = rdesc, par.set = ps, show.info = FALSE)
res = resample(lrn, task = pid.task, resampling = cv2, extract = getTuneResult, show.info = FALSE)
data = generateHyperParsEffectData(res)
plt = plotHyperParsEffect(data, x = "C", y = "sigma", z = "acc.test.mean",
  plot.type = "heatmap", interpolate = "regr.earth", show.experiments = TRUE,
  nested.agg = mean)
min_plt = min(plt$data$acc.test.mean, na.rm = TRUE)
max_plt = max(plt$data$acc.test.mean, na.rm = TRUE)
mean_plt = mean(c(min_plt, max_plt))
plt + scale_fill_gradient2(breaks = seq(min_plt, max_plt, length.out = 4),
  low = "red", mid = "white", high = "blue", midpoint = mean_plt)

{% endhighlight %}

![nested_plot](/assets/nested.png)

This was just a taste of mlr's hyperparameter tuning visualization capabilities. For the full tutorial, check out the [mlr tutorial](http://mlr-org.github.io/mlr-tutorial/devel/html/hyperpar_tuning_effects/index.html).

Some features coming soon:

- "Prettier" plot defaults
- Support for more than 2 hyperparameters
- Direct support for hyperparameter "importance"

Thanks again to the generous sponsorship from GSoC, and many thanks to my mentors Bernd Bischl and Lars Kotthoff!