---
layout: post
title:  "Visualizing blackbox models"
date:   2016-07-07 14:10:44 -0400
categories: thoughts, mlr
---
I'm sure most of you have encountered the supposed dichotomy between "black box" models and more "intuitive" / "straight forward" models such as logistic regression. A frequent argument I encounter against so-called blackbox models is that clients / upper management need to understand how the model works internally. This is an important argument that I often see both dismissed too quickly by some but also used as a crutch by others. Full disclosure, I have made this mistake myself when advising clients and students! Guilty as charged.

In my experience, the problem often arises when clients want to proactively make changes to their business based on your model. For example, if we are predicting the likelihood of a click on an ad, the client might want to know if it's a better use of internal resources to fix the colors, call to action, typeface, images, and so on. Just the prediction of click / no click isn't enough in this case!

If we use logistic regression, we can look at the coefficients. If we use SVM, how do we advise the client in this situation? If we use random forest importance score, we don't have enough information because we still don't know the sign of the most importance predictors.

Partial dependency plots to the rescue! Partial dependency plots provide a means to visualize "black box" models so that we can understand the influence of each of our predictors on the prediction. Especially helpful when your problem requires understanding of predictor influence rather than just a prediction.

Check out the [Paper](http://arxiv.org/abs/1309.6392) with implementation. Highly recommended reading!

Implementations are available in the [original R package](https://github.com/kapelner/ICEbox), the R machine learning package, [mlr](http://mlr-org.github.io/mlr-tutorial/devel/html/partial_dependence/index.html), as well as [scikit-learn](http://scikit-learn.org/stable/auto_examples/ensemble/plot_partial_dependence.html). 

![partial dependence plot](/assets/partial_dependence.svg)