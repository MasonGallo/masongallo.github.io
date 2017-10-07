---
layout: post
title: Machine Learning In Production - Pipelines
date:   2017-10-07 12:45:15 -0400
categories: machine learning, python
---

One of the big problems that I hope we as a machine learning community continue
to improve soon is the creation and maintenance of end to end machine learning systems
in production. While I enjoy reading many of the wonderful posts and analysis focusing
on prototyping and new machine learning techniques, I've wanted to write about some of
my battle stories from building machine learning systems in a production environment.
I have been especially happy to see many larger companies beginning to open up about
their in-house systems, such as [Uber's Michelangelo](https://eng.uber.com/michelangelo/).

My goal is to make this a series. This first post will focus on machine learning
pipelines using scikit-learn and pandas.

# Background

Let's say you want to deploy a new machine learning task, and your dataset contains
both categorical and numerical features. Side note: I've seen way too many tutorials
and examples just assume you only have numerical features, which isn't very reflective
of the real world.

If your tech stack involves python, you probably use both pandas and scikit-learn.
Unfortunately, scikit-learn doesn't include great support for categorical features out
of the box. The pre-processing features it does provide have a number of drawbacks
for our usecases `OneHotEncoder` requires that features are already encoded as integers
and `DictVectorizer` requires a dict.

Fortunately, scikit-learn provides us excellent building blocks to construct our own
process for handling categorical features.

# Categorical Pipeline

If you're not familiar with scikit-learn's excellent concept of pipelining, check out
the [documentation](http://scikit-learn.org/stable/modules/pipeline.html#pipeline).
Pipelines allow us to chain multiple estimators into one "central" estimator that we
can then `fit` and `transform`.

Below we create a categorical transformer that will one hot encode all categorical
features. This will also of course account for issues like new / unseen categories
in the test/prediction set. We require a pandas dataframe and use the `Categorical`
dtype.


{% highlight python %}
from pandas import Categorical, get_dummies
from sklearn.base import TransformerMixin, BaseEstimator


class CategoricalWarrior(BaseEstimator, TransformerMixin):
    """One hot encoder for all categorical features"""
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        cats = {}
        for column in self.attribute_names:
            cats[column] = X[column].unique().tolist()
        self.categoricals = cats
        return self

    def transform(self, X, y=None):
        df = X.copy()
        for column in self.attribute_names:
            df[column] = Categorical(df[column], categories=self.categoricals[column])
        new_df = get_dummies(df, drop_first=True)
        # in case we need them later
        self.columns = new_df.columns
        return new_df

{% endhighlight %}

Now we can use the familiar `fit` and `transform` paradigms by including this as a step
in our pipeline! How to use it:

{% highlight python %}
import pandas as pd


data = {"A": [2, 2, 4, 4], "B": [1, 1, 1, 1], "C": ["yes", "no", "no", "yes"]}
example_df = pd.DataFrame(data)
categorical_cols = ["C"]
cat = CategoricalWarrior(categorical_cols)
# get our one hot encoded df
output = cat.fit_transform(df_mixed)

{% endhighlight %}

More to come soon in this series!
