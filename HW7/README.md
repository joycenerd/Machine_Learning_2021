# Kernel Eigenface and t-SNE

## What is dimensionality reduction?

Dimensionality reduction, or dimension reduction, is transforming data from a high-dimensional space into a low-dimensional space so that the low-dimensional representation retains some meaningful properties of the original data, ideally close to its intrinsic dimension. Working in high-dimensional spaces can be undesirable for many reasons; raw data are often sparse due to the curse of dimensionality, and analyzing the data is usually computationally intractable. Therefore, dimensionality reduction is expected in fields with large numbers of observations and/or large numbers of variables, such as signal processing, speech recognition, neuroinformatics, and bioinformatics.

## What is t-SNE?

t-distributed stochastic neighbor embedding (t-SNE) is a statistical method for visualizing high-dimensional data by giving each datapoint a location in a two or three-dimensional map. It is based on Stochastic Neighbor Embedding initially developed by Sam Roweis and Geoffrey Hinton, where Laurens van der Maaten proposed the t-distributed variant. Thus, a nonlinear dimensionality reduction technique is well-suited for embedding high-dimensional data for visualization in a low-dimensional space of two or three dimensions. Precisely, it models each high-dimensional object by a two- or three-dimensional point so that similar objects are modeled by nearby points and distant points with high probability model different objects.

## Dependencies

* pillow
* numpy
* matplotlib
* scipy

## Running this project


