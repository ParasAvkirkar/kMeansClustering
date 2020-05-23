Commands to execute code:

python kmeans.py --dataset datasetPath --k 2 --distance Euclidean


python kmeans.py --dataset datasetPath --k 2 --distance Manhattan


The above input commands to run the file specify the k-parameter as well.
Please specify the 'k' parameter as shown above while testing the code.

The K-Means algorithm is implemented using standard pseudocode at 22.2.1 section in UML

The algorithm to select initial centroids for K-Means algorithm is inspired from the K-Means++ algorithm.
Citing for centroid selection algorithm:
1. https://en.wikipedia.org/wiki/K-means%2B%2B
2. http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf

Centroid Selection Algorithm
The centroids are selected from the dataset itself.
The first centroid is selected based on Uniform Distribution (All points in dataset have equal probability)

The next (k-1) centroids are selected as follows.
For each of the point in dataset, we calculate it's distance with the currently selected closest centroid.
We compute statistic of the square of this distance.
After generating distribution (ePMF) based on above statistic we draw our next sample as next centroid.


To terminate the k-means algorithm we use follow as convergence criteria.
If the cluster assignments after each iteration are same as before then we terminate the k-means clustering algorithm.
In such a case, the centroids computed will be same as before.