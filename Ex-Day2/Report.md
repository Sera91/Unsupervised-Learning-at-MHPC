In this directory are hosted two Python codes:
- K-means-final.py
- Fuzzy-Cmeans.py

which perform the analysis of the clustering of points in the dataset S3,
respectively exploiting the Kmeans algorithm (both with random and K++ inizialization of centroids),
and the Fuzzy Cmeans algorithm, as requested by the Exercises listed in Practices_2.pdf .


To run the first Python code we can launch the following command from terminal:
- python K-means-final.py 

and then type A or B, respectively to analyze the Aggregation and the S3 datasets.
The same is valid for the other code.



Both codes have been runned on the  S3 and Aggregation dataset, producing the plots in the subdir plots,
and the output files in the subdir outputs.

For the S3 dataset I have also done the SCREE plot, using the best value of the objective/cost function, which confirms that the best K for the number of clusters associated to this dataset is K=15.




