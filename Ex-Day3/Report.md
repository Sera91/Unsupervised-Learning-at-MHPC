This directory contains a Python code:
- density-peaks.py
performin a clustering analysis of the datasets Aggregation and S3, based on a density peaks algorithm developed by me.

To run the code which perform the clustering analysis of the wanted dataset through the density peaks algorithm,
you can launch the following command from terminal:
- python density-peaks.py 

and then type A or B, respectively to analyze the Aggregation and the S3 datasets.

This code will produce the plots shown in sub-dir plots, and the output files shown in the sub-dir outputs.
In the latter subdir there are 2 types of files.
One with the coordinates of the best centroids assigneds by the Kmeans algorithm or  the density peaks found with
the density peaks algorithm.
The second type of file contains instead the coordinate of all the poind and a column with the cluster assignment of each point.

Reading these output files,  generated respectively by the Kmeans code and the density-peak code, we can perform the clustering evaluation, with the metrics F1-score and Normalized mutual information (NMI), running the python code:

scoring-metrics.py

and then typing A or B, respectively to analyze the Aggregation and the S3 datasets.

These code will print to the screen an output similar to the one showed in the file in dir output-scores/

As we can see from the score estimated on the S3 dataset, where i tested the effect of reording the coordinated of the centroids found by my algorithm accordingly to the list of reference centroids, the F1-score is very much dependent on the matching order of the label assigned ( meaning that the label 1 in ref and in mine output should correspond to the same centroid) while the value estimated with the NMI score is not dependent on this matching and therefore it is more practical to test in general.

We can observe also that, according to the NMI score, on the Aggregation dataset the density peak algorithm performs a better assignment respect to the the Kmeans algorithm (as confirmed looking at the plots), while on the S3 dataset the score of the two algorithms are very similar.

