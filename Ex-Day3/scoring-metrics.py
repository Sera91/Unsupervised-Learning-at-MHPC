import numpy as np
# check that function is working correctly
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import f1_score
from operator import itemgetter



###################################################################################################################

def distance(data_points, centroids):
    """return distance matrix
     between vector of 2D points """
    N_points = len(data_points[:,0])
    N_clusters= len(centroids[:,0])
    #print(N_points)
    _dist_matrix =np.zeros((N_points, N_clusters), dtype=float)
    #distances_arr = []
    for i, centroid in enumerate(centroids):
            _dist_matrix[:,i] = np.sqrt((data_points[:,0] -centroid[0])**2 + (data_points[:,1]-centroid[1])**2)
    return _dist_matrix




#F1-score = 2 * (precision * recall) / (precision + recall)
#precision=TP/(TP+FP)
#recall=TP/(TP+FN)
def my_f1_score(assignments, labels, option):
    """
    this function returns an implementation
    of the multiclass F1-score
    metric to measure the goodness of a
    multiclass classification algorithm
    """
    N_points = len(assignments)
    unique_classes= np.unique(assignments)
    N_classes = len(unique_classes)
    _labels, counts_per_label  = np.unique(labels, return_counts=True)
    _assign, counts_per_assign = np.unique(assignments, return_counts=True)
    F1_score_arr=np.zeros(N_classes, dtype=np.float64)
    for i_class,cl in enumerate(unique_classes):
        indices_labels = np.argwhere(assignments==cl)
        indices_nolabels= np.argwhere(assignments!=cl)
        indices_classes = np.argwhere(labels==cl)
        TP = len(np.intersect1d(indices_labels,indices_classes))
        AP = len(np.union1d(indices_labels,indices_classes))#TP +FP
        FN = len(np.intersect1d(indices_classes, indices_nolabels))
        #print("N_FN:", FN)
        precision = TP/AP
        #print("precision:",precision)
        recall = TP/(TP+FN)
        #print("recall:", recall)
        if TP==0:
          F1_score_arr[i_class] = 0.0000005
        else:
          F1_score_arr[i_class] = 2 * (precision * recall) / (precision + recall)
    if option=='macro':
       #return the averaged f1-score
       Total_F1_score= np.sum(F1_score_arr)/N_classes
    elif option=='weighted':
       #returns the weighted averages of f1-score
       Total_F1_score=np.sum(counts_per_assign*F1_score_arr)/N_points

    return Total_F1_score

    
        



# functions to compute the normalized mutual information score
def normalized_mutual_info(assignments, labels):

    assert assignments.shape[0] == labels.shape[0]
    # get assigns probs
    computed_vals, computed_counts = np.unique(assignments, return_counts=True)
    computed_probs = computed_counts / assignments.shape[0]
    computed_entropy = - np.sum(np.multiply(computed_probs, np.log(computed_probs)))
    # get ground truth probs
    labels_vals, labels_counts = np.unique(labels, return_counts=True)
    labels_probs = labels_counts / labels.shape[0]
    true_entropy = - np.sum(np.multiply(labels_probs, np.log(labels_probs)))
    # get conditional probs
    mixed_counts = np.zeros(shape=(labels_vals.shape[0], computed_vals.shape[0]), dtype='int')
    for i in range(labels_vals.shape[0]):
        mask = (labels== labels_vals[i])
        for j in range(computed_vals.shape[0]):
            mixed_counts[i][j] += np.sum(assignments[mask]==computed_vals[j]) 
    mixed_probs = mixed_counts / assignments.shape[0]
    # compute mutual information
    mutual_info = 0.0
    for i in range(labels_vals.shape[0]):
        for j in range(computed_vals.shape[0]):
            if mixed_probs[i,j] != 0:
                mutual_info += mixed_probs[i,j] * np.log(mixed_probs[i,j]/(computed_probs[j]*labels_probs[i]))
    # return normalized mutual info
    return 2*mutual_info/(true_entropy+computed_entropy)



#############################################################################################


file_path  = 'datasets/'


option = str(input("Please enter the dataset option: A)Aggregation  or  B)s3"))
#option="B"

if option=="A":
    dataset  = "Aggregation"
    dist_cut = 2.5
    n_labels = 7
    input_file = file_path + 'Aggregation.txt'
    true_labels = np.genfromtxt(input_file, usecols=2)
    
else:   #S3 dataset
    dataset  = "S3"
    dist_cut = 57500
    n_labels   = 15
    input_file = file_path + 's3.txt'
    true_labels= np.loadtxt(file_path+'s3-label.pa')
    true_centroids= np.loadtxt(file_path+'s3-cb.txt')



ouptut_dir ='outputs/'
D_peaks_lab_file = ouptut_dir+ 'Dpeaks_labels_for_'+dataset+'-new_N'+str(n_labels)+'.txt'
K_means_lab_file = ouptut_dir+ 'Kmeans_labels_for_'+dataset+'with_K_'+str(n_labels)+'.txt'

K_means_cen_file = ouptut_dir+ 'Kmeans_centroids_for_'+dataset+'_with_K_'+str(n_labels)+'.txt'
D_peaks_cen_file = ouptut_dir+'Density_peaks_coords_for_'+dataset+'with_N_'+str(n_labels)+'.txt'

Dpeaks_labels = (np.genfromtxt(D_peaks_lab_file, usecols=2)).astype(int)
Kmeans_labels = (np.genfromtxt(K_means_lab_file, usecols=2)).astype(int)

Density_peaks = np.loadtxt(D_peaks_cen_file)
Kmeans_centroids = np.loadtxt(K_means_cen_file)


Density_peaks.shape

print("Density peaks coords:", Density_peaks)
print("Kmeans centroids coords:", Kmeans_centroids)


print("TESTING Kmeans algorithm")
print("on dataset "+dataset)
print("Manual implementation of F1-score :", my_f1_score(Kmeans_labels,true_labels,'macro' ))
print("sklearn F1-score:", f1_score(Kmeans_labels,true_labels,average='macro'))

print("Manual implementation of NMI coeff :", normalized_mutual_info(Kmeans_labels,true_labels ))
print("sklearn NMI score:", normalized_mutual_info_score(Kmeans_labels,true_labels))


print("TESTING Density-peaks algorithm")
print("on dataset "+dataset)
print("Manual implementation of F1-score (weighted)  :", my_f1_score(Dpeaks_labels,true_labels,'weighted' ))
print("sklearn F1-score (weighted) :", f1_score(Dpeaks_labels,true_labels,average='weighted'))

print("Manual implementation of NMI coeff :", normalized_mutual_info(Dpeaks_labels,true_labels ))
print("sklearn NMI score:", normalized_mutual_info_score(Dpeaks_labels,true_labels))


if dataset=="S3":
   dist_centers_Dpeaks=distance(Density_peaks, true_centroids)
   labels_D_mapped= (itemgetter(Dpeaks_labels-1)(np.argmin(dist_centers_Dpeaks, axis=1)))+1

   dist_centers_Kmeans=distance(Kmeans_centroids, true_centroids)
   labels_K_mapped= (itemgetter(Kmeans_labels-1)(np.argmin(dist_centers_Kmeans, axis=1)))+1
   print("TESTING Kmeans algorithm")
   print("on dataset "+dataset)
   print("after remapping the labels, reordering centroids coords according to the")
   print("list of true centroids coords.")
   print("Manual implementation of F1-score (weighted) :", my_f1_score(labels_K_mapped,true_labels,'weighted' ))
   print("Manual implementation of F1-score (weighted):", f1_score(labels_K_mapped,true_labels,average='weighted' ))
   print("Manual implementation of NMI coeff :", normalized_mutual_info(labels_K_mapped,true_labels ))
   print("sklearn NMI score:", normalized_mutual_info_score(labels_K_mapped,true_labels))

   print("TESTING Density-peaks algorithm")
   print("on dataset "+dataset)
   print("after remapping the labels, reordering centroids coords according to the")
   print("list of true centroids coords.")
   print("Manual implementation of F1-score (weighted)  :", my_f1_score(labels_D_mapped,true_labels,'weighted' ))
   print("Manual implementation of F1-score (weighted) :", f1_score(labels_D_mapped,true_labels,average='weighted' ))
   print("Manual implementation of NMI coeff :", normalized_mutual_info(labels_D_mapped,true_labels ))
   print("sklearn NMI score:", normalized_mutual_info_score(labels_D_mapped,true_labels))





