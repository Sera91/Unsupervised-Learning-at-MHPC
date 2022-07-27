import numpy as np
import matplotlib.pyplot as plt
import sys



######################SUBROUTINES####################################################
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




def plot_graph(rho, delta, reversed_indices, title, dataset_name , save_path=''):
        """
        Plot decision graph,
        taking as arguments:
        -rho, density
        -delta, distance from the point at higher rho
        -reversed_indices, indices associated to the point in the input dataset
        -dataset_name
        -title, title of the plot
        and return: None
        """   
                        
        x_label='density'
        y_label='distance'


        fig = plt.figure(1, figsize=(15,10))
        ax = fig.add_subplot(111)
        ax.scatter(x=rho, y=delta)

        #indices = np.arange(len(rho))

        for index, x, y in zip(reversed_indices, rho, delta):
                plt.annotate(index, (x, y), fontsize=10)

        print('instance count: %d ' % (len(rho)))
        

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if (save_path != ''):
            plt.savefig(fname= save_path + title +'-' + dataset_name + '-new.pdf')
            plt.clf()
        else:
            pass

def distance_2D(point_A, vector_point_B):
    """
    estimates Euclidean distance in 2D,
    between a point and one
    array of point 
    """

    dist = np.sqrt(np.power(point_A[0]-vector_point_B[:,0],2) + np.power(point_A[1]-vector_point_B[:,1], 2))

    return dist


def density_exp(dist_arr, dist_cutoff):
    density_approx = np.sum(np.exp(-(np.power((dist_arr/dist_cutoff),2.0))))
    return density_approx

def density_exp_vec(dist_matrix, dist_cutoff):
    out = np.sum(np.exp(-(np.power((dist_matrix/dist_cutoff),2.0))), axis=1)
    return out






#Density-peaks algorithm
#STEP 1: We estimate the density associated to each point in the dataset
#STEP 2: After estimating the density for each point in the dataset we sort the density array, from higher density to lower density
#STEP 3: we sort the array containing point coord indices, according to the sorting of the density array
#        and we estimate the distance between each point and the point which have density associated with higher value,
#        that are all hosted in the position range [0:i-point] in the new array 
#STEP 4: we build a new array (delta_arr) in which we save the minimum distance estimated for each point

def density_peaks(data_inp, dist_cut, dataset_nam):
        N_points = len(data_inp[:,0])
        dist_matrix = distance(data_inp, data_inp)
        print(dist_matrix.shape)
        #density_arr=np.zeros(N_points,dtype= np.float64)
        delta_arr = np.zeros(N_points,dtype= np.float64)
        #density_exp_arr_test=np.zeros(N_points,dtype= np.float64)
        density_exp_arr= density_exp_vec(dist_matrix, dist_cut) 
        #for i,point in enumerate(data_inp):
        #     dist_arr = dist_matrix[i]
        #     sel_dist_in_cluster= dist_arr[np.where(dist_arr<dist_cut)[0]]
        #     i_density = len(sel_dist_in_cluster) - 1
        #     i_density_exp = density_exp(dist_arr, dist_cut)
        #     print('density_old_formula:',i_density)
        #     print('density_exp_formula:',i_density_exp)
        #     density_arr[i]=  i_density
        #     density_exp_arr_test[i] =i_density_exp

        #print("checking fast densit arr:")
        #print(np.sum(density_exp_arr - density_exp_arr_test))
       
        reversed_density_arr = np.sort(density_exp_arr)[::-1]

        reversed_indices_Pdensity=np.argsort(density_exp_arr)[::-1]
        
        max_density=reversed_density_arr[0]#np.max(density_arr)
        print("max density:", max_density)

        reversed_coords_Pdensity= data_inp[reversed_indices_Pdensity]	
	#N_highest_point = len(indices_highest_density)
    
        delta_arr[1:] = [np.min(dist_matrix[reversed_indices_Pdensity[i]][reversed_indices_Pdensity[0:i]]) for i in range(1,N_points)]
        
        delta_arr[0]= np.max(delta_arr[1:])+1
        

        print(np.vstack((reversed_density_arr, delta_arr)).T)

        plot_graph(reversed_density_arr, delta_arr, reversed_indices_Pdensity, "decision_graph", dataset_nam, save_path='plots/')

        indices_outliers_sorted_per_delta = reversed_indices_Pdensity[np.argsort(delta_arr)[::-1]]
        density_outliers_sorted_per_delta = reversed_density_arr[np.argsort(delta_arr)[::-1]]
        return indices_outliers_sorted_per_delta, density_outliers_sorted_per_delta 



#######################################MAIN######################################

option_dataset = str(input("Please enter the dataset option: A)Aggregation  or  B)s3"))
#dataset="Aggregation"
dataset='Aggregation'

if option_dataset=='A':
    dataset="Aggregation"
    dist_cut = 2.5
    i_dist =1
    file_path = 'datasets/'
    n_labels = 7 #6
    input_file = file_path + 'Aggregation.txt'
else:   #S3 dataset
    dataset="S3"
    dist_cut = 57500
    i_dist    = 2
    file_path = 'datasets/'
    n_labels  = 15
    input_file = file_path + 's3.txt'





data = np.genfromtxt(input_file, usecols=[0,1])



indices_outliers, density_outliers = density_peaks(data, dist_cut, dataset)

half_mean_density = 0.5*np.mean(density_outliers)  #

real_outliers = indices_outliers[np.argwhere(density_outliers> half_mean_density)]

density_peaks = (data[real_outliers[0:n_labels]]).reshape(n_labels,2)


#OUTPUT FILE with centroids coords
desired_fmt= '%8.2f', '%8.2f'
file_output_peaks="outputs/Density_peaks_coords_for_"+ dataset+"with_N_"+str(n_labels)+".txt"
data_output_peaks= np.column_stack((density_peaks[:,0], density_peaks[:,1]))
np.savetxt(file_output_peaks, data_output_peaks, fmt=desired_fmt, delimiter='  ', header='x-coord    y-coord')


#density_peaks.reshape(n_labels,2)

dist_points_peaks = distance(data, density_peaks)

labels_arr = np.argmin(dist_points_peaks, axis=1)

clusters_figname = "points_"+ dataset +"_colored_density_peaks_witk_N_"+str(n_labels)+'.pdf'
plt.figure(1)
plt.title("plots/Cluster assignation by density-peaks algorithm")
plt.scatter(data[:,0], data[:,1], c=labels_arr)
plt.scatter(density_peaks[:,0], density_peaks[:,1], s=60, marker="x", c='black', alpha=0.6, label='peaks')
plt.legend()
plt.savefig(clusters_figname)
plt.show()

desired_fmt= '%8.2f', '%8.2f', '%10d'
file_output="outputs/Dpeaks_labels_for_"+ dataset+'-new_K'+str(n_labels)+'.txt'
data_output = np.column_stack((data[:,0], data[:,1], (labels_arr+1)))
np.savetxt(file_output, data_output, fmt=desired_fmt, delimiter='  ', header='x-coord    y-coord  labels')






