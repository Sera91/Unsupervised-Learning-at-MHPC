import numpy as np
import pandas as pd
# test function against scikit implementation
from sklearn.feature_selection import mutual_info_classif
from scipy.spatial.distance import hamming


#2nd exercise : 
#Download the Congressional Voting Records dataset from:
# http://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
#rank the importance of the variables for classifying the party of vote according with
#the mutual information criteria.



###SUBROUTINES #####################################################################à

def mutual_info_myclassif(df, feature_col, target_col):  #for categorical variables
    
    # estimate 1st var prob
    feature_vals = df[feature_col].unique()
    feature_prob = df[feature_col].value_counts() / df.shape[0]
    # estimate 2nd var probs
    target_vals = df[target_col].unique()
    target_prob = df[target_col].value_counts() / df.shape[0]
    # get conditional probs on the pair of var
    mixed_prob = df.groupby(by=[target_col,feature_col]).agg(counts = (feature_col, pd.Series.count)) / df.shape[0]
    
    # compute mutual info
    mutual_info = 0.0
    for target in target_vals:
        for feature in feature_vals:
            idx = (target,feature)
            mutual_info += mixed_prob.loc[idx]['counts'] * \
                           np.log(mixed_prob.loc[idx]['counts']/ \
                           (feature_prob[feature]*target_prob[target]))
    
    return mutual_info


def my_hamming_dist(obs_i, obs_j):
    return (obs_i != obs_j).sum()






# import data from the voting dataset
cols = ["class", "handicapped_infants", "water_project_cost_sharing", "adoption_budget_resolution",
        "physician_fee_freeze", "el_salvador_aid", "religious_groups_in_school", "anti_satellite_test_ban",
        "aid_nicaraguan_contras", "mx_missile", "immigration", "synfuels_corporation_cutback", 
        "education_spending", "superfund_right_to_sue", "crime", "duty_free_exports", 
        "export_administration_act_SA"
       ]
votes_df = pd.read_csv("house-votes-dataset/house-votes-84.data", names=cols)
votes_df.head()

# test function against scikit implementation

class_to_int = votes_df['class'].apply(lambda x: 1.0 if x=='republican' else 0.0).to_numpy()

for feature in cols[1:]:
	feat_to_int = votes_df[feature].apply(lambda x: 2.0 if x=='?' else (1.0 if x=='y' else 0.0)).to_numpy()
	feat_to_int = feat_to_int[:,np.newaxis]
	my_MIC= mutual_info_myclassif(votes_df, target_col='class', feature_col=feature)
	scikit_MIC=mutual_info_classif(feat_to_int, class_to_int, discrete_features=True)[0]
	print("Feature:",feature, '|  MI value=',my_MIC," | diff with Scikit:", (my_MIC - scikit_MIC))
        
        

#sorting the variables for their MI coeff value, estimated with respect to the voting party column
_ranks = []
for feature in cols[1:]:
    _ranks.append(mutual_info_myclassif(votes_df, feature_col=feature, target_col='class'))
    
cols_arr = np.array(cols[1:])

features_ranked= cols_arr[np.argsort(np.abs(np.array(_ranks)))[::-1] ]


print("Features ranked depending on their MI value respect to voting party:")
print(features_ranked)

#testing personal version of hamming distance with respect to the version of scipy

val_one=np.array([0,1,1,0,0,0])
val_two=np.array([1,0,1,0,0,0])

dis_scipy = hamming(val_one, val_two) * len(val_one)
dis_mine= my_hamming_dist(val_one, val_two)

test_result='not passed'
if (dis_scipy - dis_mine)< 1.e-14:
   test_result='passed!'

print("test Hamming distance routine: ", test_result )


#Computing average Hamming distance among points associated to Republican (in-class)
#and between republican point 0 and points associated to democratic (out-class)


votes_df_republican = votes_df[votes_df["class"] == 'republican']
votes_df_republican = votes_df_republican.drop(columns='class')
          

votes_df_democratic = votes_df[votes_df["class"] != 'republican']
votes_df_democratic = votes_df_democratic.drop(columns='class')

for feature in cols[1:]:
       votes_df_republican[feature] = votes_df_republican[feature].apply(lambda x: 2 if x=='?' else (1 if x=='y' else 0))
       votes_df_democratic[feature] = votes_df_democratic[feature].apply(lambda x: 2 if x=='?' else (1 if x=='y' else 0))

dist_arr_in_class = np.array([my_hamming_dist(votes_df_republican.iloc[0].values, votes_df_republican.iloc[i].values) for i in range(votes_df_republican.shape[0])])
dist_arr_out_class = np.array([my_hamming_dist(votes_df_republican.iloc[0].values, votes_df_democratic.iloc[i].values) for i in range(votes_df_democratic.shape[0])])

print('the mean Hamming distance between point 0 in the Republican dataframe and other points associated to Republican is:', np.mean(dist_arr_in_class[1:]))

print('while, the mean Hamming distance between point 0 in the Republican dataframe and points associated to democratics is:', np.mean(dist_arr_out_class))

print('therefore we can observe that the average Hamming distance is smaller between points in the same class, as expected.')


