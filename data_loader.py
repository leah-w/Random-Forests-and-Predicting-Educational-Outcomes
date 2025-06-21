import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

"""
Fix data to make it work for sklearn 
sklearn does not support categorical variables, convert categorical to numerical 
"""

data = pd.read_csv("StudentPerformanceFactors.csv")
#drop irrelevant features
data_copy = data.copy()

categoricalCols = ['Parental_Involvement','Access_to_Resources','Extracurricular_Activities','Motivation_Level', 'Internet_Access', 'Family_Income','Teacher_Quality','School_Type','Peer_Influence','Learning_Disabilities','Parental_Education_Level', 'Distance_from_Home', 'Gender']

ordinalMap = {'Low': 0, 'Medium': 1, 'High': 2, 'No': 0, 'Yes': 1, 'High School': 1, 'College': 2, 'Postgraduate': 3,'Near': 0,'Moderate': 1, 'Far': 2, 'Negative': -1, 'Neutral': 0, 'Postitive': 1 , 'Male': 0, 'Female': 1, 'Public': 0, 'Private': 1}

# change categorical variables to ordinal 
for col in data_copy:
    if col in categoricalCols:
        try:
            data_copy[col] = data_copy[col].map(ordinalMap)
        except:
            print(type(data_copy[col][0]))

#maybe just remove of incomplete data
data_copy = data_copy.dropna()
#print(data_copy.shape)

    
# make Xmat and Y 
column_names = list(data_copy.columns)
Xmat = data_copy.drop(columns=["Exam_Score"])#.to_numpy(dtype=np.float64)
Y = data_copy["Exam_Score"]#.to_numpy(dtype=np.float64)

Xmat_train, Xmat_test, Y_train, Y_test = train_test_split(Xmat, Y, test_size=0.15, random_state=42)
Xmat_train, Xmat_val, Y_train, Y_val = train_test_split(Xmat_train, Y_train, test_size=0.15, random_state=42)
n, d = Xmat_train.shape

    
def get_data():
    """
    split data into train, test and validat
    """
    return Xmat_train, Xmat_test, Xmat_val, Y_train, Y_test, Y_val
