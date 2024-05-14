import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score



man_path = 'additional_data/mean_manual_ano.csv'
auto_path = 'data/features.csv'

man_df = pd.read_csv(man_path)
auto_df = pd.read_csv(auto_path)

# Merge the DataFrames on 'image_id'
merged_df = pd.merge(man_df, auto_df, on='image_id')

# Changing scores to floats
merged_df.iloc[:, 1:] = merged_df.iloc[:, 1:].astype(float)

# Create separate DataFrames for each feature
a_df = merged_df[['image_id', 'A_x', 'A_y']]
c_df = merged_df[['image_id', 'C_x', 'C_y']]
dg_df = merged_df[['image_id', 'DG_x', 'DG_y']]


# Asymmetry cohen kappa score when rounding
a_man = [round(value) for value in a_df['A_x']]
a_auto = [round(value) for value in a_df['A_y']]

cohen_kappa_score(a_man, a_auto)


# Dots/globules cohen kappa score when rounding 
dg_man = [round(value) for value in dg_df['DG']]
dg_auto = [round(value) for value in dg_df['DG']]

cohen_kappa_score(dg_man, dg_auto)


# Define the transformation function
def transform_score(score):
    if 0.5 <= score < 0.75:
        return 1.5
    elif 0.75 <= score < 1.25:
        return 1.0
    elif 1.25 <= score < 1.75:
        return 1.5
    elif 1.75 <= score < 2.25:
        return 2.0
    elif 2.25 <= score < 2.75:
        return 2.5
    elif 2.75 <= score < 3.25:
        return 3.0
    return score

# Apply the transformation only to the C_x column
c_df.loc[:, 'C_x'] = c_df.loc[:, 'C_x'].apply(transform_score)
c_df.loc[:, ['C_x', 'C_y']] *= 10

# Color cohen kappa score
c_man = list(c_df['C_x'].astype(int))
c_auto = list(c_df['C_y'].astype(int))

cohen_kappa_score(c_man, c_auto)
