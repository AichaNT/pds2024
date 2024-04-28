import numpy as np
import pandas as pd

man_an_fp = input("Filepath for manual annotations CSV: ")

man_an = pd.read_csv(man_an_fp, header=None, skiprows=1)

# Assuming man_an is your DataFrame
ass = man_an.loc[:, [1, 2, 3, 4, 5]]
col = man_an.loc[:, [6, 7, 8, 9, 10]]
dg = man_an.loc[:, [11, 12, 13, 14, 15]]

# Replace None values with NaN in ass and col
ass.replace('None', np.nan, inplace=True)
col.replace('None', np.nan, inplace=True)
dg.replace('None', np.nan, inplace=True)

# Convert columns to numeric in ass and col (optional, if they are not already numeric)
ass = ass.apply(pd.to_numeric, errors='ignore')
col = col.apply(lambda x: pd.to_numeric(x.str.replace(',', '.'), errors='coerce')) # to account for floats that were previously saved as strings using , instead of . 
dg = dg.apply(pd.to_numeric, errors='ignore')

# Create an empty dictionary to store the means for ass, col, and dg
means_dict = {}

# Iterate through each row in the DataFrame man_an
for index, row in man_an.iterrows():
    # Calculate the mean for the current row in ass
    ass_row_mean = ass.iloc[index].mean(skipna=True)
    
    # Calculate the mean for the current row in col
    col_row_mean = col.iloc[index].mean(skipna=True)

    # Calculate the mean for the current row in dg
    dg_row_mean = dg.iloc[index].mean(skipna=True)
    
    # Get the key from the first column of man_an and convert it to string
    key = str(row.iloc[0])
    
    # Store the means in the dictionary with the key from the first column of man_an
    means_dict[key] = [round(value, 2) for value in [ass_row_mean, col_row_mean, dg_row_mean]]

# Write means_dict to a CSV file
with open("means.csv", "w") as outfile:

    outfile.write("image_id,A,C,DG\n")

    for image_id, scores in means_dict.items():

        outfile.write(f"{image_id},{scores[0]},{scores[1]},{scores[2]}\n")

