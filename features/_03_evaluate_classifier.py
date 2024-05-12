"""
@authors: cjep, kmah, feso, aith, nozo
"""

import pickle 
import pandas as pd
import matplotlib.pyplot as plt

from extract_features import asymmetry, color, dg


# The image and mask are the same size, and are already loaded using plt.imread
def classify(img, mask):

     """
     Function that classifies new images
     
     Parameters:
     • img: numpy array, the input image
     • mask: numpy array, the corresponding input mask
     
     Returns:
     • pred_label: the predicted label for the image
     • pred_prob: the predicted probability
     """
    
     # Save extracted features in dictionary
     feature_scores = {'A': [asymmetry(img, mask)], 
                       'C': [color(img, mask)], 
                       'DG': [dg(img, mask)]}

     x = pd.DataFrame.from_dict(feature_scores, orient='columns')

     # Load the trained classifier
     classifier = pickle.load(open('features/groupK_classifier.sav', 'rb'))
    
     # Use it on this example to predict the label AND posterior probability
     pred_label = classifier.predict(x)
     pred_prob = classifier.predict_proba(x)[:, 1]  # Probability of class 1
     
     
     #print('predicted label is ', pred_label)
     #print('predicted probability is ', pred_prob)

     return int(pred_label), float(pred_prob)
 
    
# The TAs will call the function above in a loop, for external test images/masks

