import pandas as pd

data = pd.read_csv('/home/dm-tomato/dm-gun/metaData/anotherModel/clip/image_classification_results.csv')
labels = data.drop(['image_path', 'probability'], axis=1)
labels.to_csv("image_classification_results_process.csv", index=False)