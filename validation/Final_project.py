# %%
import os
import joblib
import sys
sys.path.append(os.path.abspath("C:/Users/euderasm/GitHub/ud120-projects/tools/"))
from feature_format import featureFormat, targetFeatureSplit

# %%
data_dict = joblib.load(open("../final_project/final_project_dataset.pkl", "rb") )

# %%
### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

# %%
data[:5]

# %%
data.shape

# %%
### it's all yours from here forward! 

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# %%
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.3, random_state=42)

# %%
clf = DecisionTreeClassifier().fit(features, labels)
clf.score(features, labels)

# %%
clf = DecisionTreeClassifier().fit(X_train, y_train)
clf.score(X_test, y_test)

# %%



