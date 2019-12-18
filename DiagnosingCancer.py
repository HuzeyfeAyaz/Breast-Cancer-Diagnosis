# In this python file, our goal is automatically diagnosing patients as cancer or healthy.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup color map
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = -0.1,1.1
    x2_min, x2_max = -0.1,1.1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    x3.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

# Finding roc_curve and Auc score for each classifier
def find_roccurve(classifier,X_train,X_test,y_train,y_test):
    y_test_binary = np.where(y_test == 'patient',1,0)
    classifier.fit(X_train, y_train)
    y_prob = classifier.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test_binary, y_prob)  
    # print('AUC: %.2f' % auc)
    fpr, tpr, thresholds = roc_curve(y_test_binary, y_prob)
    return (auc, fpr, tpr, thresholds)

# Taking data from excel file
df = pd.read_excel('COH plasma data.xlsx', sheet_name='Sheet1')
df = df.dropna(axis=0, how='all',inplace=False)

columns = df.columns
patients = np.array(df[columns[8:]].iloc[15])
metabolites = np.array(df[columns[8:]].iloc[16:]).T # or you can use .values in pandas

# creating matrixes using data
X_train = metabolites[:70]
X_test = metabolites[70:]
patients = np.where(patients, 'patient', 'healty')
y_train = np.ravel(patients[:70])
y_test = np.ravel(patients[70:])
patient_types = df.iloc[2:5]

# to standardized data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combine_std = np.vstack((X_train_std, X_test_std))

# creating logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0, C=10.0)
lr.fit(X_train_std, y_train)
probabilities = lr.predict_proba(X_test_std)

# 10-k Cross-Validation
scores = []

# To calculate 10-K cross validation scores
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
add_score = lambda x: scores.append(
    round(sum(cross_val_score(x, X_combine_std, y=patients.ravel(), scoring='f1_micro', cv=10))*10,2))

add_score(LogisticRegression())
add_score(SVC())
add_score(RandomForestClassifier(n_estimators=40))

# Accuracy Scores
from sklearn.metrics import accuracy_score
y_pred = lr.predict(X_test_std)
y_pred = np.where(y_pred == 'patient', 1, 0)
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# Plot the scores
from sklearn.metrics import roc_curve  
from sklearn.metrics import roc_auc_score 

plt.figure(figsize=(12, 5))
x1 = plt.subplot(131)
log_reg = find_roccurve(LogisticRegression(),X_train_std,X_test_std,y_train,y_test)
svc = find_roccurve(SVC(probability=True),X_train_std,X_test_std,y_train,y_test)
rfc = find_roccurve(RandomForestClassifier(n_estimators=40),X_train_std,X_test_std,y_train,y_test)
x1.plot(log_reg[1], log_reg[2], color='orange', label='LR: %.2f'%log_reg[0])
x1.plot(svc[1], svc[2], color='blue', label='SVC: %.2f'%svc[0])
x1.plot(rfc[1], rfc[2], color='green', label='RFC: %.2f'%rfc[0])
x1.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
x1.set_xlabel('False Positive Rate')
x1.set_ylabel('True Positive Rate')
# x1.title('Receiver Operating Characteristic (ROC) Curve')
x1.title.set_text('ROC Curves')
x1.legend()

classifiers = ['LR','SVC','RFC']
y_pos = np.arange(len(classifiers))
x2 = plt.subplot(132)
clr = ('orange','blue', 'green', 'gold', 'red', 'purple')
hbar = x2.barh(y_pos, scores, align='center',color=clr)
x2.set_yticks(y_pos)
x2.set_yticklabels(classifiers)
x2.invert_yaxis()   # labels read top-to-bottom
x2.set_xlabel("Classifiers' Scores")
x2.title.set_text('10-K Cross-Validation')
x2.legend(hbar,scores)

y_test = np.where(y_test == 'patient', 1, 0)
errors = np.where(y_pred == y_test, True, False)

x3 = plt.subplot(133)
for i in range(len(y_pred)):
    if not errors[i]:
        x3.plot(probabilities[i,0],probabilities[i,1],'x',color='red',label='Wrong estimations')
    else:
        if y_test[i]:
            x3.plot(probabilities[i,0],probabilities[i,1],'x',color='blue',label='Patient cells')
        else:
            x3.plot(probabilities[i,0],probabilities[i,1],'x',color='green',label='Healty cells')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
x3.legend(by_label.values(), by_label.keys())
x3.set_xlabel('Healty Score')
x3.set_ylabel('Patient Score')
x3.title.set_text('Estimation by LogisticRegression')

cls_ = LogisticRegression()
cls_.fit(probabilities,y_pred)
plot_decision_regions(probabilities,y_pred,cls_)
# plt.savefig('breast_canser_new.png', dpi=300)
plt.show()