# imports
# Imports core Python libraries
import pandas as pd
import numpy as np

# visualization libraries 
import matplotlib.pyplot as plt
import seaborn as sns

# Classification Metrics 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc #, plot_roc_curve
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# Import models from scikit lear
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

def model_selection(X,Y):
    seed = 7
    models = []
    #models.append(('LogisticRegression', LogisticRegression(random_state=seed)))
    models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
    models.append(('KNeighborsClassifier', KNeighborsClassifier()))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))

    models.append(('RandomForestClassifier', RandomForestClassifier()))
    models.append(('ExtraTreesClassifier',ExtraTreesClassifier(random_state=seed)))
    models.append(('AdaBoostClassifier',AdaBoostClassifier(
                      DecisionTreeClassifier(random_state=seed),random_state=seed,learning_rate=0.1))
                 )
    models.append(('SVM',svm.SVC(random_state=seed)))
    models.append(('GradientBoostingClassifier',GradientBoostingClassifier(random_state=seed)))
    models.append(('XGBoost', xgb.XGBClassifier(random_state=seed)))
    #models.append(('CatBoost', CatBoostClassifier(iterations=100,l2_leaf_reg=2,random_state=seed)))
    #models.append(('MLPClassifier',MLPClassifier(random_state=seed)))
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'f1'
    
    for name, model in models:
        kfold = KFold(n_splits=10, shuffle=True, random_state=seed) 
        cv_results = cross_val_score(model, X,Y, cv=kfold, scoring=scoring,verbose=False)
        results.append([name, round(cv_results.mean(),2), round(cv_results.std(),3)])
        names.append(name)
        
    results_df = pd.DataFrame(results, columns=['Model Name', 'F1_Mean', 'F1_Standard Deviation'])
    return results_df



def ROC(classifier,X_train,X_test,y_train,y_test) : 

    # set the increment step
    inc = .05
    
    # get the predicted probabilities of the positive class
    y_score_train = classifier.predict_proba(X_train)[:,1]
    y_score_test = classifier.predict_proba(X_test)[:,1]
    
    # calculate y_train and y_test
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)
    
    # calculate the fpr, tpr and thresholds for each increment
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_score_test)
    tpr_test_smooth = []
    fpr_test_smooth = []
    
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_score_train)
    tpr_train_smooth = []
    fpr_train_smooth = []

    
    # interpolate the ROC curve at each increment
    for i in np.arange(0, 1, inc):
        tpr_test_smooth.append(np.interp(i, fpr_test, tpr_test))
        fpr_test_smooth.append(i)
        tpr_train_smooth.append(np.interp(i, fpr_train, tpr_train))
        fpr_train_smooth.append(i)

    # fixing the low and high end points of the ROC curve    
    tpr_train_smooth[0] = 0     
    tpr_test_smooth[0] = 0     
    fpr_train_smooth[0] = 0    
    fpr_test_smooth[0] = 0  
    
    tpr_train_smooth[-1] = 1     
    tpr_test_smooth[-1] = 1      
    fpr_train_smooth[-1] = 1      
    fpr_test_smooth[-1] = 1   
    
    # calculate AUC
    roc_auc_test = auc(fpr_test_smooth, tpr_test_smooth)
    roc_auc_train = auc(fpr_train_smooth, tpr_train_smooth)
    
    # Set the plot size
    plt.subplots(figsize=(8,5))

    # plot the ROC curve with smooth interpolation
    plt.plot(fpr_test_smooth, tpr_test_smooth, lw=2, label='ROC curve for TEST (AUC = %0.2f)' % roc_auc_test)
    plt.plot(fpr_train_smooth, tpr_train_smooth, lw=2, label='ROC curve for TRAIN (AUC = %0.2f)' % roc_auc_train)

    # plot the random guessing line
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random guess')

    # set plot title and labels
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    # show the plot
    plt.show()
    
    # Evaluate the model using a classification report
    training_report = classification_report(y_train, y_train_pred)
    print("TRAINING classification report: \n",training_report)
    
    # Evaluate the model using a classification report
    testing_report = classification_report(y_test, y_test_pred)
    print("\nTESTING classification report: \n",testing_report)
    
    # Calculate the accuracy, precision, F1 and recall of the model
    f1 = f1_score(y_test, y_test_pred, average='micro')
    accuracy= accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    
    # Print the results
    print(f'Test accuracy: {accuracy:.2f}')
    print(f'>> Test precision: {precision:.2f} <<') 
    print(f'Test recall: {recall:.2f}')
    print(f'Test F1 score: {f1:.2f}')
    print(f'Test AUC score: {roc_auc_test:.2f}')