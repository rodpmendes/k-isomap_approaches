from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn import metrics

#import sys
#sys.path.append('../git_alexandrelevada/Supervised_tSNE')
#from SE_tSNE_2D import estimate_sne, p_joint, Classification, q_tsne, q_joint, tsne_grad, symmetric_sne_grad

'''
 Computes the Silhouette coefficient, kappa and balanced accuracy and the supervised classification
 accuracies for several classifiers: KNN, SVM, NB, DT, QDA, MPL, GPC and RFC
 dados: learned representation (output of a dimens. reduction - DR)
 target: ground-truth (data labels)
 method: string to identify the DR method (PCA, NP-PCAKL, KPCA, ISOMAP, LLE, LAP, ...)
 
 variation from SE_tSNE_2D.py
'''
def myClassification(dados, target, mode='holdout'):
    
    list_acc = []
    list_kappa = []
    list_balanced_acc = []
    list_classifiers_labels = []
    

    # 8 different classifiers
    neigh = KNeighborsClassifier(n_neighbors=5)
    svm = SVC(gamma='auto')
    nb = GaussianNB()
    dt = DecisionTreeClassifier(random_state=42)
    qda = QuadraticDiscriminantAnalysis()
    mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=1000)
    gpc = GaussianProcessClassifier()
    rfc = RandomForestClassifier()

    if mode == 'holdout':
        # 50% for training and 40% for testing
        X_train, X_test, y_train, y_test = train_test_split(dados.real, target, train_size=0.5, random_state=42)

        # KNN
        neigh.fit(X_train, y_train) 
        acc = neigh.score(X_test, y_test)
        labels_knn = neigh.predict(X_test)
        kappa = metrics.cohen_kappa_score(labels_knn, y_test)
        balanced_acc = metrics.balanced_accuracy_score(y_test, labels_knn)
        list_acc.append(acc)
        list_kappa.append(kappa)
        list_balanced_acc.append(balanced_acc)
        list_classifiers_labels.append('KNN')

        # SMV
        svm.fit(X_train, y_train) 
        acc = svm.score(X_test, y_test)
        labels_svm = svm.predict(X_test)
        kappa = metrics.cohen_kappa_score(labels_svm, y_test)
        balanced_acc = metrics.balanced_accuracy_score(y_test, labels_svm)
        list_acc.append(acc)
        list_kappa.append(kappa)
        list_balanced_acc.append(balanced_acc)
        list_classifiers_labels.append('SMV')
        

        # Naive Bayes
        nb.fit(X_train, y_train)
        acc = nb.score(X_test, y_test)
        labels_nb = nb.predict(X_test)
        kappa = metrics.cohen_kappa_score(labels_nb, y_test)
        balanced_acc = metrics.balanced_accuracy_score(y_test, labels_nb)
        list_acc.append(acc)
        list_kappa.append(kappa)
        list_balanced_acc.append(balanced_acc)
        list_classifiers_labels.append('Naive Bayes')

        # Decision Tree
        dt.fit(X_train, y_train)
        acc = dt.score(X_test, y_test)
        labels_dt = dt.predict(X_test)
        kappa = metrics.cohen_kappa_score(labels_dt, y_test)
        balanced_acc = metrics.balanced_accuracy_score(y_test, labels_dt)
        list_acc.append(acc)
        list_kappa.append(kappa)
        list_balanced_acc.append(balanced_acc)
        list_classifiers_labels.append('Decision Tree')


        # Quadratic Discriminant 
        qda.fit(X_train, y_train)
        acc = qda.score(X_test, y_test)
        labels_qda = qda.predict(X_test)
        kappa = metrics.cohen_kappa_score(labels_qda, y_test)
        balanced_acc = metrics.balanced_accuracy_score(y_test, labels_qda)
        list_acc.append(acc)
        list_kappa.append(kappa)
        list_balanced_acc.append(balanced_acc)
        list_classifiers_labels.append('Quadratic Discriminant')

        # MPL classifier
        mpl.fit(X_train, y_train)
        acc = mpl.score(X_test, y_test)
        labels_mpl = mpl.predict(X_test)
        kappa = metrics.cohen_kappa_score(labels_mpl, y_test)
        balanced_acc = metrics.balanced_accuracy_score(y_test, labels_mpl)
        list_acc.append(acc)
        list_kappa.append(kappa)
        list_balanced_acc.append(balanced_acc)
        list_classifiers_labels.append('MPL Classifier')

        # Gaussian Process
        gpc.fit(X_train, y_train)
        acc = gpc.score(X_test, y_test)
        labels_gpc = gpc.predict(X_test)
        kappa = metrics.cohen_kappa_score(labels_gpc, y_test)
        balanced_acc = metrics.balanced_accuracy_score(y_test, labels_gpc)
        list_acc.append(acc)
        list_kappa.append(kappa)
        list_balanced_acc.append(balanced_acc)
        list_classifiers_labels.append('Gaussian Process')
        
        # Random Forest Classifier
        rfc.fit(X_train, y_train)
        acc = rfc.score(X_test, y_test)
        labels_rfc = rfc.predict(X_test)
        kappa = metrics.cohen_kappa_score(labels_rfc, y_test)
        balanced_acc = metrics.balanced_accuracy_score(y_test, labels_rfc)
        list_acc.append(acc)
        list_kappa.append(kappa)
        list_balanced_acc.append(balanced_acc)
        list_classifiers_labels.append('Random Forest Classifier')

        return list_classifiers_labels, list_acc, list_kappa, list_balanced_acc
