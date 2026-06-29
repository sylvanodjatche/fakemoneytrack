import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']

local_file = 'data_banknote_authentication.txt'
if os.path.exists(local_file):
    print("[INFO] Chargement du dataset local...")
    df = pd.read_csv(local_file, header=None, names=columns)
else:
    print("[INFO] Téléchargement du dataset depuis UCI...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    df = pd.read_csv(url, header=None, names=columns)
    df.to_csv(local_file, index=False, header=False)
    print("[INFO] Dataset sauvegardé localement.")

X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

classifier = SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', probability=True)
classifier.fit(X_train, y_train)

joblib.dump(classifier, 'model_svm_banknote.pkl')
print("[SUCCESS] Modèle sauvegardé sous 'model_svm_banknote.pkl'")

y_pred = classifier.predict(X_test)
y_proba = classifier.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print("RAPPORT DE CLASSIFICATION")
print(classification_report(y_test, y_pred, target_names=['Authentique', 'Faux']))
print("="*50)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Authentique', 'Faux'],
            yticklabels=['Authentique', 'Faux'])
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title('Matrice de confusion - SVM RBF')
plt.savefig('confusion_matrix.png')
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.close()
print("[SUCCESS] Graphiques générés : confusion_matrix.png et roc_curve.png")
