P6
from os import listdir
import pandas as pd
import numpy as np1
import numpy.linalg as np
import matplotlib.pyplot as plt

data = pd.read_csv('tips.csv')
bill = np1.array(data.total_bill)
tip = np1.array(data.tip)

mbill = np1.asmatrix(bill)
mtip = np1.asmatrix(tip)
m = np1.shape(mbill)[1]

one = np1.asmatrix(np1.ones(m))
x = np1.hstack((one.T, mbill.T))
print(x)

def kernel(point, xmat, k):
    m, n = np1.shape(xmat)
    weights = np1.asmatrix(np1.eye(m))
    for j in range(m):
        diff = point - xmat[j]
        weights[j, j] = np1.exp(diff * diff.T / (- 2.0 * k ** 2))
    return weights

def localWeight(point, xmat, ymat, k):
    wei = kernel(point, xmat, k)
    xtwx = x.T * (wei * X)
    xtwx += np1.eye(xtwx.shape[0]) * 1e-5
    w = xtwx.I * (x.T * (wei * ymat.T))
    return w

def lwr(xmat, ymat, k):
    m,n = np1.shape(xmat)
    ypred = np1.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i]*localWeight(xmat[i], xmat, ymat, k)
    return ypred

ypred = lwr(x, mtip, 0.3)
si = x[:, 1].argsort(0)
xsort = x[si][:,0]

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(1,1,1)
ax.scatter(bill, tip, color = "blue", label = "actual data")
ax.plot(xsort[:,1], ypred[si], color = "red", linewidth = 2, label = "pred")
plt.xlabel("totalbill")
plt.ylabel("tip")
plt.show()
        
P7
def linear_regression_boston():
    b_housing = pd.read_csv('BostonHousing.csv')
    X = b_housing[['rm']]
    y = b_housing['medv']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plt.plot(X_test, y_pred, color="red", label="Predicted")
    plt.xlabel("Average number of rooms (AveRooms)")
    plt.ylabel("Median value of homes ($100,000)")
    plt.title("Linear Regression - Boston Housing Dataset")
    plt.legend()    
    plt.savefig('LinearRegression.png')
    plt.show()
    
    print("Linear Regression - Boston Housing Dataset")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))
def polynomial_regression_auto_mpg():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    column_names = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]
    data = pd.read_csv(url, sep='\s+', names=column_names, na_values="?")
    data = data.dropna()
    
    X = data["displacement"].values.reshape(-1, 1)
    y = data["mpg"].values
    X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    poly_model = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), LinearRegression())
    poly_model.fit(X_train, y_train)
    
    y_pred = poly_model.predict(X_test)
    
    plt.scatter(X_test, y_test, color="blue", label="Actual")    
    plt.scatter(X_test, y_pred, color="red", label="Predicted")
    plt.xlabel("Displacement")
    plt.ylabel("Miles per gallon (mpg)")
    plt.title("Polynomial Regression - Auto MPG Dataset")
    plt.legend()
    plt.savefig('PolynomialRegression.png')
    plt.show()
    print("Polynomial Regression - Auto MPG Dataset")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))

P8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree 
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
x = data.data
y = data.target

xtr, xte, ytr, yte = train_test_split(x, y, test_size = 0.2, random_state = 42)

clf = DecisionTreeClassifier(random_state = 42)
clf.fit(xtr, ytr)

ypr = clf.predict(xte)

accuracy = accuracy_score(yte, ypr)
print(f"Accuracy: {accuracy*100:.2f}%")

newsm = np.array([xte[0]])
prediction = clf.predict(newsm)

prcls = "Benign" if prediction == 1 else "Malignant"
print(f"prediction class : {prcls}")

plt.figure(figsize=(12,8))
tree.plot_tree(clf, filled=True, label="all")
plt.title("decision tree")
plt.show()



P9
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = fetch_olivetti_faces(shuffle=True, random_state = 42)
x = data.data
y = data.target

xtr, xte, ytr, yte = train_test_split(x, y, test_size = 0.3, random_state = 42)

gnb = GaussianNB()
gnb.fit(xtr,ytr)
ypr = gnb.predict(xte)

accuracy = accuracy_score(yte, ypr)
print(f"accuracy = {accuracy*100:.2f}")

print("CR\n")
print(classification_report(yte,ypr,zero_division = 1))
print("CM\n")
print(confusion_matrix(yte,ypr))

cva = cross_val_score(gnb, x, y, cv = 5, scoring="accuracy")
print(f"cva : {cva.mean()*100:.2f}%")

fig, axes = plt.subplots(3, 5, figsize = (12,8))
for ax, image, label, prediction in zip(axes.ravel(), xte, yte, ypr):
    ax.imshow(image.reshape(64,64), cmap = plt.cm.gray)
    ax.set_title(f"true: {label}, pred: {prediction}")
    ax.axis('off')

plt.show()


P10
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
x = data.data
y = data.target

scaler = StandardScaler()
Xscaled = scaler.fit_transform(x)

kmeans = KMeans(n_clusters = 2, random_state = 42)
ykmeans = kmeans.fit_predict(Xscaled)

print("cr")
print(classification_report(y, ykmeans))
print("cm")
print(confusion_matrix(y, ykmeans))

pca = PCA(n_components = 2)
xpca = pca.fit_transform(Xscaled)

df = pd.DataFrame(data = xpca, columns = ["pc1", "pc2"])
df["cluster"] = ykmeans
df["truelabel"] = y

plt.figure(figsize=(12,8))
sns.scatterplot(data=df, x="pc1", y="pc2", hue="cluster", palette="Set1", s=100, edgecolor="black", alpha=0.7)
plt.title("Kmeans clustering dataset")
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.legend(title="cluster")
plt.show()

plt.figure(figsize=(12,8))
sns.scatterplot(data=df, x="pc1", y="pc2", hue="truelabel", palette="coolwarm", s=100, edgecolor="black", alpha=0.7)
plt.title("Kmeans clustering true labels")
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.legend(title="truelabel")
plt.show()

plt.figure(figsize=(12,8))
sns.scatterplot(data=df, x="pc1", y="pc2", hue="cluster", palette="Set1", s=100, edgecolor="black", alpha=0.7)
centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], s=200, c = "red", marker = "X", label = "centroids")
plt.title("with centers")
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.legend(title="cluster")
plt.show()



