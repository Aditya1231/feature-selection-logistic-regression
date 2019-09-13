# --------------
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Load the data
#Loading the Spam data from the path variable for the mini challenge
#Target variable is the 57 column i.e spam, non-spam classes 

df = pd.read_csv(path,header=None)
df.head()
# Overview of the data
df.describe()


#Dividing the dataset set in train and test set and apply base logistic model

X = df.iloc[:,:-1]
y = df.iloc[:,-1]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 42)

lr = LogisticRegression(random_state = 42)

lr.fit(X_train,y_train)



# Calculate accuracy , print out the Classification report and Confusion Matrix.

print('Base Models r2:',lr.score(X_test,y_test))
y_pred = lr.predict(X_test)

print('Base model Confusion matrix:',confusion_matrix(y_test,y_pred))

print('Base model Classification report:',classification_report(y_test,y_pred))


# Copy df in new variable df1
df1 = df.copy()

# Remove Correlated features above 0.75 and then apply logistic model

corr_matrix = df1.drop(57,1).corr().abs()
upper = pd.DataFrame(np.triu(corr_matrix,k=1))

to_drop = [col for col in upper.columns if any(upper[col] > 0.75)]

print(to_drop)
df1.drop(to_drop,1,inplace=True)







# Split the new subset of data and fit the logistic model on training data

X = df1.iloc[:,:-1]
y = df1.iloc[:,-1]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 42)


lr = LogisticRegression(random_state = 42)

lr.fit(X_train,y_train)


# Calculate accuracy , print out the Classification report and Confusion Matrix for new data

print('Base Models r2:',lr.score(X_test,y_test))
y_pred = lr.predict(X_test)

print('Base model Confusion matrix:',confusion_matrix(y_test,y_pred))

print('Base model Classification report:',classification_report(y_test,y_pred))

# Apply Chi Square and fit the logistic model on train data use df dataset

nof_list = [20,30,50,55]
high_scorer = 0
n=0


X = df.iloc[:,:-1]
y = df.iloc[:,-1]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 42)

for n in nof_list:
    test = SelectKBest(score_func=chi2,k=n)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 42)

    X_train = test.fit_transform(X_train,y_train)
    X_test = test.transform(X_test)
    lrchi = LogisticRegression(random_state=42)
    lrchi.fit(X_train,y_train)
    chi_score = lrchi.score(X_test,y_test)
    if chi_score > high_scorer:
       high_scorer = chi_score
       nof = n
       model=lrchi

print('Chi nof',nof)
print('Chi score',chi_score)

# Calculate accuracy , print out the Confusion Matrix 
y_pred = model.predict(X_test)
print('Base model Confusion matrix:',confusion_matrix(y_test,y_pred))

print('Base model Classification report:',classification_report(y_test,y_pred))

# Apply Anova and fit the logistic model on train data use df dataset

test = SelectKBest(score_func=f_classif,k=55)
X_train = test.fit_transform(X_train,y_train)

X_test  = test.transform(X_test)

lr1 = LogisticRegression()

lr1.fit(X_train,y_train)




# Calculate accuracy , print out the Confusion Matrix 

print(lr1.score(X_test,y_test))

y_pred = model.predict(X_test)
print('Base model Confusion matrix:',confusion_matrix(y_test,y_pred))

print('Base model Classification report:',classification_report(y_test,y_pred))
# Apply PCA and fit the logistic model on train data use df dataset

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
pca =  PCA(n_components = 55,random_state=0)


X_train_pca = pca.fit_transform(X_train_scaled)

X_test_pca = pca.transform(X_test_scaled)


model1 = LogisticRegression()

model1.fit(X_train_pca,y_train)




   

# Calculate accuracy , print out the Confusion Matrix 
print(model1.score(X_test_pca,y_test))

y_pred = model.predict(X_test_pca)
print('Base model Confusion matrix:',confusion_matrix(y_test,y_pred))

print('Base model Classification report:',classification_report(y_test,y_pred))

# Compare observed value and Predicted value
#print(LogisticRegression.predict(X_test,y_test))



