# EXPERIMENT-2: DEVELOPING A NEURAL NETWORK CLASSIFICATION MODEL

## AIM:

To develop a neural network classification model for the given dataset.

## THEORY:

The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). 
Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network 
model that can classify a given iris flower into one of these three species based on the provided features.

## Neural Network Model:

<img width="1044" height="800" alt="image" src="https://github.com/user-attachments/assets/9219cea5-c291-4307-91ee-743e15867be3" />

## DESIGN STEPS:

**Step-1:** Import necessary libraries.

**Step-2:** Load the dataset "customers.csv"

**Step-3:** Analyse the dataset and drop the rows which has null values.

**Step-4:** Use encoders and change the string datatypes in the dataset.

**Step-5:** Calculate correlation matrix ans plot heatmap and analyse the data.

**Step-6:** Use various visualizations like pairplot,displot,countplot,scatterplot and visualize the data.

**Step-7:** Split the dataset into training and testing data using train_test_split.

**Step-8:** Create a neural network model with 2 hidden layers and output layer with four neurons representing multi-classification.

**Step-9:** Compile and fit the model with the training data

**Step-10:** Validate the model using training data.

**Step-11:** Evaluate the model using confusion matrix.


## PROGRAM:

**Name:** Rahul M R

**Register Number:** 2305003005

```
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt

customer_df = pd.read_csv('customers.csv')
customer_df.columns

customer_df.dtypes
customer_df.shape

customer_df.isnull().sum()
customer_df_cleaned = customer_df.dropna(axis=0)
customer_df_cleaned.isnull().sum()
customer_df_cleaned.dtypes

customer_df_cleaned['Gender'].unique()
customer_df_cleaned['Ever_Married'].unique()
customer_df_cleaned['Graduated'].unique()

categories_list=[['Male', 'Female'],
           ['No', 'Yes'],
           ['No', 'Yes'],
           ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
            'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
           ['Low', 'Average', 'High']
           ]
enc = OrdinalEncoder(categories=categories_list)
customers_1 = customer_df_cleaned.copy()

customers_1[['Gender',
             'Ever_Married',
              'Graduated','Profession',
              'Spending_Score']] = enc.fit_transform(customers_1[['Gender','Ever_Married','Graduated','Profession','Spending_Score']])

le = LabelEncoder()
customers_1['Segmentation'] = le.fit_transform(customers_1['Segmentation'])
customers_1.dtypes

customers_1 = customers_1.drop('ID',axis=1)
customers_1 = customers_1.drop('Var_1',axis=1)
customers_1.dtypes

# Calculate the correlation matrix
corr = customers_1.corr()

# Plot the heatmap
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)

# Plot scatterplot
plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Spending_Score',data=customers_1)

customers_1.describe()
customers_1['Segmentation'].unique()

one_hot_enc = OneHotEncoder()
X = customers_1.drop('Segmentation', axis=1)
y = customers_1['Segmentation']
one_hot_enc.fit(y.values.reshape(-1,1))
y1 = one_hot_enc.transform(y.values.reshape(-1,1))
y1.shape

y1.shape
y1[0]
y[0]

X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.33,
                                               random_state=50)

X_train.shape
scaler_age = MinMaxScaler()
scaler_age.fit(X_train['Age'].values.reshape(-1,1))

X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)

X_train_scaled[:, 2] = scaler_age.transform(X_train['Age'].values.reshape(-1,1)).flatten()
X_test_scaled[:, 2] = scaler_age.transform(X_test['Age'].values.reshape(-1,1)).flatten()

# Creating the model
model = Sequential([
    Dense(units=8,activation='relu',input_shape=[8]),
    Dense(units=16,activation='relu'),
    Dense(units=4,activation='softmax')
])

model.compile(optimizer='adam',
                 loss= 'categorical_crossentropy',
                 metrics=['accuracy'])

model.fit(x=X_train_scaled,y=y_train_one_hot,
             epochs= 2000,
             batch_size= 256,
             validation_data=(X_test_scaled,y_test_one_hot),
             )

metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['loss','val_loss']].plot()
predi = model.predict(X_test_scaled)
predi
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
x_test_predictions.shape
y_test_truevalue = np.argmax(y_test_one_hot,axis=1)
y_test_truevalue.shape

print(confusion_matrix(y_test_truevalue,x_test_predictions))

print(classification_report(y_test_truevalue,x_test_predictions))

# Saving the Model
model.save('customer_classification_model.h5')

# Saving the data ,PICKLE:  Stores as binary file and then converts
with open('customer_data.pickle', 'wb') as fh:
   pickle.dump([X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le], fh)

# Loading the Model
model = load_model('customer_classification_model.h5')

# Loading the data
with open('customer_data.pickle', 'rb') as fh:
   [X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le]=pickle.load(fh)

x_single_prediction = np.argmax(model.predict(X_test_scaled[1:2,:]), axis=1)

print(x_single_prediction)

print(le.inverse_transform(x_single_prediction))
```

## Dataset Information:

<img width="1024" height="120" alt="image" src="https://github.com/user-attachments/assets/84d64b27-fcd7-4480-8b9c-66d11577b6d3" />



## OUTPUT:

### Training Loss, Validation Loss Vs Iteration Plot:

<img width="547" height="413" alt="image" src="https://github.com/user-attachments/assets/40a224dd-a346-46a2-be83-feee6a03025a" />

### Heat Map:

<img width="994" height="692" alt="Screenshot 2025-09-10 230646" src="https://github.com/user-attachments/assets/8099f9a2-7f36-4ecc-b454-652af96595df" />


### Scatter Plot:

<img width="1101" height="631" alt="Screenshot 2025-09-10 230743" src="https://github.com/user-attachments/assets/e3c96c1b-9948-4d56-8cf5-0d2e0df6b475" />


### Epoch:

<img width="1193" height="298" alt="Screenshot 2025-09-10 230926" src="https://github.com/user-attachments/assets/9d5d735a-a30b-40a1-b7a7-fe61412768b0" />


### Confusion Matrix:

<img width="311" height="114" alt="Screenshot 2025-09-10 231717" src="https://github.com/user-attachments/assets/a9c2bc54-0f00-4140-b582-44304fca1133" />


### Classification Report:

<img width="912" height="268" alt="Screenshot 2025-09-10 231727" src="https://github.com/user-attachments/assets/a048e839-a415-4094-a478-6c37ca71ca69" />


### New Sample Data Prediction:

<img width="1029" height="345" alt="Screenshot 2025-09-10 231807" src="https://github.com/user-attachments/assets/53e90e67-258e-463c-9c44-35acc8168475" />


## RESULT

Thus, A neural network classification model is developed for the given dataset.
