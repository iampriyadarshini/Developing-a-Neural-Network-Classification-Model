# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation)

### STEP 2: 
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.

### STEP 3: 
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.

### STEP 4: 
Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

### STEP 5: 
Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

### STEP 6: 
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.


## PROGRAM

### Name: Priyadarshini K

### Register Number:212224100046

```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from torch.utils.data import TensorDataset,DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
```
df = pd.read_csv("C:\\Users\\admin\\DL\\customers.csv")
df
```
```
df=df.drop(columns=["ID"])
df
```
```
df.columns
```
```
df.fillna({"Work_Experience":0,"Family_Size":df["Family_Size"].median()},inplace=True)
df
```
```
cat_columns=['Gender','Ever_Married', 'Graduated', 'Profession',
    'Spending_Score', 'Var_1']
for col in cat_columns:
    df[col]=LabelEncoder().fit_transform(df[col])
```
```
lbe=LabelEncoder()
df["Segmentation"]=lbe.fit_transform(df["Segmentation"])
df
```
```
x=df.drop(columns="Segmentation")
y=df["Segmentation"].values
xt,xst,yt,yst=train_test_split(x,y,test_size=0.2,random_state=42)
```
```
scaler=StandardScaler()
xt=scaler.fit_transform(xt)
xst=scaler.transform(xst)
```
```
import torch

xt = torch.FloatTensor(xt)
xst = torch.FloatTensor(xst)
yt = torch.FloatTensor(yt)
yst = torch.FloatTensor(yst)
```
```
tr=TensorDataset(xt,yt)
tst=TensorDataset(xst,yst)
trl=DataLoader(tr,batch_size=16,shuffle=True)
tstl=DataLoader(tst,batch_size=16)
```
```
class classifier1(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.l1=nn.Linear(input_size,32)
        self.l2=nn.Linear(32,16)
        self.l3=nn.Linear(16,8)
        self.l4=nn.Linear(8,4)
    def forward(self,x):
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        x=self.l4(x)
        return x
```
```
model=classifier1(input_size=xt.shape[1])
criterion=nn.CrossEntropyLoss()
op=optim.Adam(model.parameters(),lr=0.001)
```
```
epochs=100
for i in range(epochs):
    for a,b in trl:
        op.zero_grad()
        pred=model(a)
        loss=criterion(pred,b.long())
        loss.backward()
        op.step()
    if i%10==0:
        print(f"Loss:{i}/{epochs}",loss.item())
```
```
pre=[]
act=[]
with torch.no_grad():
    output=model(xst)
    _,predicted=torch.max(output,1)
    pre.extend(predicted.numpy())
    act.extend(yst.numpy())
    print(act,pre)
```
```
accuracy=accuracy_score(act,pre)
conf_matrix=confusion_matrix(act,pre)
cl_report=classification_report(act,pre,target_names=['A','B','C','D'])
print("Accuracy:",accuracy)
print("confusion_matrix:\n",conf_matrix)
print("classification_report:\n",cl_report)
```
```
import seaborn as sns
import matplotlib.pyplot as plt
xl=['A','B','C','D']
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=xl, yticklabels=xl,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
```
```
sample_input = xst[12].clone().unsqueeze(0).detach().type(torch.float32)
model.eval()

with torch.no_grad():
    output = model(sample_input)
    predicted_class_index = torch.argmax(output[0]).item()

predicted_class_label = lbe.inverse_transform(
    [predicted_class_index]
)[0]

actual_class_index = int(yst[12].item())
actual_class_label = lbe.inverse_transform(
    [actual_class_index]
)[0]

print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {actual_class_label}')
```

### Dataset Information
<img width="781" height="330" alt="image" src="https://github.com/user-attachments/assets/e8f932a5-0833-4492-930b-ff8bf61cbcde" />


### OUTPUT
<img width="406" height="312" alt="image" src="https://github.com/user-attachments/assets/df3a508e-0a60-4fd2-89a5-7f0e80bc6f53" />


## Classification Report
<img width="786" height="526" alt="image" src="https://github.com/user-attachments/assets/12da6ddb-710c-4549-88c8-543e17189c33" />


### New Sample Data Prediction
<img width="562" height="92" alt="image" src="https://github.com/user-attachments/assets/a7952836-7e89-4339-8c98-ac0a08da1d7b" />


## RESULT
Thus the python program to develop a neural network classification model is executed successfully
