import pandas as pd
import numpy as np
from google.colab import drive

drive.mount("/content/gdrive")  
!pwd

%cd "/content/gdrive/My Drive/Proyecto Sistemas"
!ls

df = pd.read_csv('fertility_Diagnosis.txt')

columns=["Season", "Age", "Childish_desease", "Accident_or_serious_trauma", "Surgical_intervention", "High_fevers_in_past_years", "Alcohol_consumtion", "Smoking_Habit", "Sitting_hours", "Output"]
df = pd.read_csv('fertility_Diagnosis.txt',names=columns)

df_y=df[['Output']]
df_x=df[['Season', 'Age', 'Childish_desease', 'Accident_or_serious_trauma', 'Surgical_intervention', 'High_fevers_in_past_years', 'Alcohol_consumtion', 'Smoking_Habit', 'Sitting_hours']]

from sklearn.model_selection import train_test_split
dfx_train,dfx_test,dfy_train,dfy_test=train_test_split(df_x,df_y,test_size=0.2)
print("""
The training data
""")
print(dfx_train.head())
print(dfy_train.head())
print("""
The test data
""")
print(dfx_test.head())
print(dfy_test.head())

from sklearn.tree import DecisionTreeClassifier
t_classif = DecisionTreeClassifier(max_depth = 6)
t_classif.fit(dfx_train,dfy_train)

print("Tree Classifier Configuration")
print (t_classif)

print("""
Test data results
""")
Test_results = pd.DataFrame(t_classif.predict(dfx_test))
print(Test_results)
print("""
Original data results
""")
print(dfy_test)

from sklearn import preprocessing
label_Test = preprocessing.LabelEncoder()
Rtest = dfy_test.apply(label_Test.fit_transform)

Rresults = Test_results.apply(label_Test.fit_transform)
print("")
from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(Rresults,Rtest))

import matplotlib.pyplot as plt
X = ['N','O']
Val1=Test_results.value_counts()
Val2=dfy_test.value_counts()
x_axis=np.arange(len(X))

plt.bar(x_axis - 0.2, Val1,0.4, label="N")
plt.bar(x_axis + 0.2, Val2,0.4, label="O")

from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=3, n_jobs=-1, random_state=42)
rnd_clf.fit(dfx_train,dfy_train)
Rfpred= pd.DataFrame(rnd_clf.predict(dfx_test))

print(rnd_clf)
print(Rfpred)
print("random forest", accuracy_score(dfy_test, Rfpred))

import matplotlib.pyplot as plt

X = ['N','O']
Val3=Rfpred.value_counts()
Val4=dfy_test.value_counts()
x_axis=np.arange(len(X))

plt.bar(x_axis - 0.2, Val3,0.4, label="N")
plt.bar(x_axis + 0.2, Val4,0.4, label="O")

#User Answers
Ans=[[None]*9]
Nombre = str (input('Whats your name?: '))
print("")
Ans[0][0] = float (input('Season in which the analysis was performed. Winter(-1) , Spring(-0.33) , Summer(0.33) , Fall(1): '))
print("")
Edad = int (input('Age at the time of analysis. (18-36): '))
Ans[0][1] = (Edad-18)/18
print("")
Ans[0][2] = int (input('Childish diseases (ie , chicken pox, measles, mumps, polio). Yes(0), No(1): '))
print("")
Ans[0][3] = int (input('Accident or serious trauma. Yes(0), No(1): '))
print("")
Ans[0][4] = int (input('Surgical intervention. Yes(0), No(1): '))
print("")
Ans[0][5] = int (input('High fevers in the last year. Less than three months ago(-1), More than three months ago(0), No(1): '))
print("")
Ans[0][6] = float (input('Frequency of alcohol consumption. Several times a day(0.2), Every day(0.4), Several times a week(0.6), Once a week(0.8), Hardly ever or never(1): '))
print("")
Ans[0][7] = int (input('Smoking habit. Never(-1), Occasional(0), Daily(1): '))
print("")
Horas = int (input('Number of hours spent sitting per day (1-16): '))
Ans[0][8] = (Horas-1)/15
print("")
print("")
print(Ans)
print("")
probst = t_classif.predict_proba(Ans)
#print("probability of class for query",Ans,probst)
print("")
predt =  t_classif.predict(Ans)
print("""
According to decision tree""")
print(Nombre,", your concentration of sperm is: ",predt," (normal (N), altered (O))")

probsrf = rnd_clf.predict_proba(Ans)
predrf = rnd_clf.predict(Ans) 
print("""
According to random forest""")
print(Nombre,", your concentration of sperm is: ",predrf," (normal (N), altered (O))")