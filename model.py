import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("heart.csv")
print(df)

X=df.drop("target",axis=1)
y=df['target']

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train,y_train)
pickle.dump(model,open("model.pkl","wb"))
print("Heart Disease Model trained and model.pkl saved successfully.")