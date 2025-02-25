import streamlit as st
import requests
import pandas as pd
from streamlit_option_menu import option_menu
import streamlit_lottie as st_lottie
import joblib
import numpy as np
import PIL as Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

st.set_page_config(
    page_title='water_classifier',
    page_icon=':gem:',
    initial_sidebar_state='collapsed'
)

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

model = joblib.load(open("water_classifier",'rb'))

def predict (ph, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity):
    features=np.array([ph, Solids, Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity]).reshape(1,-1)
    prediction=model.predict(features)
    return prediction

# Define function to train and evaluate model
def train_model(model_type):
  # Splitting the data set into : 
  X = df.iloc[:,:-1]
  y = df.iloc[:, -1]
  # Splitting the train and test data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)  
  if model_type == "LR":
   model = LogisticRegression()
  elif model_type == "DT":
    model = DecisionTreeClassifier()
  elif model_type == "SVM":
    model = SVC()
  else:
    st.error("Invalid model type!")
    return None
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  confusion_matrix = confusion_matrix(y_test, y_pred)
  return accuracy, confusion_matrix

#Setting the side bar menu
with st.sidebar:
    choose = option_menu(None,["Home","Classification","About Us","Contact"],
                         icons=['house','kanban','book','person lines fill'],
                         menu_icon="app-indicator",default_index=0,
                         styles={
        "container":{"padding":"5!important","background-color":"0059b5"},
        "icon":{"color":"#575757","font-size":"25px"},
        "nav-link":{"font-size":"16px","text-align":"center","margin":"2px","--hover-color":"#b0b0b0"},
        "nav-link-selected":{"background-color":"f0b3c4"},
    }
    )

#Cases for choices in the side bar menu
if choose == 'Home':
    st.write("Water Quality Prediction")
    st.subheader("Enter the details to predict your water sample's quality")
    # Input of user
    ph = st.number_input("Enter 'PH': ",min_value=0)
    Solids = st.number_input("Enter 'PH': ",min_value=0)
    Chloramines = st.number_input("Enter 'PH': ",min_value=0)
    Sulfate = st.number_input("Enter 'PH': ",min_value=0)
    Conductivity = st.number_input("Enter 'PH': ",min_value=0)
    Organic_carbon = st.number_input("Enter 'PH': ",min_value=0)
    Trihalomethanes = st.number_input("Enter 'PH': ",min_value=0)
    Turbidity = st.number_input("Enter 'PH': ",min_value=0)
    #Predict button
    sample = predict (ph, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity)
    if st.button("Predict"):
        if sample == 0:
            st.write("This water is not potable")
        elif sample == 1:
            st.write("This water is potable")
            st.balloons
    st.image("52.png")
    
    

elif choose == 'Classification':
# Streamlit app layout
    st.title("Classification Model Selection")

# Choose classification method
    model_selection = st.selectbox("Choose Classifier:", ("LR", "DT", "SVM"))

# Train and evaluate model if a choice is made
    if model_selection:
      accuracy, confusion_matrix = train_model(model_selection)
      if accuracy:
       st.write(f"Accuracy: {accuracy:.4f}")
       st.subheader("Confusion Matrix")
       st.write(confusion_matrix)
    else:
       st.write("Error training model!")
    
# elif choose == 'About Us':

# elif choose == 'Contact':
   
df = pd.read_csv ('water.potability.csv.csv')
                         