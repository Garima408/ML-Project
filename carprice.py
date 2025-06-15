# Project is old car price prediction based on supervised learning

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder    # used for encoding
from sklearn.preprocessing import StandardScaler # used for data scalling
from sklearn.model_selection import train_test_split # used for split our dataset for training and testing
from sklearn.ensemble import RandomForestRegressor # model of supervised learning
from sklearn.metrics import mean_squared_error , mean_absolute_error 

dataset=pd.read_csv("C:/Users/DELL/Desktop/oldcar price prediction/car_price_prediction_.csv")  # to read csv file
print(dataset.isnull().sum())   # To check missing data in dataset
print(dataset.info())    # To check the datatypes of each column
print(dataset.head(2))  #for data analysis

Brand_la = LabelEncoder()   # to conver str datatypes into number
dataset["Brand"] = Brand_la.fit_transform(dataset["Brand"]).astype(int)    # replace str datatypes to number (Brand)
print(dataset.info())  # To check the datatypes of each column(Brand's datatype has converted into int now)
print(dataset['Fuel Type'].unique())   # to ckeck the all  Fuel types

Fuel_Type_la = LabelEncoder()  
dataset["Fuel Type"] = Fuel_Type_la.fit_transform(dataset["Fuel Type"]).astype(int)    

Transmission_la = LabelEncoder()  
dataset["Transmission"] = Fuel_Type_la.fit_transform(dataset["Transmission"]).astype(int)    

Condition_la = LabelEncoder()  
dataset["Condition"] = Fuel_Type_la.fit_transform(dataset["Condition"]).astype(int)    

Model_la = LabelEncoder()  
dataset["Model"] = Fuel_Type_la.fit_transform(dataset["Model"]).astype(int) 
print(dataset.info())    

print(dataset.head(5))

# I faced problem from here

# divide input and output from dataset
input_data = dataset.iloc[:,:-1]
print(input_data)

output_data = dataset["Price"]
print(output_data)

#scalling our input data
ss = StandardScaler()     
input_data=pd.DataFrame(ss.fit_transform(input_data),columns=input_data.columns)
print(input_data)

#spilit my dataset for training and testing
x_train , x_test , y_train , y_test = train_test_split(input_data,output_data,test_size=0.2,random_state=42) # 0.2(20% data for testing)


#Model testing
rf = RandomForestRegressor(n_estimators = 100)
rf.fit(x_train,y_train)
accuracy = rf.score(x_test,y_test)*100
print("Module accuracy",accuracy)


# for error check
print("mean squared error",mean_squared_error(y_test,rf.predict(x_test)))
print("mean absolute error",mean_absolute_error(y_test,rf.predict(x_test)))

# prediction for new data
print(" Car_Price",rf.predict([[   5   ,   2 , 2009   ,    2.6    ,      0      ,       1  , 223009      ,    0   ,    21 ]]))
'''
new_data = pd.DataFrame([[1,"Tesla",2016,2.3,"Petrol","Manual",114832,"New","Model X"]], columns=x_train.columns)

# encode user's input
new_data["Brand"]= Brand_la.transform(new_data["Brand"])

new_data = pd.DataFrame(ss.transform(new_data),columns=new_data.columns)  # used to scale the dataframe
print(new_data)

rf.predict(new_data)

'''