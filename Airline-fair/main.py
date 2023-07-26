import pandas as pd
import numpy as np


#read data 
df=pd.read_excel("Data_Train.xlsx")
#check null if any
null=df.isnull().sum()

#to check row of null
info=df.info()
#Nan was in route
route=df[df.Route.isnull()]
#2 Nan was present in single route row so dropped row
df.dropna(inplace=True)
#check null
nullcheck=df.isnull().sum()
info=df.info()

#Airline Unique features and counts
airline=df.Airline.unique()
airline_counts=df.Airline.value_counts()

#Counts of source and Destination
source_counts=df.Source.value_counts()
destination_count=df.Destination.value_counts()
#dropping Airline Source and Destination
df=pd.get_dummies(df,drop_first=True, columns=["Airline","Source","Destination"])
#replacing total stops feature into integer
stops=df.Total_Stops.unique()
df.replace({'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4},inplace=True)
stops=df.Total_Stops.unique()

#Checked Additional infoo having 79 percent no info so dropped the column
Additional_infoo=df.Additional_Info.value_counts()
df.drop(columns="Additional_Info",axis=1,inplace=True)

#Resettong index
df.reset_index(inplace=True)
#Route was independent so dropped it
df.drop("Route",axis=1, inplace=True)
#extracring Month and day into integer then dropped
df["Journey_Day"]=pd.to_datetime(df["Date_of_Journey"],).dt.day
df["Journey_Month"]=pd.to_datetime(df["Date_of_Journey"]).dt.month
df.drop("Date_of_Journey",axis=1,inplace=True)

#extracing duration into hours and minutes then dropping
# Assuming `df["Duration"]` contains duration strings like ["2h 30m", "1h", "45m", ...]

Duration_in_minutes = []

# Convert each duration in the DataFrame to minutes and append to the Duration_in_minutes list
for duration in df["Duration"]:
    h, m = 0, 0
    for part in duration.split():
        if "h" in part:
            h = int(part[:-1])
        elif "m" in part:
            m = int(part[:-1])
    total_minutes = h * 60 + m
    Duration_in_minutes.append(total_minutes)

# Now you have Duration_in_minutes list with the duration in minutes for each entry in the DataFrame.

df["Duration_in_minutes"]=Duration_in_minutes    
df.drop("Duration",axis=1,inplace=True)

#Extracting arrival and destination hours
df["Dep_Time_hour"]=pd.to_datetime(df["Dep_Time"]).dt.hour
df.drop("Dep_Time",axis=1,inplace=True)
df["Arrival_Time_in_Hour"]=pd.to_datetime(df["Arrival_Time"]).dt.hour    
df.drop(["index","Arrival_Time"],axis=1,inplace=True)

#Saving cleaned data
df.to_excel("Exploratory Data Analysis of Data_Train.xlsx")

#Splitting train and test data
x=df.drop("Price",axis=1)
y=df["Price"]

#Checking important features
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, random_state=42,test_size=0.3)
#from sklearn.ensemble import ExtraTreesRegressor


# feature_imp=ExtraTreesRegressor().fit(x,y)
# feature_imp.feature_importances_
# feature_imp_series=pd.Series(feature_imp.feature_importances_,index=x.columns)

#plot
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12,6))
# feature_imp_series.sort_values().nlargest(26).plot(kind="barh")

#Training
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor().fit(x_train,y_train)
pred=rf.predict(x_test)

#Performance and fluctuation(errors)
score=rf.score(x_train,y_train)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
mean_squared_error(y_test,pred)

np.sqrt(mean_squared_error(y_test,pred))
fluctuation=mean_absolute_error(y_test,pred)

#Saving training model into pickel
import pickle
pickle.dump( rf, open( "Random Forest Airlines Fare Predictions.pkl", "wb" ) )
