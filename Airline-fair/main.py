from flask import Flask,request,render_template,jsonify
import pickle
import pandas as pd



app= Flask(__name__)


#Model Loading
model = pickle.load(open('Random Forest Airlines Fare Predictions.pkl', 'rb'))

#App Routing
@app.route('/',methods=["GET"])

#Rendring  Index
def Home():

    return render_template('index.html')

#Rendring prediction
@app.route('/predict',methods=["POST"])

def predict():
    
    if request.method == 'POST':
    
       ## Airlines=request.form["Airlines"]
        print(request.is_json)
        if request.is_json:
            json_data= request.json
            selected_airline = json_data["Airlines"] 
            Total_Stops=json_data["Total_Stops"]
            Source = json_data["Source"]
            Destination=json_data["Destination"]  
            Date_of_Journey=json_data["Date_of_Journey"]
            Dep_Time=json_data["Dep_Time"]
            Arrival_Time=json_data["Arrival_Time"]
            Duration=json_data["Duration"]
        else:
            print("hi")
            selected_airline = request.form["Airlines"] 
            Total_Stops=request.form["Total_Stops"]
            Source = request.form["Source"]
            Destination=request.form["Destination"]  
            Date_of_Journey=request.form["Date_of_Journey"]
            Dep_Time=request.form["Dep_Time"]
            Arrival_Time=request.form["Arrival_Time"]
            Duration=request.form["Duration"]
         # Get the value of the "Airlines" field from the form
         # Assuming this code is part of a web application where form data is being handled

        # Get the value of the "Airlines" field from the form
      
        # Set variables based on the selected airline
        airline_mapping = {
            'Air_India': 1,
            'GoAir': 2,
            'IndiGo': 3,
            'Jet_Airways': 4,
            'Jet_Airways_Business': 5,
            'Multiple_carriers': 6,
            'Multiple_carriers_Premium_economy': 7,
            'SpiceJet': 8,
            'Trujet': 9,
            'Vistara': 10,
            'Vistara_Premium_economy': 11
        }
        selected_airline = None
        airline_numeric = airline_mapping.get(selected_airline, 0)
# Now your selcted airline set to 1 and the rest of the airline variables set to 0.
# You can use these variables along with other input features for your prediction model.

       #converting categorical value into numeric through mapping
        stops_mapping = {
            'non-stop': 0,
            '1 stop': 1,
            '2 stops': 2,
            '3 stops': 3,
            '4 stops': 4
            }
        total_stops = None
        total_stops_numeric = stops_mapping.get(total_stops, 0)
       

        # Assuming this code is part of a web application where form data is being handled
        # Get the value of the "Source" field from the form
                        
        #converting categorical value into numeric through mapping               
        source_mapping = {
                'Chennai': 1,
                'Delhi': 2,
                'Kolkata': 3,
                'Mumbai': 4 }
        selected_source= None
        source_numeric = source_mapping.get(selected_source, 0)
       
         #converting categorical value into numeric through mapping       
        destination_mapping = {
            'Cochin': 1,
            'Delhi': 2,
            'Hyderabad': 3,
            'Kolkata': 4,
            'New Delhi': 5
        }
        selected_destination = None
        destination_numeric = destination_mapping.get(selected_destination, 0)
        
        categorical_features = [0] * (len(airline_mapping)  + len(source_mapping) + len(destination_mapping))
        categorical_features[len(airline_mapping)  + len(source_mapping) + destination_numeric - 1] = 1 
       ##Get Day and Month
        
        Date_of_Journey=pd.Series(Date_of_Journey)    
        Journey_Day=pd.to_datetime(Date_of_Journey).dt.day   
        Journey_Month=pd.to_datetime(Date_of_Journey).dt.month
        
        
        #into series
        Dep_Time=pd.Series(Dep_Time)
        Dep_Time_hour=pd.to_datetime(Dep_Time).dt.hour +12
 
       #get Arrival time and Destination
       
        Arrival_Time=pd.Series(Arrival_Time)  
        Arrival_Time_in_Hour=pd.to_datetime(Arrival_Time).dt.hour           
        
        
        
        
        
        prediction_input = [total_stops_numeric] + categorical_features + [Journey_Day, Journey_Month, int(Duration), Dep_Time_hour, Arrival_Time_in_Hour]
        
        #Prediction
        prediction = model.predict([prediction_input])
        output=round(prediction[0],2)
        if request.is_json:
            
            return jsonify({"price":output})
        #output with prediction
        return render_template('index.html',prediction_text="Your ticket for {} is {}".format(selected_airline,output))
    else:
        return render_template('index.html')
    

        
if __name__=="__main__":
    app.run(debug=True)




      
