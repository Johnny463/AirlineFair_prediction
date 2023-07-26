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
        Air_India = 0
        GoAir = 0
        IndiGo = 0
        Jet_Airways = 0
        Jet_Airways_Business = 0
        Multiple_carriers = 0
        Multiple_carriers_Premium_economy = 0
        SpiceJet = 0
        Trujet = 0
        Vistara = 0
        Vistara_Premium_economy = 0
        
        if selected_airline == 'Air_India':
            Air_India = 1
        elif selected_airline == 'GoAir':
            GoAir = 1
        elif selected_airline == 'IndiGo':
            IndiGo = 1
        elif selected_airline == 'Jet_Airways':
            Jet_Airways = 1
        elif selected_airline == 'Jet_Airways_Business':
            Jet_Airways_Business = 1
        elif selected_airline == 'Multiple_carriers':
            Multiple_carriers = 1
        elif selected_airline == 'Multiple_carriers_Premium_economy':
            Multiple_carriers_Premium_economy = 1
        elif selected_airline == 'SpiceJet':
            SpiceJet = 1
        elif selected_airline == 'Trujet':
            Trujet = 1
        elif selected_airline == 'Vistara':
            Vistara = 1
        elif selected_airline == 'Vistara_Premium_economy':
            Vistara_Premium_economy = 1
# Now you have the value of Air_India set to 1 and the rest of the airline variables set to 0.
# You can use these variables along with other input features for your prediction model.

       
        if Total_Stops=="non-stop":
            Total_Stops_int=0
        if Total_Stops=="1 stop":
            Total_Stops_int=1
        if Total_Stops=="2 stops":
            Total_Stops_int=2
        if Total_Stops=="3 stops":
            Total_Stops_int=3
        if Total_Stops=="4 stops":
            Total_Stops_int=4
          
        # Assuming this code is part of a web application where form data is being handled

            # Get the value of the "Source" field from the form
            
        print("source")   
        Source_Chennai=0
        Source_Delhi=0
        Source_Kolkata=0
        Source_Mumbai=0
    
        # Set the variables based on the selected source
        if Source == "Chennai": Source_Chennai = 1  
        elif Source == "Delhi": Source_Delhi = 1 
        elif Source == "Kolkata":    Source_Kolkata = 1 
        elif Source == "Mumbai": Source_Mumbai = 1
        
        
        # Get the value of the "Destination" field from the form
        
        Destination_Cochin=0
        Destination_Delhi=0
        Destination_Hyderabad=0
        Destination_Kolkata=0
        Destination_New_Delhi=0
        if Destination == "Cochin": Destination_Cochin = 1  
        elif Destination == "Delhi":   Destination_Delhi = 1 
        elif Destination == "Hyderabad": Destination_Hyderabad = 1
        elif Destination == "Kolkata": Destination_Kolkata = 1
        elif Destination == "New Delhi": Destination_New_Delhi = 1
    
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
        
        
        
        
        #Prediction
        prediction=model.predict([[ Total_Stops_int,Air_India, GoAir, IndiGo, Jet_Airways, Jet_Airways_Business,
        Multiple_carriers, Multiple_carriers_Premium_economy, SpiceJet,Trujet, Vistara, Vistara_Premium_economy, Source_Chennai, Source_Delhi, Source_Kolkata, Source_Mumbai,
        Destination_Cochin, Destination_Delhi, Destination_Hyderabad,  Destination_Kolkata, Destination_New_Delhi, Journey_Day,
        Journey_Month, int(Duration), Dep_Time_hour,Arrival_Time_in_Hour]])
        
        #Rounding
        output=round(prediction[0],2)
        if request.is_json:
            
            return jsonify({"price":output})
        #output with prediction
        return render_template('index.html',prediction_text="Your ticket for {} is {}".format(selected_airline,output))
    else:
        return render_template('index.html')
    

        
if __name__=="__main__":
    app.run(debug=True)




      