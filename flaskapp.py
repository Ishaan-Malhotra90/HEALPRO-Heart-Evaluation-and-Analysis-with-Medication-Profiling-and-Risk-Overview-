from flask import Flask, render_template, request, jsonify,redirect
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
app = Flask(__name__,static_url_path='/static')


dtypes = {'column42': 'str', 'column43': 'str', 'column44': 'float64', 'column45': 'str', 'column46': 'int64', 'column47': 'str', 'column48': 'str'}
df = pd.read_csv(r"C:\xampp\htdocs\projectsem3\medicine_dataset.csv", dtype=dtypes, low_memory=False, encoding='latin-1')
df=df.fillna('Data not available')
df=df[['name','substitute0','substitute1','use0','sideEffect0','sideEffect1','sideEffect2','Therapeutic Class','Habit Forming']]
therapeutic_classes_to_keep = ['CARDIAC', 'GASTRO INTESTINAL', 'BLOOD RELATED', 'PAIN ANALGESICS', 'ANTI DIABETIC']
mask = df['Therapeutic Class'].isin(therapeutic_classes_to_keep)
df = df[mask]
df.reset_index(drop=True,inplace=True)

model_file_name = 'heartprediction(Acc=93.53).pkl'
loaded_model = joblib.load(model_file_name)




@app.route('/')
def main():
    return render_template('homepage.html')
@app.route('/homepage')
def homepage():
    return render_template('homepage.html')
@app.route('/contactus')
def contactus():
    return render_template('contactuspage.html')
@app.route('/mainpage')
def mainpage():
    return render_template('mainpage.html')
@app.route('/loginpage')
def loginpage():
    return render_template('loginpage.html')
@app.route('/about')
def about():
    return render_template('aboutus.html')
@app.route('/page1')
def page1():
    return render_template('page1.html')
@app.route('/page2')
def page2():
    return render_template('page2.html')
@app.route('/page3')
def page3():
    return render_template('page3.html')
@app.route('/page4')
def page4():
    return render_template('page4.html')
@app.route('/page5')
def page5():
    return render_template('page5.html')


@app.route('/login', methods=['POST'])
def login():
 
    if request.method == 'POST':
        user1 = request.form.get('user')
        log_pass = request.form.get('pass')
        patientId1 = request.form.get('patientID')
        mail1 = request.form.get('mail')

        df = pd.read_excel("C:\\xampp\\htdocs\\projectsem3\\signupdataset.xlsx")
        user_data = df[df['username'] == user1]

        if user_data.empty:
            return render_template('loginpage.html', error='User does not exist!')

        # Check if the provided password matches the stored password
        if user_data['password'].iloc[0] != log_pass:
            return render_template('mainpage.html', error='Incorrect password!')

        # Check if the provided patient ID matches the stored patient ID
        if user_data['patientid'].iloc[0] != patientId1:
            return render_template('loginpage.html', error='Incorrect patient ID!')

        # Check if the provided email matches the stored email
        if user_data['mailid'].iloc[0] != mail1:
            return render_template('loginpage.html', error='Incorrect email!')

        # If all checks pass, redirect to the main page
        return render_template("mainpage.html")

    else:
        # Render the login page
        return render_template('mainpage.html')


@app.route('/signup', methods=['POST'])
def signup():
    if request.method == 'POST':
        user2=request.form.get('user')
        sign_pass=request.form.get('pass')
        confirm_pass=request.form.get('cpass')
        patientId2=request.form.get('patientID')
        mail2=request.form.get('mail')

        df = pd.read_excel("C:\\xampp\\htdocs\\projectsem3\\signupdataset.xlsx")

        # Check if the username already exists
        if user2 in df['username'].values:
            return render_template('loginpage.html', signup_error='Username already exists!')

        # Check if the passwords match
        if sign_pass != confirm_pass:
            return render_template('loginpage.html', signup_error='Passwords do not match!')

        # Add the new user to the DataFrame
        new_user = {'username': user2, 'password': sign_pass, 'patientid': patientId2, 'mailid': mail2}

        # Convert the new_user dictionary to a DataFrame with one row
        new_user_df = pd.DataFrame([new_user])

        # Concatenate the existing DataFrame with the new_user DataFrame
        df = pd.concat([df, new_user_df], ignore_index=True)

        # Save the updated DataFrame to the Excel file
        df.to_excel("C:\\xampp\\htdocs\\projectsem3\\signupdataset.xlsx", index=False)

        
        return render_template('mainpage.html')

    else:
        # Render the signup page
        return render_template('login.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['medicine']
    result_list = []
    for i, name in enumerate(df['name']):
        if query in name.lower():
            result_list.append(i)

    if not result_list:
        return jsonify({'result': 'No matches found.'})
    else:
        result_df = df.iloc[result_list][:5]
        result_data = []
        for i in result_list:
            result_data.append({
                'name': df.at[i, 'name'],
                'substitute0': df.at[i, 'substitute0'],
                'substitute1': df.at[i, 'substitute1'],
                'use0': df.at[i, 'use0'],
                'sideEffect0': df.at[i, 'sideEffect0'],
                'sideEffect1': df.at[i, 'sideEffect1'],
                'sideEffect2': df.at[i, 'sideEffect2'],
                'Therapeutic Class': df.at[i, 'Therapeutic Class'],
                'Habit Forming': df.at[i, 'Habit Forming']
            })

        return jsonify({'result': result_data})

def predict_heart_disease_risk(age, gender, cigsperday, bpmeds, prevalentstroke, prevalenthyp, diabetes, totcholesterol, sysbp, diabp, bmi, glucose):
    df = pd.read_csv(r"dataset.csv")
    df = df.fillna(df.mean())
    df = pd.DataFrame(df)
    df.rename(columns={'male': 'Gender'}, inplace=True)
    columns_to_drop = ['education']
    df.drop(columns=columns_to_drop, inplace=True)
    X = df.drop(columns=['Gender'])  
    y = df['Gender']  
    smote = SMOTE(sampling_strategy='auto', random_state=42)  
    X_resampled, y_resampled = smote.fit_resample(X, y)
    df = pd.concat([X_resampled, y_resampled], axis=1)
    X = df.drop(columns=['prevalentHyp'])  
    y = df['prevalentHyp']  
    smote = SMOTE(sampling_strategy='auto', random_state=42)  
    X_resampled, y_resampled = smote.fit_resample(X, y)
    df = pd.concat([X_resampled, y_resampled], axis=1)
    X = df.drop(columns=['prevalentStroke'])  
    y = df['prevalentStroke']  
    smote = SMOTE(sampling_strategy='auto', random_state=42)  
    X_resampled, y_resampled = smote.fit_resample(X, y)
    df = pd.concat([X_resampled, y_resampled], axis=1)
    X = df.drop(columns=['TenYearCHD'])  
    y = df['TenYearCHD']   
    smote = SMOTE(sampling_strategy='auto', random_state=42)  
    X_resampled, y_resampled = smote.fit_resample(X, y)
    df = pd.concat([X_resampled, y_resampled], axis=1)
    df=df.sample(frac=1)
    X = df.iloc[:,0:12].values
    y = df.iloc[:, 12].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    input_data = np.array([[age, gender, cigsperday, bpmeds, prevalentstroke, prevalenthyp, diabetes, totcholesterol, sysbp, diabp, bmi, glucose]])
    input_data = scaler.transform(input_data)  # Scale the input data
    probabilities = loaded_model.predict_proba(input_data)
    likelihood_of_heart_disease = ((probabilities[0][0]-0.70)/(0.19))*100
    return likelihood_of_heart_disease
# Define a route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        age = int(request.form.get('age', 0))  # Default value 0 if not provided
        gender = int(request.form.get('gender', 0))
        cigsperday = int(request.form.get('cigsPerDay', 0))
        glucose = int(request.form.get('glucose', 0))
        systolic = int(request.form.get('systolic', 0))
        diastolic = int(request.form.get('diastolic', 0))
        cholesterol = int(request.form.get('cholesterol', 0))
        bpmeds = int(request.form.get('bpMeds', 0))
        stroke = int(request.form.get('stroke', 0))
        diabetes = int(request.form.get('diabetes', 0))
        hypertension = int(request.form.get('hypertension', 0))
        bmi = int(request.form.get('bmi', 0))

        # Call your prediction function
        heart_disease_likelihood = predict_heart_disease_risk(age, gender, cigsperday, bpmeds, stroke, hypertension, diabetes, cholesterol, systolic, diastolic, bmi, glucose)
        
        if heart_disease_likelihood < 20:
            return jsonify({
        'result': f'''You have a very low chance of getting heart disease
                     Approx( {heart_disease_likelihood:f}%)
                     
                     The above result are for this data 
                     Age: {age} Gender: {gender}
                     Bpmeds: {bpmeds} Stroke: {stroke}
                     Hypertension: {hypertension} Cholesterol: {cholesterol}
                     Systolic Bp: {systolic} Diastolic Bp: {diastolic}
                     BMI: {bmi} Glucose: {glucose}
                     Diabetes: {diabetes}'''
    })

        elif 20 <= heart_disease_likelihood < 40:
            return jsonify({
        'result': f'''You have a low chance of getting heart disease
                     Approx( {heart_disease_likelihood:f}%)
                     
                     The above result are for this data 
                     Age: {age} Gender: {gender}
                     Bpmeds: {bpmeds} Stroke: {stroke}
                     Hypertension: {hypertension} Cholesterol: {cholesterol}
                     Systolic Bp: {systolic} Diastolic Bp: {diastolic}
                     BMI: {bmi} Glucose: {glucose}
                     Diabetes: {diabetes}'''
    })

        elif 40 <= heart_disease_likelihood < 60:
            return jsonify({
        'result': f'''You have a moderate chance of getting heart disease
                     Approx( {heart_disease_likelihood:f}%)
                     
                     The above result are for this data 
                     Age: {age} Gender: {gender}
                     Bpmeds: {bpmeds} Stroke: {stroke}
                     Hypertension: {hypertension} Cholesterol: {cholesterol}
                     Systolic Bp: {systolic} Diastolic Bp: {diastolic}
                     BMI: {bmi} Glucose: {glucose}
                     Diabetes: {diabetes}'''
    })

        elif 60 <= heart_disease_likelihood < 80:
            return jsonify({
        'result': f'''You have a high chance of getting heart disease
                     Approx( {heart_disease_likelihood:f}%)
                     
                     The above result are for this data 
                     Age: {age} Gender: {gender}
                     Bpmeds: {bpmeds} Stroke: {stroke}
                     Hypertension: {hypertension} Cholesterol: {cholesterol}
                     Systolic Bp: {systolic} Diastolic Bp: {diastolic}
                     BMI: {bmi} Glucose: {glucose}
                     Diabetes: {diabetes}'''
    })

        else:
            return jsonify({
        'result': f'''You have a very high chance of getting heart disease.
                     Chance of getting heart disease approx( {heart_disease_likelihood:f}%)
                     
                     The above result are for this data 
                     Age: {age} Gender: {gender}
                     Bpmeds: {bpmeds} Stroke: {stroke}
                     Hypertension: {hypertension} Cholesterol: {cholesterol}
                     Systolic Bp: {systolic} Diastolic Bp: {diastolic}
                     BMI: {bmi} Glucose: {glucose}
                     Diabetes: {diabetes}'''
    })

        
        


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)