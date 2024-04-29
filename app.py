import numpy as np
from flask import Flask, request, jsonify, render_template,redirect,session
import pickle


l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze']



disease=['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
       'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
       'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
       'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
       'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
       'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
       'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
       'Osteoarthristis', 'Arthritis',
       '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
       'Urinary tract infection', 'Psoriasis', 'Impetigo']





flask_app = Flask(__name__)
flask_app.secret_key = "mlmodel1"

filename = 'THealthModel.pkl'
with open(filename, 'rb') as file:
    loaded_mod = pickle.load(file)

loadClf3 = loaded_mod[0]
loadClf4 = loaded_mod[1]
loadKnn = loaded_mod[2]
loadGnb = loaded_mod[3]
loadSvc = loaded_mod[4]

@flask_app.route("/")
def Home():
    return render_template("index.html")

#model 1 decision tree:
@flask_app.route("/DT")
def DT():

    if "inputtest" in session:
         inputtest = session["inputtest"]
         
     
    prediction = loadClf3.predict(inputtest)
    h='no'
    for a in range(0,len(disease)):
        if(prediction == a):
                h='yes'
                break
    if (h=='yes'):
        pre1 = (disease[a])
    else:
        pre1 ="Not Found"

    session.pop("inputest", None)
    return render_template("index.html", prediction_text0 = "The disease predicted by Decision Tree is {}".format(pre1))

#model 2 random forest
@flask_app.route("/RF")
def RF():

    if "inputtest" in session:
         inputtest = session["inputtest"]
         
     
    prediction = loadClf4.predict(inputtest)
    h='no'
    for a in range(0,len(disease)):
        if(prediction == a):
                h='yes'
                break
    if (h=='yes'):
        pre1 = (disease[a])
    else:
        pre1 ="Not Found"

    session.pop("inputest", None)
    return render_template("index.html", prediction_text1 = "The disease predicted by Random Forest is {}".format(pre1))


# model 3 knn
@flask_app.route("/KNN")
def KNN():

    if "inputtest" in session:
         inputtest = session["inputtest"]
         
     
    prediction = loadKnn.predict(inputtest)
    h='no'
    for a in range(0,len(disease)):
        if(prediction == a):
                h='yes'
                break
    if (h=='yes'):
        pre1 = (disease[a])
    else:
        pre1 ="Not Found"

    session.pop("inputest", None)
    return render_template("index.html", prediction_text2 = "The disease predicted by KNN is {}".format(pre1))

#model 4 naive bayes
@flask_app.route("/NB")
def NB():

    if "inputtest" in session:
         inputtest = session["inputtest"]
         
     
    prediction = loadGnb.predict(inputtest)
    h='no'
    for a in range(0,len(disease)):
        if(prediction == a):
                h='yes'
                break
    if (h=='yes'):
        pre1 = (disease[a])
    else:
        pre1 ="Not Found"

    session.pop("inputest", None)
    return render_template("index.html", prediction_text3 = "The disease predicted by Naive Bayes is {}".format(pre1))

#model 5 svm
@flask_app.route("/SVM")
def SVM():

    if "inputtest" in session:
         inputtest = session["inputtest"]
         
     
    prediction = loadSvc.predict(inputtest)
    h='no'
    for a in range(0,len(disease)):
        if(prediction == a):
                h='yes'
                break
    if (h=='yes'):
        pre1 = (disease[a])
    else:
        pre1 ="Not Found"

    session.pop("inputest", None)
    return render_template("index.html", prediction_text4 = "The disease predicted by SVM is {}".format(pre1))

@flask_app.route("/AllModels")
def AllModels():
    if "inputtest" in session:
        inputtest = session["inputtest"]
        
    predictions = []
    models = ["DT", "RF", "KNN", "NB", "SVM"]
    
    for model in models:
        prediction = None
        if model == "DT":
            prediction = loadClf3.predict(inputtest)
        elif model == "RF":
            prediction = loadClf4.predict(inputtest)
        elif model == "KNN":
            prediction = loadKnn.predict(inputtest)
        elif model == "NB":
            prediction = loadGnb.predict(inputtest)
        elif model == "SVM":
            prediction = loadSvc.predict(inputtest)

        h = 'no'
        for a in range(0, len(disease)):
            if prediction == a:
                h = 'yes'
                break
        if h == 'yes':
            pre1 = disease[a]
        else:
            pre1 = "Not Found"

        predictions.append(pre1)

    session.pop("inputest", None)

    # Rendering the template with all predictions
    return render_template("index.html",
                           prediction_text0="The disease predicted by Decision Tree is {}".format(predictions[0]),
                           prediction_text1="The disease predicted by Random Forest is {}".format(predictions[1]),
                           prediction_text2="The disease predicted by KNN is {}".format(predictions[2]),
                           prediction_text3="The disease predicted by Naive Bayes is {}".format(predictions[3]),
                           prediction_text4="The disease predicted by SVM is {}".format(predictions[4]))


@flask_app.route("/predict", methods = ["GET","POST"])
def predict():

    if request.method == 'POST':
         Algo = request.form.get('es1')
         

    Rawfeatures = [str(x) for x in request.form.values()]

    l2=[]
    for i in range(0,len(l1)):
        l2.append(0)


    for k in range(0,len(l1)):
            for z in Rawfeatures:
                if(z==l1[k]):
                    l2[k]=1
    inputtest = [l2]

    session['inputtest'] = inputtest

    if(Algo == "DecisionTree"):
         return redirect('/DT')
    
    elif(Algo == "Randomforest"):
         return redirect('/RF')

    elif(Algo == "KNN"):
         return redirect('/KNN')

    elif(Algo == "NaiveBayes"):
         return redirect('/NB')

    elif(Algo == "SVM"):
         return redirect('/SVM')

    else:

        return render_template("index.html", prediction_text = "check model")

if __name__ == "__main__":
    flask_app.run(debug=True)