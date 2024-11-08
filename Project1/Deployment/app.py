# Importing necessary library
import pandas as pd
import matplotlib, matplotlib.pyplot as plt
matplotlib.use('Agg')
import streamlit as st
st.set_page_config(layout="wide")
import streamlit.components.v1 as components
import shap
import joblib

# User input values (slider panel)
def user_input_features():
    Year = st.sidebar.slider('Year', 2002, 2020, 2005)
    Month = st.sidebar.selectbox("Month: ", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August','September', 'October',
                                'November', 'December'], 2)
    Replication = st.sidebar.selectbox("Replication (Experimental zones)", ['R1', 'R2', 'R3', 'R4', 'R5'])
    N_rate = st.sidebar.slider('N_rate (N fertilizer rate) [kg N ha-1]', 0, 220, 81)
    PP2 = st.sidebar.slider('PP2 (Cumulative Precipitation in last two days before gas sampling) [mm]', float(0), float(100), 18.18)
    PP7 = st.sidebar.slider('PP7 (Cumulative Precipitation in last week before gas sampling) [mm]', float(0), float(255), 38.11)
    Air_T = st.sidebar.slider('Air_T (Daily average air temperature) [°C]', float(0), float(32), 23.93)
    DAF_TD = st.sidebar.slider('DAF_TD (Days after top dressed N fertilizer) [Days]', 0, 700, 365)
    DAF_SD = st.sidebar.slider('DAF_SD (Days after side dressed N fertilizer) [Days]', 0, 700, 50)
    WFPS25cm = st.sidebar.slider('WFPS25cm (Water Filled Pore Space) [Fraction]', 0.02, 1.00, 0.43 )
    NH4 = st.sidebar.slider('NH4 (Ammonium N content in the top 25-cm soil layer) [kg ha-1]', float(1), float(220), 8.78)
    NO3 = st.sidebar.slider('NO3 (Nitrate N content in the top 25-cm soil layer) [kg ha-1]', float(1), float(220), 22.67)
    Clay = st.sidebar.slider('Clay (Clay concentration in the top 25-cm soil layer) [g kg-1]', 50, 250, 150)
    SOM = st.sidebar.slider('SOM (Soil Organic Matter concentration) [%]', float(1), float(5), 3.33)
    Data_Use = st.sidebar.radio("Data_Use (Model phase: training/ testing)", ('Building', 'Testing'), 0)
    Vegetation = st.sidebar.radio("Vegetation: ", ('Corn', 'GLYMX', 'TRIAE'), 0)
    
    # Order for displaying the features as dataframe
    data = {'Year': Year, 'Month': Month, 'Replication': Replication, 'Data_Use': Data_Use, 'N_rate': N_rate, 'PP2': PP2,
            'PP7': PP7, 'Air_T': Air_T, 'DAF_TD': DAF_TD, 'DAF_SD': DAF_SD, 'WFPS25cm': WFPS25cm, 'NH4': NH4, 'NO3': NO3,
            'Clay': Clay, 'SOM': SOM, 'Vegetation': Vegetation}
    features = pd.DataFrame(data, index=[0])

    return features

# Load model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load(open('model/model_xgb_61_final.joblib.compressed', 'rb'))

# Explain model prediction results
def explain_model_prediction(data):
    explainer = shap.Explainer(model)
    shap_values = explainer(data)
    return shap_values

def shap_summary_plot(df):
  shap_values = explain_model_prediction(df)
  st.subheader('Summary Plots (Impact on model output)')
  st.write('It shows contributing the each features to push the model output from the base value. \
            _Features pushing the prediction __higher__ are shown in __red__, \
            those pushing the prediction __lower__ are in __blue__._')
  
  fig2,ax2 = plt.subplots()
  ax2.set_title('Feature Importance')
  shap.plots.waterfall(shap_values[0], max_display=14)
  st.pyplot(fig2)

  fig,ax = plt.subplots()
  ax.set_title('Feature Importance')  
  shap.summary_plot(shap_values, df)
  st.pyplot(fig)

  fig1,ax1 = plt.subplots()
  ax1.set_title('Feature Importance - Batplot')
  shap.summary_plot(shap_values, df, plot_type='bar')
  st.pyplot(fig1)

model = load_model()

st.markdown('''
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
 ''', unsafe_allow_html=True
)

st.markdown('''
  <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #330033;">
    <a class="navbar-brand" href="https://www.linkedin.com/in/kunalkolhe3/" target="_blank">Kunal K</a>
    <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
      <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navbarNav">
      <ul class="navbar-nav">
        <li class="nav-item active">
          <a class="nav-link disabled" href="">Home<span class="sr-only">(current)</span></a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="https://www.linkedin.com/in/kunalkolhe3/">Linkedin</a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="https://github.com/kunalk3/">Github</a>
        </li>
      </ul>
    </div>
  </nav>
''', unsafe_allow_html=True)

st.write("""
# __N2O Prediction App with Explainable AI__
#### Machine learning improves predictions of agricultural nitrous oxide (N2O) emissions from intensively managed cropping systems.
""")
st.write('---')  

## ----------- Sidebar ----------- ##
st.sidebar.subheader('User Menu Selection')
input_type = st.sidebar.selectbox(label='Select choice: Single user data / User dataset', 
              options=['Predict your on single entity'], index=0)

# Choice 1
if input_type == 'Predict your on single entity':
  st.sidebar.subheader('i) Specify your inputs:')
  df = user_input_features()
  
  # Show selected inputs as dataframe
  st.header('A) Selected inputs: ')
  st.write('Below are the input parameters to model which are selected by user.')
  st.dataframe(data = df)
  st.write('---')

  # Data-preprocessing (Features endoding)
  months_dict = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8,
                'September': 9, 'October': 10, 'November': 11, 'December': 12}
  Vegetation_dict = {"Corn": 1, "GLYMX": 2, "TRIAE": 3}
  datause_dict = {"Building": 1, "Testing": 0}
  replication_dict = {"R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5}
  df["Month"] = df["Month"].apply(lambda x: months_dict[x])
  df["Vegetation"] = df["Vegetation"].apply(lambda x: Vegetation_dict[x])
  df["Data_Use"] = df["Data_Use"].apply(lambda x: datause_dict[x])
  df["Replication"] = df["Replication"].apply(lambda x: replication_dict[x])
  
  # Model predictions(XgbRegressor)
  prediction = model.predict(df)
  st.header('B) Prediction of N2O:')
  st.write('Find the predicted value of N2O with user selected. _Press __Predict__ button_.')

  if st.button("Predict"):
    st.subheader('Predicted value :')
    st.write(round(prediction[0],3))
    st.text('Predictions are computed on your inputs and ML model (Xgb).')
    st.write('---')
  
    st.header('C) Explain model prediction results:')
    shap_summary_plot(df)

st.sidebar.info(
  '''
  **Note:** Is application run slow ? _Computation time (sec.) depends on your data complexity with server run._
  ''')

st.markdown("""
  <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
""", unsafe_allow_html=True)
