import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import shap
import streamlit.components.v1 as components


st.set_page_config(page_title="Student Depression Predictor", layout="wide", page_icon="🧠")
df = pd.read_csv("student_depression_dataset.csv")


def preprocess(df):
  df = df.copy()
  df.drop(columns=["id", "City", "Profession"], inplace=True)
  categorical_cols = ["Gender", "Sleep Duration", "Dietary Habits", "Degree",
                      "Have you ever had suicidal thoughts ?", "Financial Stress",
                      "Family History of Mental Illness"]
  for col in categorical_cols:
      df[col] = LabelEncoder().fit_transform(df[col].astype(str))
  return df




df = preprocess(df)




page = st.sidebar.selectbox("Choose a page", [
  "01 Introduction", "02 Data Overview", "03 Data Visualization",
  "04 Prediction", "05 Explanation", "06 Hyperparameter Tuning", "07 Conclusion"])




# Variables used across pages
X = df.drop(columns=["Depression"])
y = df["Depression"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Introduction
if page == "01 Introduction":
    st.title("🧠 Student Depression Risk Predictor")
    st.subheader("Supporting student wellness through data and AI.")




    st.markdown("""
    ### What this app does:
    - 🧬 *Analyzes* key lifestyle, academic, and personal stress factors
    - 🤖 *Predicts* risk levels of depression using machine learning models
    - 📊 *Visualizes* trends, feature importance, and provides actionable insights
    """)


    st.markdown("---")


    # Dataset Overview Section
    st.header("📂 About the Dataset")
    st.markdown("""
    This dataset was collected to study the relationship between *student lifestyle habits, academic pressures*,
    and *mental health outcomes*. It includes variables such as:
    - 📚 Academic workload and satisfaction
    - 🏃‍♂️ Physical activity levels
    - 🧑‍🤝‍🧑 Social support and relationships
    - 🧠 Psychological stress indicators
    - 💤 Sleep habits
    - 🍎 Health and nutrition factors


    The goal is to leverage these factors to *predict the likelihood of depression* in students and better understand
    which variables are most impactful.
    """)


    with st.expander("🔍 Dataset Snapshot", expanded=False):
        st.write(f"*Total Records:* {df.shape[0]}  |  *Total Features:* {df.shape[1]}")
        st.dataframe(df.head())




    st.markdown("---")




    # Quick Stats
    st.subheader("📈 Quick Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Students", df.shape[0])
    with col2:
        if 'Depression' in df.columns:
            st.metric("Positive Depression Cases", df['Depression'].sum())
    with col3:
        st.metric("Number of Features", df.shape[1])




    st.markdown("---")




    # Optional deeper dive
    if st.checkbox("Show feature distributions 📊"):
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()


        selected_feature = st.selectbox("Select a feature to explore:", numeric_cols)


        fig, ax = plt.subplots()
        ax.hist(df[selected_feature].dropna(), bins=20, edgecolor='black')
        ax.set_title(f'Distribution of {selected_feature}')
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)




    if st.checkbox("Show missing values 🔎"):
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values detected!")
        else:
            st.error("Missing values detected!")
            st.dataframe(missing)


    st.markdown("---")


    # Fun little interactive part
    st.info("✨ *Fun fact:* Students with strong social support networks are statistically less likely to experience severe depression symptoms!")






# Data Overview
elif page == "02 Data Overview":
   # Import necessary libraries
   from st_aggrid import AgGrid
   from st_aggrid.grid_options_builder import GridOptionsBuilder
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt






   # Display Header
   st.header("📊 Dataset Overview")
   st.markdown("The first 10 lines of the dataframe allow us to see a snapshot of various factors that may predict whether a student reports having depression. The following table details the interpretation of each dataframe column.")


   # Feature Information
   feature_info = {
       "Column Name": [
           "Gender", "Age", "Academic Pressure", "Work Pressure", "CGPA", "Study Satisfaction", "Job Satisfaction", "Sleep Duration",
           "Dietary Habits", "Degree", "Suicidal Thoughts", "Work/Study Hours", "Financial Stress",
           "Family History of Mental Illness", "Depression"
       ],
       "Description": [
           "Student's gender encoded as numerical value ('1' for Male, '0' for Female)",
           "Student's age",
           "Reported academic pressure level on a scale of 0-5, with 5 being the highest",
           "Reported work pressure level on a scale of 0-5, with 5 being the highest",
           "Cumulative Grade Point Average (CGPA) on a scale of 0-10",
           "Reported satisfaction with own study efforts on a scale of 0 to 5, with 5 being the highest",
           "Reported satisfaction with own work efforts on a scale of 0 to 5, with 5 being the highest",
           "Hours of sleep each night",
           "Dietary habits encoded ('0' for Healthy, '1' for Moderate, '3' for Unhealthy)",
           "Type of academic degree pursued, encoded",
           "History of suicidal thoughts encoded ('0' for No, '1' for Yes)",
           "Hours spent studying or working each day",
           "PReported financial stress level on a scale of 0-5, with 5 being the highest",
           "Family history of mental illness encoded ('0' for No, '1' for Yes)",
           "Depression status encoded ('0' for No, '1' for Yes)"
       ]
   }




   feature_df = pd.DataFrame(feature_info)
   grid_options = GridOptionsBuilder.from_dataframe(feature_df)
   grid_options.configure_default_column(width=300)
   grid_options.configure_columns(["Column Name"], width=250)
   grid_options.configure_columns(["Description"], width=450)
   grid_options.configure_columns(["Description"], autoSize=True)
   grid_options.configure_grid_options(domLayout='autoHeight', enableRangeSelection=True, enableSorting=True,
                                       enableFilter=True, pagination=True, paginationPageSize=5)




   grid_options.configure_column("Description", cellStyle={'white-space': 'normal'})
   grid_options.configure_column("Column Name", cellStyle={'textAlign': 'center'})
   grid_options.configure_column("Column Name", headerClass="header-style")
   grid_options.configure_column("Description", headerClass="header-style")
   grid_options.configure_column("Column Name", cellStyle={'backgroundColor': '#dee2ff'})
   grid_options.configure_column("Description", cellStyle={'backgroundColor': '#e9ecff'})




   st.subheader("📝 Feature Description Table")
   AgGrid(feature_df, gridOptions=grid_options.build(), fit_columns_on_grid_load=True)


   st.subheader("📋 First 10 Rows of the Dataset")
   st.dataframe(df.head(10))


   st.subheader("📌 Descriptive Statistics")
   st.markdown("The following table allows us to see important statistics for each value, including the Mean, Minimum, and Maximum")
   st.write(df.describe())


   st.subheader("🔗 Correlation Matrix")
   selected_columns = st.multiselect(
   "Select columns to include in the correlation matrix:",
   options=df.select_dtypes(include=['float64', 'int64']).columns.tolist(),
   default=df.select_dtypes(include=['float64', 'int64']).columns.tolist())
   if len(selected_columns) >= 2:
       fig, ax = plt.subplots(figsize=(12, 6))
       sns.heatmap(df[selected_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
       st.pyplot(fig)
   else:
       st.warning("Please select at least two columns to display the correlation matrix.")


   st.subheader("🔍 Correlation Analysis with Depression")
   st.write(" - Academic Pressure shows a moderate positive correlation with Depression (0.47), indicating that higher academic stress is associated with higher depression symptoms.")
   st.write(" - Work Pressure is strongly positively correlated with Job Satisfaction (0.77), which might indicate that people feeling more work pressure are those more engaged in their jobs (or possibly jobs that are demanding but fulfilling).")
   st.write(" - Financial Stress also shows a moderate correlation with Depression (0.36), suggesting financial concerns are a risk factor for mental health issues.")
   st.write(" - Family History of Mental Illness has a very low correlation with Depression (0.053), possibly due to underreporting or sample characteristics.")
   st.write(" - Age is negatively correlated with Depression (-0.23), implying younger individuals may be more prone to depressive symptoms in this sample.")




# Data Visualization
elif page == "03 Data Visualization":
    st.header("📈 Visualizations")


    import streamlit.components.v1 as components
    looker_studio_url = "https://lookerstudio.google.com/embed/reporting/306064b9-3834-48b9-a630-58921c61b23b/page/lEXIF"
    components.iframe(looker_studio_url, width=800, height=600)




# Prediction
elif page == "04 Prediction":
    st.header("🤖 Depression Risk Prediction")




    # Prepare data
    X = df.drop(columns=["Depression"])
    y = df["Depression"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




    model_name = st.radio("Choose model", [
        "Logistic Regression",
        "Random Forest",
        "K-Nearest Neighbors (KNN)",
        "Linear Regression"
    ])




    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "K-Nearest Neighbors (KNN)":
        n_neighbors = st.slider("Choose number of neighbors (k)", min_value=1, max_value=20, value=5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    else:  # Linear Regression
        model = LinearRegression()




    with mlflow.start_run():
        with st.spinner("Training the model..."):
            model.fit(X_train, y_train)




            if model_name == "Linear Regression":
                y_pred_proba = model.predict(X_test)
                y_pred = (y_pred_proba >= 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)




            acc = accuracy_score(y_test, y_pred)
            from sklearn.metrics import mean_absolute_error, r2_score




            if model_name == "Linear Regression":
                mae = mean_absolute_error(y_test, y_pred_proba)
                r2 = r2_score(y_test, y_pred_proba)




                st.subheader("📏 Regression Metrics (Linear Regression)")
                st.write(f"**Mean Absolute Error (MAE):** {mae:.3f}")
                st.write(f"**R² Score:** {r2:.3f}")




            mlflow.log_param("model", model_name)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, "model")




    st.success(f"✅ Accuracy: {acc:.2f}")




    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)




    if model_name == "Random Forest":
        st.subheader("📊 Feature Importances")
        feat_imp = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.bar_chart(feat_imp.set_index("Feature"))




    # --- Personalized Prediction Section ---
    st.subheader("🎯 Personalized Prediction")
    st.write("Input your information below to predict your depression risk:")




    # Manual inputs for each feature
    age = st.slider("Age (years)", 10, 100, 25)
    gender = st.radio("Gender", ("Male", "Female"))
    cgpa = st.slider("CGPA (0.0 - 4.0)", 0.0, 4.0, 3.0, step=0.01)
    sleep_hours = st.slider("Sleep Hours per Night", 0, 15, 7)
    social_media_use = st.slider("Social Media Use (hours per day)", 0, 10, 3)
    physical_activity = st.slider("Physical Activity (hours per week)", 0, 20, 3)
    financial_stress = st.radio("Financial Stress?", ("Yes", "No"))
    family_history = st.radio("Family History of Depression?", ("Yes", "No"))
    academic_pressure = st.radio("Academic Pressure?", ("Yes", "No"))




    # Build input DataFrame
    user_input = {
        "Age": age,
        "Gender": 1 if gender == "Male" else 0,
        "CGPA": cgpa,
        "Sleep Hours": sleep_hours,
        "Social Media Use": social_media_use,
        "Physical Activity": physical_activity,
        "Financial Stress": 1 if financial_stress == "Yes" else 0,
        "Family History": 1 if family_history == "Yes" else 0,
        "Academic Pressure": 1 if academic_pressure == "Yes" else 0
    }




    user_df = pd.DataFrame([user_input])




    # ✅ Fix: Align user_df columns with training data
    user_df = user_df.reindex(columns=X_train.columns, fill_value=0)




    # Predict button
    if st.button("Predict Depression Risk"):
        user_pred = model.predict(user_df)[0]
        user_pred_prob = model.predict_proba(user_df)[0][1] if hasattr(model, "predict_proba") else 0.0




        if user_pred == 1:
            st.error(f"⚠️ High risk of depression detected (Probability: {user_pred_prob:.2f})")
        else:
            st.success(f"✅ Low risk of depression detected (Probability: {user_pred_prob:.2f})")








# Explanation




elif page == "05 Explanation":
   import shap
   import matplotlib.pyplot as plt
   import streamlit.components.v1 as components
   from sklearn.ensemble import RandomForestClassifier


   st.header("🧐 Model Explainability with SHAP")
   st.markdown("""
   SHAP (SHapley Additive exPlanations) helps us understand how individual features contribute to a prediction.
   """)
   st.markdown("**Note:** In this dataset, `0` indicates no depression, and `1` indicates presence of depression.")




   # Optional: Subsample test set for speed
   X_small = X_test.sample(100, random_state=42)


   # Train model
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)


   # Generate SHAP values
   explainer = shap.Explainer(model, X_small)
   shap_values = explainer(X_small)


   # Detect if this is a multi-class model
   is_multiclass = shap_values.values.ndim == 3




   # 📊 Summary Plot (Feature Importance)
   st.subheader("📊 Summary Plot (Feature Importance)")
   if is_multiclass:
       # For multi-class, pick one class (e.g., class 0)
       selected_class = st.selectbox("Select class for explanation", list(range(shap_values.values.shape[2])))
       shap_values_class = shap.Explanation(
           values=shap_values.values[:, :, selected_class],
           base_values=shap_values.base_values[:, selected_class],
           data=shap_values.data,
           feature_names=shap_values.feature_names
       )
       shap.plots.beeswarm(shap_values_class, max_display=10, show=False)
   else:
       shap.plots.beeswarm(shap_values, max_display=10, show=False)


   st.pyplot(plt.gcf())


   # 🔍 Force Plot (Single Sample)
   st.subheader("🔍 Force Plot (1 Sample)")
   sample_idx = st.slider("Choose an index", 0, len(X_small) - 1, 0)




   if is_multiclass:
       shap_value_sample = shap_values.values[sample_idx, :, selected_class]
       base_value = explainer.expected_value[selected_class]
   else:
       shap_value_sample = shap_values.values[sample_idx]
       base_value = explainer.expected_value


  # Generate and display force plot
   force_plot = shap.plots.force(base_value, shap_value_sample, X_small.iloc[sample_idx])
   components.html(shap.getjs() + force_plot.html(), height=300)




# Hyperparameter Tuning
elif page == "06 Hyperparameter Tuning":
  st.header("🛠️ Hyperparameter Tuning with MLflow")
  from sklearn.model_selection import GridSearchCV
  param_grid = {
      "n_estimators": [50, 100],
      "max_depth": [5, 10],
      "min_samples_split": [2, 5]
  }
  if st.button("Run Tuning with MLflow"):
      with mlflow.start_run():
          grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
          grid_search.fit(X_train, y_train)
          best_model = grid_search.best_estimator_
          y_pred = best_model.predict(X_test)
          acc = accuracy_score(y_test, y_pred)
          mlflow.log_params(grid_search.best_params_)
          mlflow.log_metric("accuracy", acc)
          mlflow.sklearn.log_model(best_model, "best_model")
          st.success(f"Best Accuracy from GridSearch: {acc:.2f}")
          st.write("🔍 Best Parameters:", grid_search.best_params_)




# Conclusion
elif page == "07 Conclusion":
  st.header("🧠 Final Thoughts & Explainable AI")
  st.markdown("""
  ### Summary
  - The model shows promising results using Random Forest, KNN, and Logistic Regression.
  - The model achieved 83% accuracy using a Random Forest classifier tuned with MLflow. Optimal parameters were max_depth=10, min_samples_split=5, and n_estimators=100.
  - Key predictors of depression risk were: suicidal thoughts, academic pressure, financial stress, age, dietary habits, and work/study hours.
  - SHAP values revealed that high academic pressure and history of suicidal thoughts strongly increase predicted depression risk, while healthy dietary habits and higher study satisfaction reduce it.




  """)
  st.balloons()





