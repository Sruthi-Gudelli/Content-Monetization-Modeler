import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

data_cleaned = pd.read_csv('youtube_revenue_cleaned.csv')
data_cleaned.drop('Unnamed: 0', axis=1, inplace=True)
X = data_cleaned.drop(['ad_revenue_usd'], axis=1)
y = data_cleaned['ad_revenue_usd']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

with open("youtube_data_scaling.pkl", 'rb') as f:
    scaler = pickle.load(f)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open("youtube_revenue_model.pkl", 'rb') as f:
    li_model = pickle.load(f)

with open("youtube_revenue_model_dt.pkl", 'rb') as f:
    dt_model = pickle.load(f)

RF_model = joblib.load('youtube_revenue_model_RF.pkl')

with open("youtube_revenue_model_GB.pkl", 'rb') as f:
    GB_model = pickle.load(f)

with open("youtube_revenue_model_XGB.pkl", 'rb') as f:
    XGB_model = pickle.load(f)


def home_page():
    st.title("Content Monetization Modeler")
    st.image('content_monetization_image.jpg')
    st.write("\n As creators and media companies depend more heavily on YouTube for steady income, understanding ad revenue becomes a strategic necessity. Accurate revenue predictions help teams plan budgets, allocate resources, and decide which content formats are most profitable. They also guide upload frequency, audience targeting, and long term growth strategies. In a competitive digital landscape, forecasting earnings isn't optional anymore, it's a core part of sustainable content planning.")
    st.write("So to predict YouTube ad revenue for individual videos and understand which is the driving feature for revenue, 5 regression models were trained. In that one model which is more accurate was choosed for prediction. ")
    st.write("The detailed comparision of 5 models is given in 'Models Comparision' page")
    st.write("To predict the ad revenue of your video go to the 'Check Your Revenue' page")
    

def detailed_comparision():
    st.title('Comparision')
    st.subheader('Navigate through each model given below to understand how accurate the model is ')
    st.write('\n')


    def visualise(model, model_name):
        #col1, col2 = st.columns(2, gap='small')

        y_test_pred = model.predict(X_test_scaled)
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_test_pred, alpha=0.6, color="purple", edgecolors="w")
    

        ideal_line = [min(y_test), max(y_test)]
        ax.plot(ideal_line, ideal_line, color="darkorange", linestyle="--", lw=2, label="Ideal Line")
    
        ax.set_title(f"{model_name}: Actual vs Predicted Plot\n")
        ax.set_xlabel("\nActual Values")
        ax.set_ylabel("Predicted Values\n")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        col1, col2, col3 = st.columns(3, gap='small')
        col1.metric('R2 Score:', value=r2_score(y_test, y_test_pred))
        col2.metric('MAE:', value=mean_absolute_error(y_test, y_test_pred))
        col3.metric('RMSE:', value=root_mean_squared_error(y_test, y_test_pred))

    
    model_names = ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"]

    selected_model = st.segmented_control(
    label="Select Model",
    options=model_names,
    default="Linear Regression", 
    selection_mode="single"
    )
    
    if selected_model == "Linear Regression":
        visualise(li_model, selected_model)
    
    if selected_model == "Decision Tree":
        visualise(dt_model, selected_model)
    
    if selected_model == "Random Forest":
        visualise(RF_model, selected_model)

    if selected_model == "Gradient Boosting":
        visualise(GB_model, selected_model)

    if selected_model == "XGBoost":
        visualise(XGB_model, selected_model)
    
    metrics_data = {
        'models':model_names,
        'R2_Score':[1.0, 0.98, 0.99, 0.99, 0.99],
        'MAE':[0.0, 5.23, 0.57, 0.70, 0.40],
        'RMSE':[0.0, 6.41, 0.73, 0.89, 0.51]
    }
    metrics_df = pd.DataFrame(metrics_data).set_index('models')
    st.write('\n')
    st.subheader("📈 Metric Comparison Charts")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Variance Explained (R2 Score) - Higher is Better**")
        st.bar_chart(metrics_df["R2_Score"])

    with col2:
        st.write("**Error Metrics (MAE vs RMSE) - Lower is Better**")
        st.bar_chart(metrics_df[["MAE", "RMSE"]])
    
    best_model = metrics_df["R2_Score"].idxmax()
    best_r2 = metrics_df["R2_Score"].max()

    st.success(f"🥇 **Best Performing Model:** {best_model} with an R² Score of **{best_r2:.3f}**")

    features_df = pd.DataFrame({
        "Features":X_train.columns,
        "Importance":li_model.coef_
    })
    features_df.sort_values(by='Importance', ascending=False)
    st.subheader("Feature Importance")
    st.write("Let's see which feature is driving the revenue")
    st.table(features_df)
    best_feature = features_df.loc[features_df['Importance'] == features_df['Importance'].max(), 'Features'].values[0]
    st.success(f"**Best Feature:** {best_feature}")


def prediction_page():
    st.header("Check Your Revennue")
    st.write("Enter the following asked values for your video to predict the revenue.")
    col1, col2 = st.columns(2, gap='small')
    views = col1.number_input("Views", min_value=0)
    likes = col2.number_input("Likes", min_value=0)
    comments = col1.number_input("Comments", min_value=0)
    watch_time_minutes = col2.number_input("Watch time in minutes")
    video_length_minutes = col1.number_input("Video length in minutes")
    subscribers = col2.number_input("Subscribers", min_value=0)
    date = col1.date_input("Uploaded Date")
    category = col2.selectbox("Choose Category", ['Education','Entertainment', 'Gaming', 'Lifestyle', 'Music', 'Tech'])
    device = col1.selectbox("Choose Device", ['Desktop', 'Mobile', 'TV', 'Tablet'])
    country = col2.selectbox("Choose Country", ['AU', 'CA', 'DE', 'IN', 'UK', 'US'])
    button = st.button('Predict', type="primary")

    Month = date.strftime("%B")
    Day_of_Week = date.strftime("%A")

    data_columns = list(X_train.columns)

    if button:
        data_df = pd.DataFrame({
        'views':views, 'likes':likes, 'comments':comments, 'watch_time_minutes':watch_time_minutes, 
        'video_length_minutes':video_length_minutes, 'subscribers':subscribers, 'Month':Month,
        'Day_of_Week':Day_of_Week, 'category':category, 'device':device, 'country':country
        }, index=[0])

        dummy_data = pd.get_dummies(data_df)
        final_data = dummy_data.reindex(columns=data_columns, fill_value=0)
        scaled_data = scaler.transform(final_data)
        prediction = li_model.predict(scaled_data)
        st.success(f"Estimated ad revenue: **${prediction[0]}**")


home = st.Page(home_page, title='Home')
comparision = st.Page(detailed_comparision, title='Models Comparision')
predict = st.Page(prediction_page, title='Check Your Revenue')

pg = st.navigation([home, comparision, predict])
pg.run()
