import os
import pickle

# Define functions to save and load the model
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Define a function to check if the model has been saved
def is_model_saved(filename):
    return os.path.exists(filename)

# Fit the NeuralProphet model
if not is_model_saved("neuralprophet_model.pkl"):
    stocks = modified_stock_data[['Date', 'Close']].copy()
    stocks.columns = ['ds', 'y']
    model_neuralprophet = NeuralProphet()
    model_neuralprophet.fit(stocks)
    
    # Save the trained model
    save_model(model_neuralprophet, "neuralprophet_model.pkl")
else:
    # Load the pre-trained model
    model_neuralprophet = load_model("neuralprophet_model.pkl")

# Make predictions with NeuralProphet
future = model_neuralprophet.make_future_dataframe(stocks, periods=500)
forecast = model_neuralprophet.predict(future)
actual_prediction = model_neuralprophet.predict(stocks)

# Plot predictions from the NeuralProphet model
st.subheader('NeuralProphet Predictions vs Original Chart (Days vs Price)')
fig3 = plt.figure(figsize=(12, 6))
plt.plot(actual_prediction['ds'], actual_prediction['yhat1'], label='Actual Prediction', c='r')
plt.plot(forecast['ds'], forecast['yhat1'], label='Future Prediction', c='b')
plt.plot(stocks['ds'], stocks['y'], label='Original', c='g')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)