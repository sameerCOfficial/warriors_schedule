import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('schedule.csv')

print(df.head())

# Keeping relevant columns
df = df[['G', 'Date', 'Opponent', 'HomeAway']]

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%a %b %d %Y')

# Extract useful date features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Encode the 'Opponent' column as numbers
label_encoder = LabelEncoder()
df['OpponentEncoded'] = label_encoder.fit_transform(df['Opponent'])

# Encode 'Home/Away' status as binary, Away is 1, Home is 0
df['HomeAway'] = df['HomeAway'].apply(lambda x: 1 if x == '@' else 0)

print(df.head())

X = df[['Year', 'Month', 'Day', 'HomeAway']]

y = df['OpponentEncoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Function to predict the opponent for a given date
def predict_opponent(date_input):
    # Convert the input date into the same format as the training data
    date_input = pd.to_datetime(date_input, format='%Y-%m-%d')
    
    input_features = {
        'Year': [date_input.year],
        'Month': [date_input.month],
        'Day': [date_input.day],
        'HomeAway': [1]
    }
    
    input_df = pd.DataFrame(input_features)
    
    opponent_encoded = rf_model.predict(input_df)
    
    opponent = label_encoder.inverse_transform(opponent_encoded)
    
    return opponent[0]

input_date = '2025-11-8'
predicted_opponent = predict_opponent(input_date)
print(f"The predicted opponent on {input_date} is: {predicted_opponent}")
