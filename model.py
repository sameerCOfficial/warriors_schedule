import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data
df = pd.read_csv('schedule.csv')

# Keep relevant columns
df = df[['G', 'Date', 'Opponent', 'HomeAway']]

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%a %b %d %Y')

# Sort data by date to ensure that we can track the sequence of games
df = df.sort_values(by='Date')

# Encode the 'Opponent' column as numbers
label_encoder = LabelEncoder()
df['OpponentEncoded'] = label_encoder.fit_transform(df['Opponent'])

# Encode 'Home/Away' status as binary, Away is 1, Home is 0
df['HomeAway'] = df['HomeAway'].apply(lambda x: 1 if x == '@' else 0)

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Calculate number of previous games before the current opponent
df['GamesSinceOpponent'] = df.groupby('Opponent')['G'].diff().fillna(0) - 1
df['GamesSinceOpponent'] = df['GamesSinceOpponent'].apply(lambda x: max(0, x))


X = df[['G', 'GamesSinceOpponent', 'HomeAway', 'Month', 'Year']]
y = df['OpponentEncoded']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
#cm = confusion_matrix(y_test, y_pred)
#print(cm)

def predict_opponent_by_game(game_number, month, year):
    input_features = {
        'G': [game_number],
        'GamesSinceOpponent': [df[df['G'] == game_number]['GamesSinceOpponent'].values[0]], 
        'HomeAway': [1], 
        'Month': [month],
        'Year': [year]
    }
    
    input_df = pd.DataFrame(input_features)
    
    opponent_encoded = rf_model.predict(input_df)
    
    opponent = label_encoder.inverse_transform(opponent_encoded)
    
    return opponent[0]

input_game_number = 25 
input_month = 12 
input_year = 2025

predicted_opponent = predict_opponent_by_game(input_game_number, input_month, input_year)
print(f"The predicted opponent for game {input_game_number} in {input_month}/{input_year} is: {predicted_opponent}")
