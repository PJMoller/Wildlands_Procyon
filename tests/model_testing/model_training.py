from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

# loading the processed data
processed_df = pd.read_csv("../../data/processed/processed_merge.csv")


# getting the data ready
X = processed_df.drop(columns=["ticket_num"])
y = processed_df["ticket_num"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# change the model keep the rest, namings too!
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(y_test.values)
print(y_pred)
print(f"MAE: {mae}, MSE: {mse}, R2: {r2}")
# MAE: 32.28782051282052, MSE: 9095.703798717948, R2: 0.9374264233080393 sheesh rly nice scores

# save the model to connect to the website later