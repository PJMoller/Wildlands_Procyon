from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
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

# crossval
scoring = {"MAE": make_scorer(mean_absolute_error),"MSE": make_scorer(mean_squared_error),"R2": "r2"}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(model, X, y, cv=kf, scoring=scoring)

print("CV MAE:", cv_results["test_MAE"])
print("CV MSE:", cv_results["test_MSE"])
print("CV R2:", cv_results["test_R2"])
print("Mean CV MAE:", cv_results["test_MAE"].mean())
print("Mean CV MSE:", cv_results["test_MSE"].mean())
print("Mean CV R2:", cv_results["test_R2"].mean())

# save the model to connect to the website later
