import pandas as pd 
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

#Load the dataset 
df = pd.read_csv("data/warranty.csv")
print(df)

#Check columns
print(df.columns)

#Check the missing data 

#Fill the missing values data
df["operating_hours"] = df["operating_hours"].fillna(df["operating_hours"].median())
df["last_service_days_ago"] = df["last_service_days_ago"].fillna(df["last_service_days_ago"].median())
print(df.isnull().sum())

#Info
print(df.info())


#Now split the data 
X = df.drop(columns=["equipment_id" , "dealer_id" , "is_fraud"])
y = df["is_fraud"]

#Train-test-split 
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.25 , random_state=42 )

cols = ["equipment_type" , "region" , "fuel_type" , "claim_severity"]
encoder = OneHotEncoder(handle_unknown="ignore" , sparse_output=False)

X_train_encoded = encoder.fit_transform(X_train[cols])
X_test_encoded = encoder.transform(X_test[cols])
joblib.dump(encoder, "models/encoder.joblib")
# print("One hot encoding technique applies successfully.")


#Now apply TF-IDF to get numerical values for mechanic notes
corpus = df["mechanic_notes"]

tfidf = TfidfVectorizer(max_features=20 , stop_words="english")

X_train_tfidf = tfidf.fit_transform(X_train["mechanic_notes"]).toarray()
X_test_tfidf = tfidf.transform(X_test["mechanic_notes"]).toarray()

joblib.dump(tfidf , "models/tfidf.joblib")
print("Now all the conversion of categorical data into numerical data done.")

#Now extract purely numerical columns 
num_cols = [
    "operating_hours",
    "runtime_hrs",
    "past_claims",
    "last_service_days_ago",
    "claim_amount",
    "part_replacement_cost"
]
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_cols])
X_test_num = scaler.transform(X_test[num_cols])

joblib.dump(scaler, "models/scaler.joblib")
print("Successfully saved the scaler model.")


X_train_final = np.hstack([X_train_encoded , X_train_tfidf , X_train_num])
X_test_final = np.hstack([X_test_encoded , X_test_tfidf , X_test_num])

np.save("data/X_train_final.npy", X_train_final)
np.save("data/X_test_final.npy", X_test_final)
np.save("data/y_train.npy", y_train.values)
np.save("data/y_test.npy", y_test.values)
feature_columns = list(X_train.columns)  # Replace X_train_df with your final training DataFrame variable
joblib.dump(feature_columns, "models/feature_columns.joblib")
print("Saved feature columns successfully!")

print("All the files saved successfully.")
