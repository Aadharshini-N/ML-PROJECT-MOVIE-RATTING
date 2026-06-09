# ML-PROJECT-MOVIE-RATTING
# =====================================
# CODSOFT TASK 2
# Movie Rating Prediction using ML
# =====================================

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import joblib


# =====================================
# 2️⃣ Load Dataset
# =====================================

print("📂 Loading Dataset...")

file_path = r"C:\Users\Aadha\Downloads\archive (6)\IMDb Movies India.csv"

df = pd.read_csv(file_path, encoding='latin1')

print("✅ Dataset Loaded Successfully")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())


# =====================================
# 3️⃣ Data Cleaning
# =====================================

print("\n🧼 Cleaning & Preprocessing Data...")

# Remove rows with missing important values
df = df.dropna(subset=['Rating', 'Genre', 'Director', 'Actor 1'])


# Clean Duration column
df['Duration'] = df['Duration'].str.replace('min', '', regex=False)
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')


# Clean Votes column
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')


# Fill missing values
df['Duration'] = df['Duration'].fillna(df['Duration'].median())
df['Votes'] = df['Votes'].fillna(df['Votes'].median())


# Convert Genre column to list
df['Genre'] = df['Genre'].apply(lambda x: [g.strip() for g in str(x).split(',')])


# =====================================
# 4️⃣ Encode Genre (One Hot Encoding)
# =====================================

mlb = MultiLabelBinarizer()

genre_encoded = pd.DataFrame(
    mlb.fit_transform(df['Genre']),
    columns=mlb.classes_,
    index=df.index
)

df = pd.concat([df, genre_encoded], axis=1)


# =====================================
# 5️⃣ Encode Director and Actors
# =====================================

encoder = LabelEncoder()

for col in ['Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    df[col] = df[col].astype(str)
    df[col] = encoder.fit_transform(df[col])


print("✅ Data Cleaning Completed")


# =====================================
# 6️⃣ Data Visualization
# =====================================

print("📊 Plotting Data Visualizations...")

# Rating Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Rating'], bins=15, kde=True, color='skyblue')
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Number of Movies')
plt.show()


# Votes Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Votes'], bins=30, color='orange')
plt.title('Distribution of Votes')
plt.xlabel('Votes')
plt.ylabel('Number of Movies')
plt.show()


# Top 10 Directors
top_directors = df['Director'].value_counts().head(10)

plt.figure(figsize=(10,5))
sns.barplot(
    x=top_directors.values,
    y=top_directors.index,
    hue=top_directors.index,
    palette='pastel',
    legend=False
)

plt.title("Top 10 Directors by Number of Movies")
plt.xlabel("Number of Movies")
plt.ylabel("Director")
plt.show()


# Duration Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='Duration', color='lightgreen')
plt.title('Movie Duration Distribution')
plt.xlabel('Duration (minutes)')
plt.show()


# Votes vs Rating
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='Votes', y='Rating', alpha=0.6)
plt.title("Votes vs Rating")
plt.xscale("log")
plt.show()


# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# =====================================
# 7️⃣ Feature Selection
# =====================================

print("\n🎯 Preparing Data for Training...")

features = ['Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Duration', 'Votes']

# Add encoded genres
features = features + list(mlb.classes_)

X = df[features]
y = df['Rating']


# =====================================
# 8️⃣ Train Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training Size:", X_train.shape)
print("Testing Size:", X_test.shape)


# =====================================
# 9️⃣ Train Random Forest Model
# =====================================

print("\n🤖 Training Machine Learning Model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("✅ Model Training Completed")


# =====================================
# 🔟 Model Evaluation
# =====================================

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance")

print("Mean Squared Error:", mse)
print("R2 Score:", r2)


# =====================================
# 11️⃣ Save Model
# =====================================

joblib.dump(model, "movie_rating_model.pkl")

print("\n💾 Model Saved as movie_rating_model.pkl")


# =====================================
# 12️⃣ Example Prediction
# =====================================

print("\n🎬 Example Prediction")

sample_movie = X_test.iloc[0:1]

prediction = model.predict(sample_movie)

print("Predicted Rating:", prediction[0])
