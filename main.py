import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
# ==========================================================
# PART 1: CSV DATASET CREATE AUR LOAD KARNA
# ==========================================================
csv_filename = 'house_data.csv'

if not os.path.exists(csv_filename):
    np.random.seed(42)
    n_samples = 300
    locations = np.random.choice(['Downtown', 'Suburb', 'IT_Park', 'Outskirts'], n_samples)
    sqft = np.random.randint(600, 3500, n_samples)
    bhk = np.random.randint(1, 6, n_samples)
    
    loc_map = {'Downtown': 85, 'IT_Park': 70, 'Suburb': 45, 'Outskirts': 25} 
    base_price = [loc_map[loc] for loc in locations]
    
    # Realistic Price Calculation (in Lakhs)
    price = (sqft * 0.045) + (bhk * 6.0) + base_price + np.random.normal(0, 5, n_samples)
    
    sample_df = pd.DataFrame({
        'Location': locations,
        'Sqft': sqft,
        'Bedrooms': bhk,
        'Price_Lakhs': np.round(price, 2)
    })
    sample_df.to_csv(csv_filename, index=False)
    print(f"[SUCCESS] '{csv_filename}' create ho gayi hai!\n")

# CSV File Load Karna
df = pd.read_csv(csv_filename).dropna()
print("--- Dataset ke Top 5 Rows ---")
print(df.head())

# ==========================================================
# PART 2: DATA PREPROCESSING & MODEL TRAINING
# ==========================================================
# Location text ko numeric (0/1) columns mein convert karna
df_encoded = pd.get_dummies(df, columns=['Location'], drop_first=True)

X = df_encoded.drop('Price_Lakhs', axis=1)
y = df_encoded['Price_Lakhs']

# 80% Training aur 20% Testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model Train karna
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy Check karna
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n--- Model Evaluation Report ---")
print(f"Model Accuracy (R2 Score): {r2 * 100:.2f}%")
print(f"Mean Absolute Error (MAE): +/- ₹ {mae:.2f} Lakhs")

# ==========================================================
# PART 3: ALL-IN-ONE VISUALIZATION DASHBOARD (3 GRAPHS)
# ==========================================================
plt.figure(figsize=(18, 5.5))

# --- Graph 1: Sqft vs Price (Scatter + Trendline) ---
ax1 = plt.subplot(1, 3, 1)
ax1.scatter(df['Sqft'], df['Price_Lakhs'], color='royalblue', alpha=0.6, label='Houses')
m, b = np.polyfit(df['Sqft'], df['Price_Lakhs'], 1)
ax1.plot(df['Sqft'], m * df['Sqft'] + b, color='red', linewidth=2, label='Trendline')
ax1.set_title('1. House Area (Sqft) vs Price', fontsize=12, fontweight='bold')
ax1.set_xlabel('Square Feet (Sqft)')
ax1.set_ylabel('Price (₹ Lakhs)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

# --- Graph 2: Actual vs Predicted Price ---
ax2 = plt.subplot(1, 3, 2)
ax2.scatter(y_test, y_pred, color='seagreen', alpha=0.7, label='Predictions')
min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], color='darkorange', linestyle='--', linewidth=2, label='Perfect Fit')
ax2.set_title('2. Actual vs Predicted Price', fontsize=12, fontweight='bold')
ax2.set_xlabel('Actual Price (₹ Lakhs)')
ax2.set_ylabel('Predicted Price (₹ Lakhs)')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

# --- Graph 3: Location-Wise Average Price (Seaborn Bar Chart) ---
ax3 = plt.subplot(1, 3, 3)
avg_price_df = df.groupby('Location', as_index=False)['Price_Lakhs'].mean().sort_values(by='Price_Lakhs', ascending=False)
sns.barplot(data=avg_price_df, x='Location', y='Price_Lakhs', hue='Location', palette='viridis', legend=False, ax=ax3)

for container in ax3.containers:
    ax3.bar_label(container, fmt='₹ %.1f L', padding=3, fontweight='bold')

ax3.set_title('3. Average Price by Location', fontsize=12, fontweight='bold')
ax3.set_xlabel('Location')
ax3.set_ylabel('Average Price (₹ Lakhs)')
ax3.set_ylim(0, avg_price_df['Price_Lakhs'].max() * 1.15)
ax3.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('real_estate_dashboard.png', dpi=300)
print("\n[INFO] Teeno graphs 'real_estate_dashboard.png' mein save ho gaye hain!")
plt.show()

# ==========================================================
# PART 4: CUSTOM HOUSE PRICE PREDICTION FUNCTION
# ==========================================================
def predict_price(sqft, bedrooms, location):
    input_df = pd.DataFrame(np.zeros((1, len(X.columns))), columns=X.columns)
    input_df['Sqft'] = sqft
    input_df['Bedrooms'] = bedrooms
    
    loc_col = f'Location_{location}'
    if loc_col in input_df.columns:
        input_df[loc_col] = 1
        
    return round(model.predict(input_df)[0], 2)

# Sample Test Prediction
print("\n--- Custom Price Prediction Test ---")
test_sqft = 1800
test_bhk = 3
test_loc = 'IT_Park'

predicted_val = predict_price(test_sqft, test_bhk, test_loc)
print(f"Ghar Details : {test_sqft} Sqft | {test_bhk} BHK | Location: {test_loc}")
print(f"Estimated Price : ₹ {predicted_val} Lakhs")