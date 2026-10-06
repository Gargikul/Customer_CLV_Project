import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import shap

print("All libraries loaded ✓")

# PHASE 1 — LOAD DATA

# %%
df_09 = pd.read_excel('/Users/gargik/Documents/Customer Churn Project/Data/online_retail_II.xlsx', sheet_name='Year 2009-2010')
df_10 = pd.read_excel('/Users/gargik/Documents/Customer Churn Project/Data/online_retail_II.xlsx', sheet_name='Year 2010-2011')

df = pd.concat([df_09, df_10], ignore_index=True)

print(f"Total rows: {df.shape[0]:,}")
print(f"Total columns: {df.shape[1]}")
print(f"\nColumn names:\n{df.columns.tolist()}")

# %%
print("=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Data Types ===")
print(df.dtypes)

print("\n=== Basic Stats ===")
print(f"Date range: {df['InvoiceDate'].min()} → {df['InvoiceDate'].max()}")
print(f"Unique customers: {df['Customer ID'].nunique():,}")
print(f"Unique products: {df['StockCode'].nunique():,}")
print(f"Unique countries: {df['Country'].nunique()}")

# %%
print("=== Negative Quantities (returns) ===")
print(f"Count: {(df['Quantity'] < 0).sum():,}")

print("\n=== Negative Prices ===")
print(f"Count: {(df['Price'] <= 0).sum():,}")

print("\n=== Missing Customer IDs ===")
print(f"Count: {df['Customer ID'].isnull().sum():,}")

print("\n=== Sample cancelled invoices (start with C) ===")
print(df[df['Invoice'].astype(str).str.startswith('C')].head(3))

# PHASE 2 — DATA CLEANING

# %%
# 1. Drop missing Customer IDs
df_clean = df.dropna(subset=['Customer ID'])

# 2. Remove cancelled invoices (start with C)
df_clean = df_clean[~df_clean['Invoice'].astype(str).str.startswith('C')]

# 3. Remove negative or zero quantities
df_clean = df_clean[df_clean['Quantity'] > 0]

# 4. Remove negative or zero prices
df_clean = df_clean[df_clean['Price'] > 0]

# 5. Fix Customer ID type
df_clean['Customer ID'] = df_clean['Customer ID'].astype(int).astype(str)

# 6. Create revenue column
df_clean['Revenue'] = df_clean['Quantity'] * df_clean['Price']

print(f"Rows before cleaning: {df.shape[0]:,}")
print(f"Rows after cleaning:  {df_clean.shape[0]:,}")
print(f"Rows removed:         {df.shape[0] - df_clean.shape[0]:,}")
print(f"\nUnique customers after cleaning: {df_clean['Customer ID'].nunique():,}")

# PHASE 2 — RFM FEATURE ENGINEERING

# %%
snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
print(f"Snapshot date: {snapshot_date}")

rfm = df_clean.groupby('Customer ID').agg(
    last_purchase   = ('InvoiceDate', 'max'),
    frequency       = ('Invoice', 'nunique'),
    monetary        = ('Revenue', 'sum')
).reset_index()

rfm['recency'] = (snapshot_date - rfm['last_purchase']).dt.days
rfm = rfm.drop(columns=['last_purchase'])
rfm.columns = ['customer_id', 'frequency', 'monetary', 'recency']

print(f"\nRFM table shape: {rfm.shape}")
print(f"\nSample:")
print(rfm.head())
print(f"\nStats:")
print(rfm[['recency','frequency','monetary']].describe().round(2))

# PHASE 2 — DEFINE CHURN LABEL

# %%
churn_threshold = 90

rfm['churned'] = (rfm['recency'] > churn_threshold).astype(int)

print(f"Churn threshold: {churn_threshold} days")
print(f"\nChurn distribution:")
print(rfm['churned'].value_counts())
print(f"\nChurn rate: {rfm['churned'].mean()*100:.1f}%")

# %%
rfm.to_csv('/Users/gargik/Documents/Customer Churn Project/Data/customer_features.csv', index=False)
print("Saved → Data/customer_features.csv ✓")
print(f"\nFinal shape: {rfm.shape}")
print(rfm.head())

# PHASE 3 — MODELING

# %%
rfm = pd.read_csv('/Users/gargik/Documents/Customer Churn Project/Data/customer_features.csv')

# Add clean features — no leakage
rfm['avg_order_value'] = rfm['monetary'] / rfm['frequency']

X = rfm[['frequency', 'monetary', 'avg_order_value']]
y = rfm['churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]:,} customers")
print(f"Test set:     {X_test.shape[0]:,} customers")

# %%
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

model.fit(X_train, y_train)
print("Model trained ✓")

# %%
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(f"True Negatives  (correctly predicted active):  {cm[0][0]}")
print(f"False Positives (active predicted as churned): {cm[0][1]}")
print(f"False Negatives (churned missed by model):     {cm[1][0]}")
print(f"True Positives  (correctly predicted churned): {cm[1][1]}")

# %%
explainer   = shap.Explainer(model)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)

# %%
rfm['churn_probability'] = model.predict_proba(X)[:, 1]
rfm['churn_predicted']   = model.predict(X)

rfm.to_csv('/Users/gargik/Documents/Customer Churn Project/Data/predictions.csv', index=False)
print("Saved → Data/predictions.csv ✓")
print(rfm[['customer_id','frequency','monetary','churn_probability','churned']].head(10))

# ============================================================
# PHASE 4 — DASHBOARD
# ============================================================

# %%
df = pd.read_csv('/Users/gargik/Documents/Customer Churn Project/Data/predictions.csv')

# Convert GBP to USD
GBP_TO_USD = 1.27
df['monetary']        = df['monetary'] * GBP_TO_USD
df['avg_order_value'] = df['avg_order_value'] * GBP_TO_USD

# Create risk segments
def risk_segment(prob):
    if prob >= 0.7:
        return 'High Risk'
    elif prob >= 0.4:
        return 'Medium Risk'
    else:
        return 'Low Risk'

df['risk_segment'] = df['churn_probability'].apply(risk_segment)

print("Currency converted to USD ✓")
print(f"\nRisk segment breakdown:")
print(df['risk_segment'].value_counts())

# %%
# CHART 1 — Customer Risk Segments
colors   = {'High Risk': '#E74C3C', 'Medium Risk': '#F39C12', 'Low Risk': '#2ECC71'}
segments = df['risk_segment'].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(segments.index, segments.values,
              color=[colors[s] for s in segments.index],
              edgecolor='white', linewidth=0.5)

for bar, val in zip(bars, segments.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title('Customer Churn Risk Segments', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Risk Segment', fontsize=11)
ax.set_ylabel('Number of Customers', fontsize=11)
ax.set_ylim(0, segments.max() * 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/Users/gargik/Documents/Customer Churn Project/Data/chart1_segments.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved chart 1 ✓")

# %%
# CHART 2 — Top 10 High-Risk Customers
top_atrisk = df[df['risk_segment'] == 'High Risk'].nlargest(10, 'monetary')

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top_atrisk['customer_id'].astype(str),
               top_atrisk['monetary'],
               color='#E74C3C', edgecolor='white', linewidth=0.5)

for bar, val in zip(bars, top_atrisk['monetary']):
    ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
            f'${val:,.0f}', va='center', fontsize=10)

ax.set_title('Top 10 High-Risk Customers by Revenue', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Total Revenue ($)', fontsize=11)
ax.set_ylabel('Customer ID', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/Users/gargik/Documents/Customer Churn Project/Data/chart2_atrisk.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved chart 2 ✓")

# %%
# CHART 3 — Revenue Impact
high_risk        = df[df['risk_segment'] == 'High Risk']
at_risk_revenue  = high_risk['monetary'].sum()
retention_rate   = 0.20
discount         = 0.15
retained_revenue = at_risk_revenue * retention_rate
discount_cost    = retained_revenue * discount
net_saved        = retained_revenue - discount_cost

print("=== Business Impact Summary ===")
print(f"High-risk customers:          {len(high_risk):,}")
print(f"Total at-risk revenue:        ${at_risk_revenue:,.0f}")
print(f"Estimated retained (20%):     ${retained_revenue:,.0f}")
print(f"Discount cost (15%):          ${discount_cost:,.0f}")
print(f"Net revenue saved:            ${net_saved:,.0f}")

labels = ['At-Risk Revenue', 'Retained Revenue', 'Net Saved\nAfter Discount']
values = [at_risk_revenue, retained_revenue, net_saved]
chart_colors = ['#E74C3C', '#F39C12', '#2ECC71']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, values, color=chart_colors, edgecolor='white', linewidth=0.5)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
            f'${val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_title('Revenue Impact of Churn Intervention', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Revenue ($)', fontsize=11)
ax.set_ylim(0, max(values) * 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/Users/gargik/Documents/Customer Churn Project/Data/chart3_impact.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved chart 3 ✓")
print("\nAll done! Project complete")

