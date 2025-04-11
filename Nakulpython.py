import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm
from scipy import stats

# Set plot style
sns.set(style="whitegrid")

# Load the data
path = r"C:\Users\Nidhi Rana\Downloads\Death_rates_for_suicide__by_sex__race__Hispanic_origin__and_age__United_States.csv"
df = pd.read_csv(path)

# Clean column names
df.columns = [col.strip() for col in df.columns]

# Drop rows with missing estimates
df = df.dropna(subset=["ESTIMATE"])

# Convert YEAR to integer safely
df["YEAR"] = pd.to_numeric(df["YEAR"], errors='coerce')
df = df.dropna(subset=["YEAR"])
df["YEAR"] = df["YEAR"].astype(int)

# Strip whitespace from text columns
text_columns = df.select_dtypes(include='object').columns
df[text_columns] = df[text_columns].apply(lambda x: x.str.strip())

# Show dataset info
print("Dataset shape:", df.shape)
print("Missing values per column:\n", df.isnull().sum())

# Keep latest year
latest_year = df["YEAR"].max()

# Show basic dataset info
print("=== Dataset Info ===")
print("Dataset shape:", df.shape)
print("\nMissing values per column:\n", df.isnull().sum())
print("\nData Types:\n")
df.info()
print("\nDescriptive Statistics ")
print(df.describe())

# Compute basic statistics for 'ESTIMATE'
estimate_col = df["ESTIMATE"]
mean_val = estimate_col.mean()
median_val = estimate_col.median()
mode_val = estimate_col.mode().values

print(f"\nMean of Suicide Rate (ESTIMATE): {mean_val:.2f}")
print(f"Median of Suicide Rate (ESTIMATE): {median_val:.2f}")
print(f"Mode(s) of Suicide Rate (ESTIMATE): {mode_val}")


# 1. Histogram - Suicide Rate Distribution

plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(df["ESTIMATE"], bins=30, edgecolor='black')
norm = plt.Normalize(bins.min(), bins.max())
cmap = plt.cm.rainbow
for patch, bin_left in zip(patches, bins[:-1]):
    patch.set_facecolor(cmap(norm(bin_left)))
plt.title("Distribution of Suicide Rates", fontsize=14, weight='bold')
plt.xlabel("Suicide Rate")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 2. Bar Graph - Average Suicide Rate by Sex
avg_sex = df[df["STUB_LABEL"].isin(["Male", "Female"])].groupby("STUB_LABEL")["ESTIMATE"].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(data=avg_sex, x="STUB_LABEL", y="ESTIMATE", hue="STUB_LABEL", palette="pastel", legend=False)
plt.title("Average Suicide Rate by Sex", fontsize=14, weight='bold')
plt.xlabel("Sex")
plt.ylabel("Average Suicide Rate")
plt.tight_layout()
plt.show()

# 3. Z-Test: Compare Male vs Female Suicide Rates
male_rates = df[df["STUB_LABEL"] == "Male"]["ESTIMATE"]
female_rates = df[df["STUB_LABEL"] == "Female"]["ESTIMATE"]
z_stat, p_val = stats.ttest_ind(male_rates, female_rates, equal_var=False)
print("Z-Test / T-Test between Male and Female Suicide Rates")
print(f"Z-statistic (t-statistic): {z_stat:.4f}")
print(f"P-value: {p_val:.4f}")

# 4. Donut Chart - Suicide Rate by Age Group

age_df = df[(df["YEAR"] == latest_year) & (df["AGE"] != "All ages")]
age_grouped = age_df.groupby("AGE")["ESTIMATE"].sum().sort_values(ascending=False)
top_groups = age_grouped.head(6)
other_sum = age_grouped.iloc[6:].sum()
final_data = pd.concat([top_groups, pd.Series({"Other": other_sum})])
colors = plt.cm.rainbow(np.linspace(0, 1, len(final_data)))

plt.figure(figsize=(10, 10))
total = final_data.sum()
plt.pie(
    final_data,
    labels=[f"{label}\n{value:.0f} ({(value / total) * 100:.1f}%)" for label, value in zip(final_data.index, final_data.values)],
    startangle=90,
    colors=colors,
    textprops={'fontsize': 11, 'weight': 'bold'},
    pctdistance=0.7,
    wedgeprops=dict(width=0.4, edgecolor='white')  # Donut style
)
plt.title(f"Suicide Rate Distribution by Age Group ({latest_year})", fontsize=15, weight='bold')
plt.axis('equal')
plt.tight_layout()
plt.show()

# 5. Heatmap - Suicide Rates by Year and Sex
sex_df = df[df["STUB_LABEL"].isin(["Male", "Female"])]
pivot_sex = sex_df.pivot_table(values="ESTIMATE", index="YEAR", columns="STUB_LABEL")
plt.figure(figsize=(14, 6))
sns.heatmap(
    pivot_sex,
    annot=True,
    fmt=".1f",
    cmap="viridis",
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"shrink": 0.8, "label": "Suicide Rate"},
    annot_kws={"fontsize": 10}
)
plt.title("Suicide Rates by Year and Sex", fontsize=16, weight='bold')
plt.xlabel("Sex", fontsize=12)
plt.ylabel("Year", fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.show()
