
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from CSV file
df = pd.read_csv('MTA_Daily_Ridership.csv')

# 1. Checking and handling null values
print("Checking for null values in the dataset:")
null_counts = df.isnull().sum()
print(null_counts)

# Handle null values
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)

#categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Verify null values are handled
print("\nAfter handling null values:")
print(df.isnull().sum())

# 2. Show statistical data from the dataset
print("\nStatistical summary of the dataset:")
print(df.describe())

# 3. Data Visualization
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Create a correlation heatmap
plt.figure(figsize=(14, 10))
correlation_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Transit Ridership Data')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# Create bar plots for average ridership by year
ridership_cols = ['Subways: Total Estimated Ridership',
                  'Buses: Total Estimated Ridership',
                  'LIRR: Total Estimated Ridership',
                  'Metro-North: Total Estimated Ridership',
                  'Staten Island Railway: Total Estimated Ridership']

yearly_avg = df.groupby('Year')[ridership_cols].mean()

plt.figure(figsize=(12, 8))
yearly_avg.T.plot(kind='bar')
plt.title('Average Daily Ridership by Year')
plt.ylabel('Average Ridership')
plt.xlabel('Transit Mode')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('yearly_ridership_barplot.png')
plt.close()

# Create histograms for percentage recovery distributions
pct_cols = [col for col in df.columns if '% of Comparable' in col]
plt.figure(figsize=(15, 10))

for i, col in enumerate(pct_cols, 1):
    plt.subplot(3, 3, i)
    plt.hist(df[col], bins=20, edgecolor='black')
    plt.title(col.split(':')[0])
    plt.xlabel('Percent of Pre-Pandemic Level')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('recovery_histograms.png')
plt.close()


# 4. Outlier Detection using IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]

    print(f"Outlier Analysis for {column}:")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Outlier percentage: {(len(outliers) / len(df[column])) * 100:.2f}%")
    print(f"IQR: {IQR}")
    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")
    print(f"Min value: {df[column].min()}")
    print(f"Max value: {df[column].max()}")
    print("\n")

    return outliers


# Create box plots for ridership data
plt.figure(figsize=(15, 10))
boxplot = df[ridership_cols].boxplot(vert=False, grid=False)
plt.title('Box Plot of Ridership Data')
plt.tight_layout()
plt.savefig('ridership_boxplot.png')
plt.close()

# Perform IQR analysis on subway ridership
subway_outliers = detect_outliers_iqr(df, 'Subways: Total Estimated Ridership')


# 5. Z-test for outlier detection
def detect_outliers_zscore(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = abs((df[column] - mean) / std)
    outliers = df[z_scores > threshold][column]

    print(f"Z-Score Outlier Analysis for {column}:")
    print(f"Number of outliers (|z| > {threshold}): {len(outliers)}")
    print(f"Outlier percentage: {(len(outliers) / len(df[column])) * 100:.2f}%")
    print(f"Min z-score: {z_scores.min()}")
    print(f"Max z-score: {z_scores.max()}")
    print("\n")

    return outliers


# Apply Z-test to subway ridership
subway_z_outliers = detect_outliers_zscore(df, 'Subways: Total Estimated Ridership')


# 7. Perform t-test between two time periods
def perform_ttest(sample1, sample2):
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)

    # Calculate variances
    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)

    # Calculate sample sizes
    n1 = len(sample1)
    n2 = len(sample2)

    # Calculate t-statistic
    t_stat = (mean1 - mean2) / np.sqrt((var1 / n1) + (var2 / n2))
    df_numerator = ((var1 / n1) + (var2 / n2)) ** 2
    df_denominator = ((var1 / n1) ** 2 / (n1 - 1)) + ((var2 / n2) ** 2 / (n2 - 1))
    df = df_numerator / df_denominator

    return t_stat, df


# Compare pre-2022 vs 2022-and-later subway ridership
early_period = df[df['Year'] < 2022]['Subways: Total Estimated Ridership']
later_period = df[df['Year'] >= 2022]['Subways: Total Estimated Ridership']

t_stat, df_value = perform_ttest(early_period, later_period)

print("\nT-Test Results for Subway Ridership Comparison:")
print(f"Early period (pre-2022) mean: {early_period.mean():.2f}")
print(f"Later period (2022+) mean: {later_period.mean():.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"Approximate degrees of freedom: {df_value:.2f}")
print("For interpretation: If |t-statistic| > 2, there is likely a significant difference")
print(
    f"Result: There is {'likely' if abs(t_stat) > 2 else 'unlikely to be'} a statistically significant difference between the two periods")

# 8. Analyze correlation and covariance
print("\nCovariance Matrix:")
cov_matrix = df[ridership_cols].cov()
print(cov_matrix)

# 9. Feature relationships through scatter plots
plt.figure(figsize=(10, 8))
plt.scatter(df['Subways: Total Estimated Ridership'],
            df['Buses: Total Estimated Ridership'])
plt.title('Relationship Between Subway and Bus Ridership')
plt.xlabel('Subway Ridership')
plt.ylabel('Bus Ridership')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('subway_bus_scatter.png')
plt.close()

# Monthly trend analysis
monthly_avg = df.groupby(['Year', 'Month'])[ridership_cols].mean().reset_index()

plt.figure(figsize=(15, 8))
for year in monthly_avg['Year'].unique():
    year_data = monthly_avg[monthly_avg['Year'] == year]
    plt.plot(year_data['Month'], year_data['Subways: Total Estimated Ridership'],
             marker='o', label=f'Year {year}')

plt.title('Monthly Trend of Subway Ridership')
plt.xlabel('Month')
plt.ylabel('Average Daily Ridership')
plt.xticks(range(1, 13))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('monthly_trend.png')
plt.close()

print("Analysis completed successfully!")