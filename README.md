# MTA Transit Ridership Analysis

This repository contains a comprehensive data analysis of the MTA (Metropolitan Transportation Authority) transit ridership dataset, focusing on ridership patterns during and after the pandemic period.

## Project Overview

This analysis examines New York City's public transit usage across multiple transit modes, including subways, buses, LIRR, Metro-North, and the Staten Island Railway. The dataset tracks both raw ridership numbers and percentage comparisons to pre-pandemic levels, providing insights into recovery patterns over time.

## Features

The analysis includes:

- Null value detection and handling
- Descriptive statistics and data exploration
- Correlation and covariance analysis
- Outlier detection using IQR and Z-score methods
- Skewness analysis
- Statistical significance testing (t-test)
- Time series analysis of ridership trends
- Visualization through heatmaps, bar plots, histograms, and box plots

## Requirements

The code requires the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn

## Usage

1. Clone this repository to your local machine
2. Install required dependencies:
   ```
   pip install pandas numpy matplotlib seaborn
   ```
3. Update the file path in the script to point to your dataset:
   ```python
   df = pd.read_csv('your_file_path.csv')
   ```

## Dataset Description

The dataset contains the following columns:
- `Date`: Date of observation
- Transit mode columns (for each mode):
  - `Total Estimated Ridership`: Raw number of riders
  - `% of Comparable Pre-Pandemic Day`: Percentage compared to similar pre-pandemic day
- Transit modes included:
  - Subways
  - Buses
  - LIRR (Long Island Rail Road)
  - Metro-North
  - Access-A-Ride
  - Bridges and Tunnels
  - Staten Island Railway

## Analysis Components

### Data Cleaning
- Detection and handling of null values in both categorical and numerical data
- Data type conversion and preparation for analysis

### Exploratory Data Analysis
- Statistical summaries of ridership across transit modes
- Temporal aggregation by month and year

### Relationship Analysis
- Correlation analysis between different transit modes
- Covariance calculation for understanding shared variance

### Statistical Testing
- IQR method for outlier detection
- Z-score based outlier identification
- Welch's t-test for comparing time periods (pre-2022 vs. 2022+)

### Visualization
- Correlation heatmaps
- Bar plots of yearly ridership averages
- Histograms of pandemic recovery percentages
- Box plots for identifying outliers
- Scatter plots for relationship analysis
- Line plots for monthly trends

## Results

The analysis generates several key visualizations:
- `correlation_heatmap.png`: Shows relationships between transit modes
- `yearly_ridership_barplot.png`: Displays yearly average ridership by transit mode
- `recovery_histograms.png`: Shows distribution of pandemic recovery percentages
- `ridership_boxplot.png`: Illustrates outliers in ridership data
- `subway_bus_scatter.png`: Explores the relationship between subway and bus ridership
- `monthly_trend.png`: Tracks monthly ridership trends over multiple years


## Acknowledgments

- Metropolitan Transportation Authority (MTA) for providing the dataset
- The pandas, numpy, matplotlib, and seaborn development teams
