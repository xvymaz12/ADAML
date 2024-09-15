# BM20A6100 Project work wind turbine failure
# Data summary and visualisation
# Emma Hirvonen, Helmi Toropainen, Jan Vymazal

import pandas as pd

# load data for all turbines
WT2 = pd.read_excel('/content/sample_data/data.xlsx', 'No.2WT')
WT39 = pd.read_excel('/content/sample_data/data.xlsx', 'No.39WT')
WT14 = pd.read_excel('/content/sample_data/data.xlsx', 'No.14WT')
WT3 = pd.read_excel('/content/sample_data/data.xlsx', 'No.3')

# check for missing data
turbines = [WT2, WT39, WT14, WT3]
for turbine in turbines:
  missing_values = turbine.isnull().sum()
  record_count = len(turbine)
  means = turbine.mean()
  print(f"Number of missing values{missing_values}")
  print(f"Number of records {record_count}")
  print(means)

import matplotlib.pyplot as plt

# correlations between variables for turbine 2
corr_matrix_2 = WT2.corr()
plt.imshow(corr_matrix_2, cmap='Blues')

# correlations between variables for turbine 39
corr_matrix_39 = WT39.corr()
plt.imshow(corr_matrix_39, cmap='Blues')

# boxplots for all variables
for column in WT39.columns:
  plt.figure()
  plt.boxplot([WT2[column],WT39[column]], labels=["WT2", "WT39"])
  plt.title(f'Boxplot of attribute {column}')
  plt.grid(True)
  plt.show()

plt.figure()
plt.boxplot(WT2[28])
plt.title(f"Boxplot of 28")
plt.grid(True)
plt.show()

# chosen variables to look into: 5, 11, 12, 15, 18, 27

# plot time series data for chosen variables
picked_columns = [5, 11, 12, 15, 18, 27]
for col in picked_columns:
  plt.figure()
  plt.plot(WT2.iloc[:, col-1], label="WT2", color='blue')
  plt.plot(WT39.iloc[:, col-1], label="WT39", color='green')
  plt.title(f"Time series of attribute {col}")
  plt.xlabel("Time")
  plt.legend()
  plt.grid(True)

# plot time series data for all variables for further inspection
picked_columns = range(27)
for col in picked_columns:
  plt.figure()
  plt.plot(WT2.iloc[:, col], label="WT2", color='blue')
  plt.plot(WT39.iloc[:, col], label="WT39", color='green')
  plt.title(f"Time series of attribute {col+1}")
  plt.xlabel("Time")
  plt.legend()
  plt.grid(True)
  plt.show()

# plot variable 28 from WT2 with variable 27 from WT39
plt.figure()
plt.plot(WT2.iloc[:, 27], label="WT2", color='blue')
plt.plot(WT39.iloc[:, 26], label="WT39", color='green')
plt.title(f"Time series of attribute 28 from WT2 and attribute 27 from WT39")
plt.xlabel("Time")
plt.legend()
plt.grid(True)
plt.show()

# plot variable 27 from WT2 with variable 26 from WT39
plt.figure()
plt.plot(WT2.iloc[:, 26], label="WT2", color='blue')
plt.plot(WT39.iloc[:, 25], label="WT39", color='green')
plt.title(f"Time series of attribute 27 from WT2 and attribute 26 from WT39")
plt.xlabel("Time")
plt.legend()
plt.grid(True)
plt.show()