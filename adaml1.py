# BM20A6100 Project work wind turbine failure
# Emma Hirvonen, Helmi Toropainen, Jan Vymazal

# Data summary and visualisation

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

# Data pretreatment

WT2 = WT2.drop(WT2.columns[[25,8,11,12,14,18]], axis=1) # We dropped attribute 26, because there is no attribute 26 in WT39 and attributes 9,12,13,15,19 because the values do not vary
WT39 = WT39.drop(WT39.columns[[8,11,12,14,18]], axis=1)
WT2 = WT2.drop(WT2.tail(165).index) # We dropped 165 last rows because WT39 has fewer samples

WT39.columns = WT2.columns

import numpy as np
train = WT2
train_f = WT39

# mean center
mean_train = np.mean(train, 0)
std_train = np.std(train, 0)
median_train = np.median(train, 0)

no_treatment = train
mean_centered = train - mean_train
z_score = (train - mean_train)/std_train
z_robust = (train - median_train)/(np.median(np.absolute(train - median_train)))
center_mad = (train - mean_train)/(np.median(np.absolute(train - median_train)))

models = [no_treatment, mean_centered, z_score, z_robust, center_mad]

treatments = ["Non-treated data", "Mean Centered", "Z-score (STD)",  "Z-score (robust)", "Center and MAD"];

from sklearn.decomposition import PCA
pca = PCA ()
j=0
for model in models:
  pca_data = pca.fit_transform(model)
  pca_data /= np.max(np.abs(pca_data), axis=0)
  pca_data *= np.max(np.abs(pca.components_))
  plt.scatter(pca_data[:,0], pca_data[:,1])
  for i, feature in enumerate(train.columns):
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i],
              head_width=0.01, color="r")
    plt.text(pca.components_[0, i], pca.components_[1, i], feature)
  plt.xlabel("PC1")
  plt.ylabel("PC2")
  plt.grid(True)
  plt.title(f"Biplot for PC1 and PC2 for model {treatments[j]}")
  j=j+1
  plt.show()

z_score_f = (train_f - mean_train)/std_train

j=0
for model in models:
  pca = PCA()
  pca_data = pca.fit_transform(model)

  #Ploting amount of variance each PC explains
  plt.figure(figsize=(8, 6))
  plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
  plt.title(f'Explained Variance by Principal Components for model {treatments[j]}')
  plt.xlabel('Principal Component')
  plt.ylabel('Explained Variance')
  plt.grid(True)
  plt.show()
  #Ploting cumulative sum of explained variance
  plt.figure(figsize=(8, 6))
  plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o')
  plt.title(f'Cumulative Explained Variance by Principal Components for model {treatments[j]}')
  plt.xlabel('Principal Component')
  plt.ylabel('Cumulative Explained Variance')
  plt.grid(True)
  plt.show()
  j+=1

ax = plt.axes()
ax.set_xticklabels(WT2.columns)
plt.boxplot(z_score)
plt.title("Distribution of variables after pretreatment (z-score STD)")
plt.show()

pca = PCA(n_components=5)
pca_data = pca.fit_transform(z_score)
max_score = np.max(np.abs(pca_data), axis=0)
pca_data /= max_score
pca_data *= np.max(np.abs(pca.components_))

pca_projection = np.dot(z_score_f, np.transpose(pca.components_))
#pca_projection /= np.max(np.abs(pca_projection), axis=0)
pca_projection /= max_score
pca_projection *= np.max(np.abs(pca.components_))

for PC1 in range(5):
  for PC2 in range(4-PC1):
    plt.scatter(pca_data[:,PC1], pca_data[:,PC1+PC2+1], label="Healthy")
    plt.scatter(pca_projection[:,PC1], pca_projection[:,PC1+PC2+1], marker=".", color='red', label="Faulty")
    for i, feature in enumerate(train.columns):
      plt.arrow(0, 0, pca.components_[PC1, i], pca.components_[PC1+PC2+1, i],
              head_width=0.01, color="r")
      plt.text(pca.components_[PC1, i], pca.components_[PC1+PC2+1, i], feature)
    plt.xlabel(f"PC{PC1+1}")
    plt.ylabel(f"PC{PC1+PC2+2}")
    plt.grid(True)
    plt.title(f"Biplot for PC{PC1+1} and PC{PC1+PC2+2} for model z-score (STD)")
    plt.legend()
    plt.show()

for PC1 in range(5):
  for PC2 in range(4-PC1):
    plt.scatter(pca_data[:469,PC1], pca_data[:469,PC1+PC2+1], color="blue", label="Healthy 1:469")
    plt.scatter(pca_data[470:,PC1], pca_data[470:,PC1+PC2+1], color="purple", label="Healthy 471:end")
    plt.scatter(pca_projection[:469,PC1], pca_projection[:469,PC1+PC2+1], marker=".", color='red', label="Broken")
    plt.scatter(pca_projection[470:,PC1], pca_projection[470:,PC1+PC2+1], marker=".", color='green', label="Fixed")
    for i, feature in enumerate(train.columns):
      plt.arrow(0, 0, pca.components_[PC1, i], pca.components_[PC1+PC2+1, i],
              head_width=0.01, color="r")
      plt.text(pca.components_[PC1, i], pca.components_[PC1+PC2+1, i], feature)
    plt.xlabel(f"PC{PC1+1}")
    plt.ylabel(f"PC{PC1+PC2+2}")
    plt.grid(True)
    plt.title(f"Biplot for PC{PC1+1} and PC{PC1+PC2+2} for model z-score (STD)")
    plt.legend()
    plt.show()

z_score_reconstructed = pca.inverse_transform(pca_data)
spe = np.sum((z_score - z_score_reconstructed) ** 2, axis=1)

cov_matrix = np.cov(pca_data, rowvar=False)
cov_matrix_inv = np.linalg.inv(cov_matrix)
t2 = np.sum((pca_data / np.sqrt(pca.explained_variance_))**2, axis=1)

t2_mean, t2_std = np.mean(t2), np.std(t2)
spe_mean, spe_std = np.mean(spe), np.std(spe)
t2_limit_2sd = t2_mean + 2 * t2_std
t2_limit_3sd = t2_mean + 3 * t2_std
spe_limit_2sd = spe_mean + 2 * spe_std
spe_limit_3sd = spe_mean + 3 * spe_std

plt.figure(figsize=(12, 6))
plt.plot(t2, 'o', label='T2 score')
plt.axhline(t2_limit_2sd, color='r', linestyle='--', label='2 SD Limit')
plt.axhline(t2_limit_3sd, color='orange', linestyle='--', label='3 SD Limit')
plt.title('T2 Control Chart')
plt.xlabel('Observation')
plt.ylabel('T2 score')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(spe, 'o', label='SPEx score')
plt.axhline(spe_limit_2sd, color='r', linestyle='--', label='2 SD Limit')
plt.axhline(spe_limit_3sd, color='orange', linestyle='--', label='3 SD Limit')
plt.title('SPEx Control Chart')
plt.xlabel('Observation')
plt.ylabel('SPEx score')
plt.legend()
plt.grid(True)
plt.show()

z_score_reconstructed_f = pca.inverse_transform(pca_projection)
spe_f = np.sum((z_score_f - z_score_reconstructed_f) ** 2, axis=1)
spe_f = spe_f.to_numpy()

t2_f = np.sum((pca_projection / np.sqrt(pca.explained_variance_))**2, axis=1)

plt.plot(spe, t2, 'o')
plt.plot(spe_f, t2_f, '.')
plt.axhline(t2_limit_2sd, color='r', linestyle='--', label='2 SD Limit')
plt.axhline(t2_limit_3sd, color='orange', linestyle='--', label='3 SD Limit')
plt.axvline(spe_limit_2sd, color='r', linestyle='--', label='2 SD Limit')
plt.axvline(spe_limit_3sd, color='orange', linestyle='--', label='3 SD Limit')
#plt.title('')
plt.ylabel('T2 score')
plt.xlabel('SPEx score')
#plt.legend()
plt.grid(True)
plt.show()

plt.plot(spe, t2, 'o')
plt.plot(np.delete(spe_f, 469), np.delete(t2_f, 469), '.')
plt.axhline(t2_limit_2sd, color='r', linestyle='--', label='2 SD Limit')
plt.axhline(t2_limit_3sd, color='orange', linestyle='--', label='3 SD Limit')
plt.axvline(spe_limit_2sd, color='r', linestyle='--', label='2 SD Limit')
plt.axvline(spe_limit_3sd, color='orange', linestyle='--', label='3 SD Limit')
#plt.title('')
plt.ylabel('T2 score')
plt.xlabel('SPEx score')
#plt.legend()
plt.grid(True)
plt.show()

plt.plot(spe, t2, 'o')
plt.plot(spe_f[470:], t2_f[470:], '.')
plt.axhline(t2_limit_2sd, color='r', linestyle='--', label='2 SD Limit')
plt.axhline(t2_limit_3sd, color='orange', linestyle='--', label='3 SD Limit')
plt.axvline(spe_limit_2sd, color='r', linestyle='--', label='2 SD Limit')
plt.axvline(spe_limit_3sd, color='orange', linestyle='--', label='3 SD Limit')
#plt.title('')
plt.ylabel('T2 score')
plt.xlabel('SPEx score')
#plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(t2, 'o', label='T2 score')
plt.plot(t2_f[470:], '.', label='projected T2 score')
plt.axhline(t2_limit_2sd, color='r', linestyle='--', label='2 SD Limit')
plt.axhline(t2_limit_3sd, color='orange', linestyle='--', label='3 SD Limit')
plt.title('T2 Control Chart')
plt.xlabel('Observation')
plt.ylabel('T2 score')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(spe, 'o', label='SPEx score')
plt.plot(spe_f[470:], '.', label='projected SPEx score')
plt.axhline(spe_limit_2sd, color='r', linestyle='--', label='2 SD Limit')
plt.axhline(spe_limit_3sd, color='orange', linestyle='--', label='3 SD Limit')
plt.title('SPEx Control Chart')
plt.xlabel('Observation')
plt.ylabel('SPEx score')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.decomposition import KernelPCA
kernelMods = ["poly", "rbf", "cosine"]
for kmod in kernelMods:
  kpca = KernelPCA(kernel=kmod)
  kpca_data = kpca.fit_transform(z_score)
#pca_data /= np.max(np.abs(pca_data), axis=0)
#pca_data *= np.max(np.abs(pca.components_))
  for PC1 in range(5):
    for PC2 in range(4-PC1):
      plt.scatter(kpca_data[:,PC1], kpca_data[:,PC1+PC2+1])
    #for i, feature in enumerate(train.columns):
      #plt.arrow(0, 0, pca.components_[PC1, i], pca.components_[PC1+PC2+1, i],
              #head_width=0.01, color="r")
      #plt.text(pca.components_[PC1, i], pca.components_[PC1+PC2+1, i], feature)
      plt.xlabel(f"PC{PC1+1}")
      plt.ylabel(f"PC{PC1+PC2+2}")
      plt.grid(True)
      plt.title(f"Biplot for PC{PC1+1} and PC{PC1+PC2+2} for Kernel PCA with {kmod} kernel")
      plt.show()