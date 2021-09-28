import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv
import pandas as pd
from sklearn.cluster import KMeans

rows = []

with open("star_with_gravity.csv",'r') as f:
    csvreader = csv.reader(f)
    for row in csvreader:
        rows.append(row)

headers = rows[0]
star_data = rows[1:]

df = pd.read_csv("star_with_gravity.csv")
df1 = df.dropna()

solar_mass_list = df["Mass"].tolist()
solar_radius_list = df["Radius"].tolist()

solar_mass_list.pop(0)
solar_radius_list.pop(0)

star_solar_mass_si_unit = []

for data in solar_mass_list:
    if data != '?':
        si_unit_mass = float(data)*1.989e+30
        star_solar_mass_si_unit.append(si_unit_mass)

star_solar_radius_si_unit = []

for data in solar_radius_list:
    if data != '?':
        si_unit_radius = float(data)* 6.957e+8
        star_solar_radius_si_unit.append(si_unit_radius)


star_masses = star_solar_mass_si_unit
star_radiuses = star_solar_radius_si_unit
star_names = df["Star_name"].tolist()
star_names.pop(0)

mass_list = df["Mass"].tolist()
mass_list.pop(0)

radius_list = df["Radius"].tolist()
radius_list.pop(0)

mass_radius_column = df.iloc[:, [3,4]].values

wcss = []
for k in range(1, 11):
    k_means = KMeans(n_clusters = k, random_state = 42)
    k_means.fit(mass_radius_column)
    wcss.append(k_means.inertia_)

plt.plot(range(1,11),wcss)
plt.title("elbow method")
plt.xlabel('Number of clusters')
plt.show()

k_means = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
prediction = k_means.fit_predict(mass_radius_column)

plt.figure(figsize = (10, 5))
sns.scatterplot(x = mass_radius_column[prediction == 0, 0], y = mass_radius_column[prediction == 0, 1], color = 'orange', label = 'Star Cluster 1')
sns.scatterplot(x = mass_radius_column[prediction == 1, 0], y = mass_radius_column[prediction == 1, 1], color = 'blue', label = 'Star Cluster 2')
sns.scatterplot(x = mass_radius_column[prediction == 2, 0], y = mass_radius_column[prediction == 2, 1], color = 'green', label = 'Star Cluster 3')
sns.scatterplot(x = k_means.cluster_centers_[:, 0], y = k_means.cluster_centers_[:, 1], color = 'red', label = 'Centroids', s = 100, marker = ',')

plt.title('Clusters of Stars')
plt.xlabel('Mass of Stars')
plt.ylabel('Radius of Stars')
plt.legend()
sns.scatterplot(x = mass_list,y = radius_list)
plt.title("STAR MASS AND RADIUS")
plt.xlabel('MASS')
plt.ylabel('RADIUS')
plt.show()
plt.figure()

