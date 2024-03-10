import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

#x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
#y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
# Create a Pandas DataFrame
#points = pd.DataFrame({'x': x, 'y': y})

# Convert the DataFrame to CSV
#points.to_csv('xy.csv', index=False)
#size = 10

# another set of data points
points = pd.read_csv("https://tinyurl.com/y25lvxug")
size = len(points)
x = points.iloc[:, :-1].values
y = points.iloc[:, 1].values


plt.scatter(x, y)
plt.show()

# how to determine the number of clusters??
inertias = []

for i in range(1, size + 1):  # the max number of clusters possible is size of data
    kmeans = (KMeans(n_clusters=i).fit(points))
    inertias.append(kmeans.inertia_)

plt.plot(range(1, size+1), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
# in the above plot we see that the plot becomes linear at size 4, so, K=4

kmeans = KMeans(n_clusters=4).fit(points)
#kmeans = KMeans(n_clusters=2).fit(points)

plt.scatter(x, y, c=kmeans.labels_)
plt.show()

print(kmeans.cluster_centers_)
print(kmeans.predict([[14, 5]]))
