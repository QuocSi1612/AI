import random as rd
import math
import matplotlib.pyplot as plt

# Tạo 20 điểm ngẫu nhiên
data_X = [rd.uniform(0, 50) for _ in range(200)]
data_Y = [rd.uniform(0, 50) for _ in range(200)]
points = list(zip(data_X, data_Y))

# Số cụm và số lần lặp tối đa
k = 3
max_iterations = 100

# Khởi tạo tâm cụm ngẫu nhiên
centroids = [(rd.uniform(0, 50), rd.uniform(0, 50)) for _ in range(k)]

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def assign_point(data, centroids):
    clus = [[] for _ in range(k)]
    for eachPoint in data:
        dis = [distance(eachPoint, c) for c in centroids]
        min_index = dis.index(min(dis))
        clus[min_index].append(eachPoint)
    return clus

def update_centroids(clusters):
    new_points = []
    for cluster in clusters:
        if len(cluster) == 0:
            new_points.append((0, 0))
        else:
            x_mean = sum(p[0] for p in cluster) / len(cluster)
            y_mean = sum(p[1] for p in cluster) / len(cluster)
            new_points.append((x_mean, y_mean))
    return new_points

# Thuật toán K-Means
for _ in range(max_iterations):
    clusters = assign_point(points, centroids)
    new_centroids = update_centroids(clusters)
    if new_centroids == centroids:
        break
    centroids = new_centroids

# Vẽ trực quan kết quả
colors = ['red', 'green', 'blue']
for i, cluster in enumerate(clusters):
    xs = [p[0] for p in cluster]
    ys = [p[1] for p in cluster]
    plt.scatter(xs, ys, color=colors[i])

# Vẽ tâm cụm
for i, centroid in enumerate(centroids):
    plt.scatter(centroid[0], centroid[1], color='black', marker='x', s=100)

plt.title("Phân cụm K-Means (Tự cài đặt)")
plt.xlabel("X")
plt.ylabel("Y")

plt.grid(True)

plt.show()



