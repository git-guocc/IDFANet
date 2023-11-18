import numpy as np

n = 3
p = [10, 2, 1]  # P(1:3)
q = [3, 2, 1, 1]  # Q(0:3)

cost = np.zeros((n+2, n+1))
root = np.zeros((n+1, n+1), dtype=int)

w = np.zeros((n+2, n+1))
for i in range(n+1):
    w[i+1][i] = q[i]
    for j in range(i+1, n+1):
        w[i+1][j] = w[i+1][j-1] + p[j-1] + q[j]

for interval in range(1, n+1):
    for i in range(1, n-interval+2):
        j = i + interval - 1
        cost[i][j] = np.inf
        for r in range(i, j+1):
            t = cost[i][r-1] + cost[r+1][j] + w[i][j]
            if t < cost[i][j]:
                cost[i][j] = t
                root[i][j] = r

print(cost[1][n], root[1:n+1, 1:n+1])
