import math

def euclidean_distance(pt1, pt2):
    distance = 0
    squared = 0
    count = 0
    for point_a in pt1:
        squared += ((point_a - pt2[count])**2)
        count += 1

    distance += squared
    distance = math.sqrt(distance)

    return distance

print(euclidean_distance([1,2], [4,0]))
print(euclidean_distance([5,4,3], [1,7,9]))
