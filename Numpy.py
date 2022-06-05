import numpy as np

np_array = np.array([1, 2, 3, 4])
zeros_arr = np.zeros((2, 2))
ones_arr = np.ones((3, 4))
full_arr = np.full((4, 4), 7)  # 4 by 4 dimension matrix with value of 7
random_arr = np.random.rand(3, 3) * 100  # 3 by 3 dimension matrix with random values.

csv_file = np.genfromtxt('./dataset/username.csv', delimiter=';', skip_header=1)

# np.savetxt('./dataset/new_dataset.csv', random_arr, delimiter=';')

second_np_arr = np.append(np_array, [5, 6, 7])

# print(second_np_arr)
# print(np_array[0])
# print(random_arr[1, 1:])
# print(random_arr[:, 1])  # extract second column

# print(csv_file.min(axis=0))  # found min value of csv column
# print(np_array.sum())
# print(zeros_arr.min())
