import pandas as pd
import matplotlib.pyplot as plt

'''
pandas and matplotlib practice
'''

numbers = [1, 2, 3, 4, 5]

matrix = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15]
]

data_frame = pd.DataFrame(matrix)

dict_table = {'column1': [1, 2, 3, 4, 5], 'column2': [6, 7, 8, 9, 10]}

dict_data_frame = pd.DataFrame(dict_table, index=['row1', 'row2', 'row3', 'row4', 'row5'])

print(dict_data_frame)

# dataframe investigations
print(dict_data_frame.head(2))
print(dict_data_frame.tail(3))
print(dict_data_frame.columns)
print(dict_data_frame.index)
print(dict_data_frame.dtypes)
print(dict_data_frame.shape)
print(dict_data_frame.shape[0])
print(dict_data_frame.shape[1])
print(dict_data_frame.describe())
print(dict_data_frame.isnull().sum())

# remove first row of dataframe
new_dict_data_frame = dict_data_frame.drop(labels='row1', axis=0)

# find average value of each row
dict_data_frame['average'] = dict_data_frame.mean(axis=1)

# select the value of second row and first column
print(dict_data_frame.iloc[1, 0])

# select the first column of dataframe
print(dict_data_frame['column1'])

# select the third row of dataframe
print(dict_data_frame.loc['row3'])

dict_data_frame['column1'] *= 3

print(dict_data_frame['column1'].apply(lambda x: x * 3))

first_collection = [2, 5, 9]
second_collection = [7, 4, 6]

figure1 = plt.figure()

axis_1 = figure1.add_subplot(2, 2, 1)
axis_2 = figure1.add_subplot(2, 2, 2)
axis_3 = figure1.add_subplot(2, 2, 4)

axis_1.plot(first_collection, second_collection)
axis_2.barh(first_collection, second_collection, color='red')
axis_3.scatter(first_collection, second_collection, color='green')
axis_1.bar(first_collection, second_collection)

axis_1.set_xlim(2, 6)
axis_1.set_ylim(4, 7)

axis_2.set(title='second axis', xlabel='lower axis')

axis_3.set_xlabel('x label')
axis_3.set_ylabel('y label')

plt.show()

figure1.savefig('first.png')
