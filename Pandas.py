import pandas as pd
import matplotlib.pyplot as plt

numbers = [1, 2, 3, 4, 5]
data_frame = pd.DataFrame(numbers)

matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
]
matrix_data_frame = pd.DataFrame(matrix)

dict_table = {'column1': [1, 2, 3, 4, 5], 'column2': [6, 7, 8, 9, 10]}
dict_data_frame = pd.DataFrame(dict_table, index=['row1', 'row2', 'row3', 'row4', 'row5'])

print(dict_data_frame)
print(dict_data_frame.head(2))  # first 2 rows of the dataset.
print(dict_data_frame.tail(2))  # last 2 rows of the dataset.
print(dict_data_frame.columns)  # shows existing columns in dataset.
print(dict_data_frame.index)  # shows existing rows in dataset
print(dict_data_frame.dtypes)  # shows data types of each column
print(dict_data_frame.shape)  # shows the size of dataset

addresses_data_frame = pd.read_csv('./dataset/addresses.csv', sep=',', header=None)
addresses_data_frame.columns = ['name', 'lastname', 'addresses', 'state', 'town', 'postal code']

print(addresses_data_frame)
print(addresses_data_frame['lastname'])  # extract and shows lastname column using bracket notation
print(addresses_data_frame.name)  # extract and shows name column using dot notation
print(addresses_data_frame[['name', 'lastname']])  # extract and shows more than one column using names
print(addresses_data_frame.iloc[0])  # shows the details of first row using index number
print(addresses_data_frame.iloc[1, 0])  # shows the details of second row and first row using index number
print(addresses_data_frame.iloc[1:4, 0])  # show the details of second row and first row

addresses_data_frame['phone'] = ['+43', '+23', '+98', '+97', '+85', '+24']  # add new phone column to dataset.
addresses_data_frame['zip'] = addresses_data_frame['postal code'] + 5  # add another new column
addresses_data_frame['highest_zip'] = addresses_data_frame['zip'] > 9000  # returns true or false

print(addresses_data_frame.describe())  # shows the descriptive statictics of the dataset

print(addresses_data_frame[['postal code', 'zip']].sum(axis=1))  # summation of columns.
print(addresses_data_frame[['postal code', 'zip']].mean())  # extract the mean value of dataset.
print(addresses_data_frame['zip'].unique())  # shows the unique value of zip column.

addresses_data_frame['zip'].plot.bar()  # generates charts for dataset
plt.show()

addresses_data_frame = addresses_data_frame.drop('highest_zip', axis=1)  # summation of columns

print(addresses_data_frame[addresses_data_frame['zip'] > 9000])

# Filtering in dataset
print(addresses_data_frame[(addresses_data_frame['zip'] > 1000) & (addresses_data_frame['name'] != 'Jack')])

addresses_data_frame['zip'] = addresses_data_frame['zip'].apply(lambda x: x * 2 + 10)  # update a column of dataset

print(addresses_data_frame.isnull().sum())  # find the missing value in dataset.