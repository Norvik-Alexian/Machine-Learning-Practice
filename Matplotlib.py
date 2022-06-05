import matplotlib.pyplot as plt

first_numbers_collection = [2, 5, 9]
second_numbers_collection = [7, 3, 4]

plt.xlabel('X axis')  # created x lable and called x-axis
plt.ylabel('Y axis')  # created y label and called y-axis

plt.xlim(4, 6)  # x-axis starts from 4 to 6
plt.ylim(3, 5)  # y-axis starts from 3 to 5

plt.plot(first_numbers_collection, second_numbers_collection, 'b.-')
plt.plot(second_numbers_collection, first_numbers_collection, 'gH--')
plt.scatter(first_numbers_collection, second_numbers_collection)
plt.bar(first_numbers_collection, second_numbers_collection)
plt.barh(first_numbers_collection, second_numbers_collection)

figure1 = plt.figure()
figure2 = plt.figure()

axis11 = figure1.add_subplot(2, 2, 1)
axis12 = figure1.add_subplot(2, 2, 2)
axis13 = figure1.add_subplot(2, 2, 4)
axis21 = figure2.add_subplot(1, 1, 1)

axis11.plot(first_numbers_collection, second_numbers_collection)
axis12.barh(first_numbers_collection, second_numbers_collection, color='red')
axis13.scatter(first_numbers_collection, second_numbers_collection, color='green')
axis21.bar(second_numbers_collection, first_numbers_collection)

axis11.set_xlim(2, 6)
axis21.set(title='the second axis', xlabel='lower axis')
axis13.set_xlabel('xxx')

plt.show()

# figure1.savefig('figure-one')