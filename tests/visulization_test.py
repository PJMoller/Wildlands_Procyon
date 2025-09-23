import matplotlib.pyplot as plt
import csv
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(base_dir, 'data', 'visitor_sample.csv')

x = []
y = []


with open(file_path,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
    
    for row in plots:
        x.append(row[0])
        y.append(row[0])

plt.bar(x, y, color = 'g', width = 0.72, label = "Age")
plt.xlabel('Names')
plt.ylabel('Ages')
plt.title('Ages of different persons')
plt.legend()
plt.show()