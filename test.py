# import openpyxl
# import time
# import csv
#
# # print(type(int(time.time()).__str__()))
# # print('"aaa"')
# filename='C:\\Users\\22097\\Desktop\\LBMA-GOLD.csv'
# with open(filename,'r') as filecsv:
#     csvreader=csv.reader(filecsv)
#     # print(csvreader)
#     data=[]
#     for row in csvreader:
#         try:
#             print(float(row[1]))
#             data.append(row[1])
#         except:
#             continue
# print(data)
#
import numpy as np

x = np.arange(5)
y = np.arange(8)

print(y[5:8])

# for i in 1:
#     print(i)
