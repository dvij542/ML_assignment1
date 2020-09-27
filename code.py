import csv
import numpy as np

dataset = []

# Preprocessing
with open('AggregatedCountriesCOVIDStats.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    i=0
    curr_country = "haha"
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
            continue
        if(curr_country!=row["Country"]) :
        	curr_country = row["Country"]
        	i=0
        row["Date"] = i
        dataset.append(row)
        line_count += 1
        i+=1
    print(f'Processed {line_count} lines.')
    dataset = np.array(dataset)
    print(dataset[500])
    print(dataset.shape)