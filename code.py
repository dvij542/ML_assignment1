import csv
import numpy as np

dataset = []
dataset_split_countries = []
countries = []

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
            countries.append(curr_country)
        row["Date"] = i
        dataset.append(row)
        line_count += 1
        i+=1
    print(f'Processed {line_count} lines.')
    dataset = np.array(dataset)
    print(dataset[500])
    print(dataset.shape)

def split(field, dataset, val) :
    groups = {}
    if(field=='Country') :
        for country in countries :
            groups[country] = []
        for row in dataset :
            groups[row[field]].append(row)
        return groups
    else :
        groups['left'] = []
        groups['right'] = []
        for row in dataset :
            #print(row[field])
            if(float(row[field])<val) :
                groups['left'].append(row)
            else :
                groups['right'].append(row)
        return groups

def calc_entropy_loss(groups, total_n, init_entropy) :
    final_entropy = 0
    for group in groups.items() :
        temp_l = []
        #print(len(group[1]))
        for j in range(len(group[1])) :
            temp_l.append(int(group[1][j]["Deaths"]))
        final_entropy += (len(group[1])/total_n)*np.sqrt(np.var(np.array(temp_l)))
    print(init_entropy-final_entropy)
    return (init_entropy-final_entropy)
    

def create_split(dataset) :
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    temp_l = []
    for j in range(dataset.shape[0]) :
        temp_l.append(int(dataset[j]["Deaths"]))
    init_entropy = np.sqrt(np.var(np.array(temp_l)))
    print("Current Entropy : ", init_entropy)
    for field in ["Country","Confirmed", "Recovered", "Date"] :
        print("Entropy loss according to split ",field, " is :-")
        if(field=="Country") :
            groups = split(field, dataset, "all")
            score = calc_entropy_loss(groups,dataset.shape[0], init_entropy)
            if(score<b_score) :
                b_score = score
                b_value = "all"
                b_index = "Country"
                b_groups = groups
            continue
        temp_l = []
        for j in range(dataset.shape[0]) :
            temp_l.append(int(dataset[j][field]))
        value_to_split = np.average(np.array(temp_l))
        groups = split(field, dataset, value_to_split)
        score = calc_entropy_loss(groups,dataset.shape[0], init_entropy)
        if(score<b_score) :
            b_score = score
            b_value = value_to_split
            b_index = field
            b_groups = groups
    return b_score, b_value, b_index, b_groups

first_split = create_split(dataset)