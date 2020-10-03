import csv
import numpy as np

dataset = []
dataset_split_countries = []
countries = []
min_threshold = 5

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
        if(len(group[1])==0) :
            continue
        for j in range(len(group[1])) :
            temp_l.append(int(group[1][j]["Deaths"]))
        final_entropy += (len(group[1])/total_n)*np.sqrt(np.var(np.array(temp_l)))
    #print(init_entropy-final_entropy)
    return (init_entropy-final_entropy)
    

def create_split(dataset) :
    b_index, b_value, b_score, b_groups = "nothing", 0, -1, None
    temp_l = []
    for j in range(dataset.shape[0]) :
        temp_l.append(int(dataset[j]["Deaths"]))
    init_entropy = np.sqrt(np.var(np.array(temp_l)))
    #print("Current Entropy : ", init_entropy)
    for field in ["Country","Confirmed", "Recovered", "Date"] :
        #print("Entropy loss according to split ",field, " is :-")
        if(field=="Country") :
            groups = split(field, dataset, "all")
            
            score = calc_entropy_loss(groups,dataset.shape[0], init_entropy)
            if(score>b_score) :
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
        #print("With value", value_to_split)
        if(score>b_score) :
            b_score = score
            b_value = value_to_split
            b_index = field
            b_groups = groups
    return {'score':b_score, 'value':b_value, 'index':b_index, 'groups':b_groups}

def to_leaf(val) :
    return {'score':-1, 'value':val, 'index':"Leaf", 'groups':"None"}

def rec_split(node, max_depth, depth) : # To recursively form the decision tree with max height as max_depth
    node['new_groups'] = {}
    if (node['index']=='nothing') :
        return
    if (depth>=max_depth and max_depth!=-1) or node['score']<=min_threshold:
        for group in node['groups'].items() :
            temp_l = []
            #print(len(group[1]))
            for j in range(len(group[1])) :
                temp_l.append(int(group[1][j]["Deaths"]))
            node['new_groups'][group[0]] = []
            node['new_groups'][group[0]] = to_leaf(np.average(np.array(temp_l)))
        del(node['groups'])
        return
    
    for group in node['groups'].items() :
        node['new_groups'][group[0]] = []
        node['new_groups'][group[0]] = create_split(np.array(group[1]))
        
    del(node['groups'])
    for group in node['new_groups'].items() :
        rec_split(group[1], max_depth, depth+1)

def predict(node, row):
    if node['index']=="Leaf" :
        return node['value']
    if node['index'] == 'Country' :
        return predict(node['new_groups'][row['Country']], row)
    if float(row[node['index']]) < node['value']:
        return predict(node['new_groups']['left'], row)
    else:
        return predict(node['new_groups']['right'], row)

def calculate_RMSE_error(root_node, test_dataset): # To calculate RMSE error for dataset test_dataset and tree with root node as root_node
    val = 0
    size = 0
    for data in test_dataset:
        size = size + 1
        val += (float(data['Deaths']) - float(predict(root_node, data)))*(float(data['Deaths']) - float(predict(root_node, data)))
    return np.sqrt(val/size)

# Example : Create decision tree using full dataset
root_node = create_split(dataset) # Calculate the root node by first splitting the data
rec_split(root_node, 9, 0) # Recursively split further taking max height as input, max_height = -1 if full decision tree is to be built
# NOTE : Adjust the threshold for leaf node from above, keep it 0 to fully construct the tree to the optimal limit
print("Training error :",calculate_error(root_node,dataset))
# Part 1 : Create 10 random 80 : 20 train and test dataset splits and evaluate their average loss
# Part 2 : Vary max height and find the value which gives better average value as per part 1
# Part 3 : Prune the best decision tree observed from part 2 further