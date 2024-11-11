import pandas as pd
import os
import ast
import chardet
import csv

if __name__ == '__main__':
    net_name = 'Winnipeg'
    curr_dir = os.getcwd()

    # Define file paths
    net_file_path = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\{net_name}_network.csv')
    drl_file_path = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\csv\\{net_name}_DRL.csv')

    # Detect encoding of the net_file
    with open(net_file_path, 'rb') as file:
        content = file.read()
        result = chardet.detect(content)
        encoding = result['encoding']

    # Read net_file and parse 'od' column into a list of tuples
    net_od_data = []
    net_cost_data = {}

    with open(net_file_path, 'r', newline='', encoding=encoding) as file:
        reader = csv.DictReader(file)
        for row in reader:
            from_to_tuple = (int(row['From']), int(row['To']))
            net_od_data.append(from_to_tuple)
            net_cost_data[from_to_tuple] = float(row['Cost'])
    print(net_od_data[0])
    # Read drl_file into a DataFrame
    drl_file = pd.read_csv(drl_file_path)

    # Initialize an empty 'let' column
    drl_file['let'] = 0

    # Iterate over each row in drl_file
    for index, row in drl_file.iterrows():
        path = row['path']
        path_od_pairs = ast.literal_eval(path)  # Convert path string to list
        total_cost = 0

        # Iterate over each od pair in the path
        for i in range(len(path_od_pairs) - 1):
            start = path_od_pairs[i]
            end = path_od_pairs[i + 1]
            od = (start, end)
            # Find the corresponding od in net_od_data and accumulate the cost
            if od in net_od_data:
                cost = net_cost_data[od]
                total_cost += cost

        # Write the accumulated total_cost into the 'let' column
        drl_file.at[index, 'let'] = total_cost

    # Write the updated DataFrame back to CSV
    drl_file.to_csv(drl_file_path, index=False)

    print(f"Successfully updated the 'let' column in {drl_file_path}")
