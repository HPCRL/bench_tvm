import json
import re
import random

def replace_auto_unroll_max_step(input_string, new_value):
    # Replace auto_unroll_max_step with the new value
    return re.sub(r'auto_unroll_max_step\$([0-9]+)', f'auto_unroll_max_step${new_value}', input_string)

def process_json(json_str, new_tile_vectors):
    data = json.loads(json_str)
    # print("Data[i]:", data['i'])
    # print("Data[i][1]:", data['i'][1])
    
    if 'i' in data and len(data['i']) > 0 and len(data['i'][0]) > 1 and len(data['i'][0][1]) > 1:
        nodes = data['i'][1][1]
        print("Nodes:", nodes)
        # Counter for the new tile vectors
        counter = 0
        
        # Iterate over each node in the nodes list
        for node in nodes:
            # Check if the node type is "SP"
            print("Node:", node)
            if len(node) > 0 and node[0] == "SP":
                # replace the tile-vector
                if counter < len(new_tile_vectors):
                    print("Replacing tile vector", node[4], "with", new_tile_vectors[counter])
                    node[4] = new_tile_vectors[counter]
                    counter += 1
                else:
                    raise ValueError("Insufficient tile vectors provided.")
                    break  # Exit loop if there are no more new vectors to replace
                
        # Now replace the auto_unroll_max_step pattern in the second part of "i"
        second_nodes = data['i'][1][1]
        for node in second_nodes:
            if isinstance(node, list) and len(node) > 0 and isinstance(node[-1], str):
                node[-1] = replace_auto_unroll_max_step(node[-1], new_tile_vectors[-1][0])
    else:
        raise ValueError("No suitable node structure found in JSON.")

    # Convert the modified data back to JSON string
    return json.dumps(data)

# # Example JSON string
json_str = '{"i": [["[\\"_matmul\\", 512, 1024, 64, \\"float32\\"]", "llvm -keys=cpu -mcpu=core-avx2", [16, 64, 64, 0, 0, 0, 0, 0], "", 1, []], [[], [["CHW", 2, "local"], ["SP", 2, 0, 512, [1, 64, 8], 1], ["SP", 2, 4, 64, [2, 32, 1], 1], ["SP", 2, 8, 1024, [16], 1], ["RE", 2, [0, 4, 1, 5, 8, 2, 6, 9, 3, 7]], ["FSP", 3, 0, 1, 2], ["FSP", 3, 3, 2, 2], ["RE", 3, [0, 3, 1, 4, 2, 5]], ["CA", 2, 3, 3], ["FU", 3, [0, 1, 2, 3]], ["AN", 3, 0, 3], ["PR", 2, 0, "auto_unroll_max_step$64"], ["AN", 2, 9, 2], ["AN", 3, 2, 2]]]], "r": [[0.00198047, 0.0019683, 0.00198756, 0.00197635, 0.00198229, 0.00197825, 0.00199231, 0.00198754, 0.001991, 0.00198804], 0, 0.130737, 1714270206], "v": "v0.6"}'

# # New tile vectors for the example
new_tile_vectors1 = [
    [1, 8, 64],
    [2, 1, 32],
    [1],
    [0]
]
new_tile_vectors2 = [
    [1, 1, 1],
    [1, 1, 1],
    [1],
    [1]
]
new_tile_vectors = [new_tile_vectors1, new_tile_vectors2]


# # Process the JSON string and print the new JSON
# print ("Original JSON:")
# print(json_str)
# print("\nModified JSON:")
# modified_json_str = process_json(json_str, new_tile_vectors)
# print(modified_json_str)

for each in new_tile_vectors:
    print("Original JSON:")
    print(json_str)
    print("each:", each)
    modified_json_str = process_json(json_str, each)
    print("modified_json_str:", modified_json_str)

    # Write the modified JSON string to an output file
    with open('output.json', 'a') as file:
        file.write(modified_json_str + '\n')

print("Modified JSON has been written to 'output.json'.")
