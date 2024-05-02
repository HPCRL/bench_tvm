import json
import re
import numpy as np

def replace_auto_unroll_max_step(input_string, new_value):
    # Replace auto_unroll_max_step with the new value
    return re.sub(r'auto_unroll_max_step\$([0-9]+)', f'auto_unroll_max_step${new_value}', input_string)

def process_json(json_str, new_tile_vectors):
    data = json.loads(json_str)
    # print("Data[i]:", data['i'])
    # print("Data[i][1]:", data['i'][1])
    
    if 'i' in data and len(data['i']) > 0 and len(data['i'][0]) > 1 and len(data['i'][0][1]) > 1:
        nodes = data['i'][1][1]
        # print("Nodes:", nodes)
        counter = 0
        
        for node in nodes:
            # Check "SP"
            # print("Node:", node)
            if len(node) > 0 and node[0] == "SP":
                # Replace the tile-vector
                if counter < len(new_tile_vectors):
                    # print("Replacing tile vector", node[4], "with", new_tile_vectors[counter])
                    node[4] = new_tile_vectors[counter]
                    counter += 1
                else:
                    raise ValueError("Insufficient tile vectors provided.")

        # replace the auto_unroll_max_step
        second_nodes = data['i'][1][1]
        for node in second_nodes:
            if isinstance(node, list) and len(node) > 0 and isinstance(node[-1], str):
                node[-1] = replace_auto_unroll_max_step(node[-1], new_tile_vectors[-1][0])
    else:
        raise ValueError("No suitable node structure found in JSON.")

    # Convert the modified data back to JSON string
    return json.dumps(data)


# new_tile_vectors = [
#     [1, 8, 64],
#     [2, 1, 32],
#     [1],
#     [0]
# ]

# # Process the JSON string and print the new JSON
# print ("Original JSON:")
# print(json_str)
# print("\nModified JSON:")
# modified_json_str = process_json(json_str, new_tile_vectors)
# print(modified_json_str)


def tvm_config_space(
                y1, y2, y3, M,
                x1, x2, x3, N,
                k1, K,
                max_unroll):
    if int(np.prod([y1, y2, y3])) > M or \
       int(np.prod([x1, x2, x3])) > N or \
       k1 > K:
        return None
    else:
        # creat the new_tile_vectors structure list
        new_tile_vectors = [
            [y1, y2, y3],
            [x1, x2, x3],
            [k1],
            [max_unroll]
        ]
        return new_tile_vectors

def get_factors(x):
    factor_list = []
    for i in range(1, x + 1):
        if x % i == 0:
            factor_list.append(i)
    return factor_list

def launch_hyper(M, N, K, fi):
    
    # only for testcase 0 sketch0
    json_str = '{"i": [["[\\"_matmul\\", 512, 1024, 64, \\"float32\\"]", "llvm -keys=cpu -mcpu=core-avx2", [16, 64, 64, 0, 0, 0, 0, 0], "", 1, []], [[], [["CHW", 2, "local"], ["SP", 2, 0, 512, [1, 64, 8], 1], ["SP", 2, 4, 64, [2, 32, 1], 1], ["SP", 2, 8, 1024, [16], 1], ["RE", 2, [0, 4, 1, 5, 8, 2, 6, 9, 3, 7]], ["FSP", 3, 0, 1, 2], ["FSP", 3, 3, 2, 2], ["RE", 3, [0, 3, 1, 4, 2, 5]], ["CA", 2, 3, 3], ["FU", 3, [0, 1, 2, 3]], ["AN", 3, 0, 3], ["PR", 2, 0, "auto_unroll_max_step$64"], ["AN", 2, 9, 2], ["AN", 3, 2, 2]]]], "r": [[0.00198047, 0.0019683, 0.00198756, 0.00197635, 0.00198229, 0.00197825, 0.00199231, 0.00198754, 0.001991, 0.00198804], 0, 0.130737, 1714270206], "v": "v0.6"}'

    factors_M = get_factors(M)
    factors_N = get_factors(N)
    factors_K = get_factors(K)
    print(f"size: {M}x{N}x{K}, factors_M: {factors_M}, factors_N: {factors_N}, factors_K: {factors_K}")
    print(f"size of factors_M: {len(factors_M)}, size of factors_N: {len(factors_N)}, size of factors_K: {len(factors_K)}")
    
    configs = []
    length = 0
    for y1 in factors_M:
        for y2 in factors_M:
            for y3 in factors_M:
                for x1 in factors_N:
                    for x2 in factors_N:
                        for x3 in factors_N:
                            for k1 in factors_K:
                                for max_unroll in [0, 16, 64, 512]:
                                    tile_list = tvm_config_space(
                                        y1, y2, y3, M,
                                        x1, x2, x3, N,
                                        k1, K,
                                        max_unroll
                                    )
                                    if tile_list:
                                        modified_json_str = process_json(json_str, tile_list)
                                        with open('input_full.json', 'a') as file:
                                            file.write(modified_json_str + '\n')
                                        # print("tile_list:", tile_list)
                                        # input("pause")
                                        # configs.append(tile_list)
                                        length += 1
    # print("configs[0]:", configs[0])
    # input("pause")
    print(f"M = {M}, N = {N}, K = {K}, sizes of configs: {length}")

sizes=[
    #Bert large
[512,64,1024],      #BMATmul
[512,4096,1024],    #MLP1
[512,1024,4096],    #MLP2

    #Bert basic
[512,64,768],       #BMATmul
[512,3072,768],     #MLP1
[512,768,3072],     #MLP2
]

if __name__ == "__main__":
    
    for fi, size in enumerate(sizes):
        # if fi != 0:
        #     continue
        M, N, K = size
        launch_hyper(M, N, K, fi)