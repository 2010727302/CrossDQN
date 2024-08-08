import numpy as np
from config import *
import csv
import pandas as pd
# 读取 CSV 文件
df = pd.read_csv('max_q_action_index.csv', header=None)

# 提取倒数10行
last_10_rows = df.tail(10)

# 转换为 NumPy 数组
max_q_action_index = last_10_rows.to_numpy()

print(max_q_action_index)


# Function to load ad_bid_id_list from a CSV file
def load_ad_bid_data_from_csv(csv_file_name):
    with open(csv_file_name, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        ad_bid_data = [list(map(float, row)) for row in reader]
    return np.array(ad_bid_data)

# Assuming 'ad_bid_id_list.csv' is the file we saved previously
csv_file_name = 'ad_bid_id_list.csv'
ad_bid_id_list = load_ad_bid_data_from_csv(csv_file_name)
csv_file_name = 'ad_id_list.csv'
ad_id_list = load_ad_bid_data_from_csv(csv_file_name)
csv_file_name = 'oi_id_list.csv'
oi_id_list = load_ad_bid_data_from_csv(csv_file_name)
print("Loaded ad_bid_id_list from CSV:")
print(ad_bid_id_list)
print(ad_id_list)
print(oi_id_list)
# Find the index of the maximum value in each row of max_q_action_index
max_indices = np.argmax(max_q_action_index, axis=1)

# Convert indices to corresponding WHOLE_ACTION
allocations = [WHOLE_ACTION[index] for index in max_indices]

for allocation in allocations:
    print(allocation)
# Function to calculate the payment for an allocation
def calculate_payment_and_gmv(bids, allocation, ad_id_list, oi_id_list):
    # Identify the indices where allocation is 1 (ads) and 0 (ois)
    ad_indices = [i for i, alloc in enumerate(allocation) if alloc == 1]
    oi_indices = [i for i, alloc in enumerate(allocation) if alloc == 0]

    if len(ad_indices) == 0:
       payment=0  # No bids allocated, so payment and GMV are both 0
    # print("ad",ad_indices)
    # print("oi",oi_indices)
    # Sort bids in descending order, but keep track of the original indices
    sorted_bid_indices = np.argsort(bids)[::-1]
    # print("sorted_bid_indices",sorted_bid_indices)
    sorted_bids = bids[sorted_bid_indices]

    # Calculate the payment as the sum of the second highest to (1+n)th highest bids
    if len(ad_indices) == 5:
        payment = sum(sorted_bids[1:5])
    else:
        payment = sum(sorted_bids[1:1 + len(ad_indices)])

    # Calculate GMV by summing the corresponding ad_id_list values and the top OI values
    gmv_from_ads = sum(ad_id_list[sorted_bid_indices[i]] for i in range(1, 1 + len(ad_indices)))

    # For OIs, sort the oi_id_list values at the non-allocated positions and take the top ones
    oi_count = len(oi_indices)  # Count how many positions are allocated to OIs
    sorted_oi_values = sorted(oi_id_list, reverse=True)
    gmv_from_ois = sum(sorted_oi_values[:oi_count])
    # print(gmv_from_ois)
    # Total GMV is the sum of GMV from ads and OIs
    gmv = gmv_from_ads + gmv_from_ois

    return payment, gmv


# Example usage
for bids, allocation, ad_ids, oi_ids in zip(ad_bid_id_list, allocations, ad_id_list, oi_id_list):
    payment, gmv = calculate_payment_and_gmv(bids, allocation, ad_ids, oi_ids)
    print(f"Allocation: {allocation}, Payment: {payment}, GMV: {gmv}")
