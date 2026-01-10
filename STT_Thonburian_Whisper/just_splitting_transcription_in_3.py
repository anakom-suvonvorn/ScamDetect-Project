
import random

def split_list_by_percentages(data_list, percentages):
    shuffled_list = list(data_list)
    random.shuffle(shuffled_list)

    if sum(percentages) != 100:
        raise ValueError("Percentages must sum to 100")

    chunks = []
    start_index = 0
    cumulative_percent = 0
    list_length = len(shuffled_list)

    for percent in percentages:
        cumulative_percent += percent
        # Calculate the end index. Using integer division (//) might drop elements,
        # so recalculating from the cumulative percent avoids accumulating errors.
        end_index = (cumulative_percent * list_length) // 100
        
        # Slice the list and append the chunk
        chunks.append(shuffled_list[start_index:end_index])
        
        # Update the start index for the next iteration
        start_index = end_index
        
    return chunks

all_numbers = [1, 2, 3, 4, 5, 7, 8, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 34, 35, 37, 38, 39, 40, 41, 46, 47, 49, 50, 51, 52, 55, 56, 57, 59, 60, 61, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75]
split_percents = [50, 25, 25]
result_chunks = split_list_by_percentages(all_numbers, split_percents)

print(f"Original list length: {len(all_numbers)}")

print("\nChunks:")
for i, chunk in enumerate(result_chunks):
    print(f"Chunk {i+1} (Length: {len(chunk)}): {chunk}")