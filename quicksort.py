import random

def deterministic_partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def deterministic_quicksort(arr, low, high):
    if low < high:
        pi = deterministic_partition(arr, low, high)
        deterministic_quicksort(arr, low, pi - 1)
        deterministic_quicksort(arr, pi + 1, high)

def randomized_partition(arr, low, high):
    pivot_index = random.randint(low, high)
    arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
    return deterministic_partition(arr, low, high)

def randomized_quicksort(arr, low, high):
    if low < high:
        pi = randomized_partition(arr, low, high)
        randomized_quicksort(arr, low, pi - 1)
        randomized_quicksort(arr, pi + 1, high)

if __name__ == "__main__":
    data = [10, 7, 8, 9, 1, 5]
    print("Original Array:", data)

    arr1 = data.copy()
    deterministic_quicksort(arr1, 0, len(arr1) - 1)
    print("\nSorted Array using Deterministic Quick Sort:", arr1)

    arr2 = data.copy()
    randomized_quicksort(arr2, 0, len(arr2) - 1)
    print("Sorted Array using Randomized Quick Sort:", arr2)
