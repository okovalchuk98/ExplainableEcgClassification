import gzip
import pickle
import numpy as np

def load_from_gz_file(file_path):
    chunks = []
    chunk_index = 0
    with gzip.GzipFile(file_path, 'r') as f:
        while True:
            try:
                chunk = pickle.load(f)
                chunks.append(chunk)
                chunk_index = chunk_index + 1
            except EOFError:
                break
    large_array = np.concatenate(chunks)
    print(f'loaded {str(len(large_array))} items')
    del chunk
    return large_array

def save_array_as_gz_file(filepath, array_data, chunk_size = 1500):
    with gzip.GzipFile(filepath, 'w', compresslevel=5) as f:
        for i in range(0, len(array_data), chunk_size):
            chunk = array_data[i:i+chunk_size]
            pickle.dump(chunk, f)
            print(f"Saved {str(i)} chank")