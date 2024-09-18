import os
import numpy as np
import faiss
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import recall_score
from sentence_transformers import SentenceTransformer


faiss.omp_set_num_threads(1)

# Function to load embeddings from the dataset
def load_embeddings():
    df = pd.read_parquet("hf://datasets/rachid16/rag_finetuning_data/data/train-00000-of-00001.parquet")
    questions = df['question'][:10000].tolist()
    answers = df['answer'][:10000].tolist()

    model = SentenceTransformer('all-MiniLM-L6-v2')
    question_embeddings = model.encode(questions)
    answers_embeddings = model.encode(answers)
    return np.array(question_embeddings), np.array(answers_embeddings)


def quantize_vector(fp32_vectors, target_precision='fp32'):
    fp32_vectors = fp32_vectors.astype(np.float32)
    
    if target_precision == 'int8':
        # Normalize to the range of INT8 (-128 to 127)
        int8_max = 127  # Max value for signed 8-bit integer
        int8_min = -128  # Min value for signed 8-bit integer
        normalized = np.clip(fp32_vectors, 0, 1) * (int8_max - int8_min) + int8_min
        int8_quantized = np.round(normalized).astype(np.int8)
        return int8_quantized
    
    elif target_precision == 'int4':
        int4_max = 15  
        min_value, max_value = np.min(fp32_vectors), np.max(fp32_vectors)  
        normalized = (fp32_vectors - min_value) / (max_value - min_value) * int4_max
        int4_quantized = np.round(normalized).astype(np.uint8)
        return int4_quantized
    
    elif target_precision == 'binary':
        binary_vectors = (fp32_vectors > 0).astype(np.uint8)
        return binary_vectors
    else:
        TypeError('err')

def float_to_int8(fp32_vectors):
    int8_max = 127  # Max value for signed 8-bit integer
    int8_min = -128  # Min value for signed 8-bit integer
    normalized = np.clip(fp32_vectors, 0, 1) * (int8_max - int8_min) + int8_min
    int8_quantized = np.round(normalized).astype(np.int8)
    return int8_quantized


def float_to_int4(fp32_vectors):
    int4_max = 15  
    min_value, max_value = np.min(fp32_vectors), np.max(fp32_vectors)  
    normalized = (fp32_vectors - min_value) / (max_value - min_value) * int4_max
    int4_quantized = np.round(normalized).astype(np.uint8)
    return int4_quantized


def float_to_binary(fp32_vectors):
    binary_vectors = (fp32_vectors > 0).astype(np.uint8)
    return binary_vectors


# Function to create or load FAISS index
def create_or_load_faiss_index(vectors, index_type, cache_file):
    if os.path.exists(cache_file):
        print(f"Loading {index_type} index from cache...")
        if index_type == 'binary':
            index = faiss.read_index_binary(cache_file)
        else:
            index = faiss.read_index(cache_file)
    else:
        print(f"Creating {index_type} index and caching it...")
        dimension = vectors.shape[1]

        if index_type == 'int8':
            index = faiss.IndexHNSWFlat(dimension, 32)
            int8_vectors = quantize_vector(vectors, 'int8').astype(np.float32)
            index.add(int8_vectors)
        elif index_type == 'int4':
            index = faiss.IndexHNSWFlat(dimension, 32)
            int4_vectors = quantize_vector(vectors, 'int4').astype(np.float32)
            index.add(int4_vectors)
        elif index_type == 'binary':
            binary_vectors = quantize_vector(vectors, 'binary')
            dimension_binary = binary_vectors.shape[1] * 8
            index = faiss.IndexBinaryHNSW(dimension_binary, 32)
            index.add(binary_vectors)
        elif index_type == 'fp32':
            index = faiss.IndexHNSWFlat(dimension, 32)
            vectors = vectors.astype(np.float32)
            index.add(vectors)

        # Save index to cache file
        if index_type == 'binary':
            faiss.write_index_binary(index, cache_file)
        else:
            faiss.write_index(index, cache_file)

    return index


def benchmark_index(index, query_vectors, k=5, precision='fp32'):
    query_vectors = quantize_vector(query_vectors, precision)
    # if isinstance(index, faiss.IndexBinaryHNSW):
    #     query_vectors = float_to_binary(query_vectors)
    # elif isinstance(index, faiss.IndexHNSWFlat):
    #     if precision == 'int8':
    #         query_vectors = float_to_int8(query_vectors).astype(np.float32)
    #     elif precision == 'int4':
    #         query_vectors = float_to_int4(query_vectors).astype(np.float32)
    #     else:
    #         query_vectors = query_vectors.astype(np.float32)

    start_time = time.time()
    _, indices = index.search(query_vectors, k)
    latency = time.time() - start_time
    return latency, indices


def compute_ground_truth(answer_embeddings, query_embeddings, k=5):
    index_fp32 = faiss.IndexFlatL2(answer_embeddings.shape[1])
    index_fp32.add(answer_embeddings)
    _, ground_truth_indices = index_fp32.search(query_embeddings, k)
    return ground_truth_indices


def compute_recall(predicted_indices, ground_truth_indices, k):
    correct = 0
    total_queries = len(ground_truth_indices)

    for i in range(total_queries):
        correct += len(set(predicted_indices[i]).intersection(set(ground_truth_indices[i])))

    recall = correct / (total_queries * k)
    return recall


# Main function to load data, create/load indexes, and run benchmarks
def main():
    question_embeddings, answer_embeddings = load_embeddings()

    int8_index = create_or_load_faiss_index(answer_embeddings, 'int8', 'int8_index.faiss')
    int4_index = create_or_load_faiss_index(answer_embeddings, 'int4', 'int4_index.faiss')
    binary_index = create_or_load_faiss_index(answer_embeddings, 'binary', 'binary_index.faiss')
    fp32_index = create_or_load_faiss_index(answer_embeddings, 'fp32', 'fp32_index.faiss')

    query_vectors_fp32 = question_embeddings[:10]
    int8_queries = quantize_vector(query_vectors_fp32, 'int8')
    int4_queries = quantize_vector(query_vectors_fp32, 'int4')
    # binary_queries = (query_vectors_fp32 > 0).astype(np.uint8)
    binary_queries = quantize_vector(query_vectors_fp32, 'binary')

    ground_truth_indices = compute_ground_truth(answer_embeddings, query_vectors_fp32)

    print("Benchmarking INT8 Index")
    int8_latency, int8_indices = benchmark_index(int8_index, int8_queries, 5, 'int8')
    print(f"INT8 Latency: {int8_latency:.4f} seconds")
    int8_recall = compute_recall(int8_indices, ground_truth_indices, k=5)
    print(f"INT8 Recall: {int8_recall:.4f}")

    print("\nBenchmarking INT4 Index")
    int4_latency, int4_indices = benchmark_index(int4_index, int4_queries, 5, 'int4')
    print(f"INT4 Latency: {int4_latency:.4f} seconds")
    int4_recall = compute_recall(int4_indices, ground_truth_indices, k=5)
    print(f"INT4 Recall: {int4_recall:.4f}")

    print("\nBenchmarking Binary Index")
    binary_latency, binary_indices = benchmark_index(binary_index, binary_queries, 5, 'binary')
    print(f"Binary Latency: {binary_latency:.4f} seconds")
    binary_recall = compute_recall(binary_indices, ground_truth_indices, k=5)
    print(f"Binary Recall: {binary_recall:.4f}")

if __name__ == "__main__":
    main()
