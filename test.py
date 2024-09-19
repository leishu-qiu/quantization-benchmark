from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import recall_score
from transformers import AutoTokenizer


faiss.omp_set_num_threads(1)

def load_embeddings():
    df = pd.read_parquet("hf://datasets/rachid16/rag_finetuning_data/data/train-00000-of-00001.parquet")
    # print(df.head())

    questions = df['question'][:1000].tolist()
    answers = df['answer'][:1000].tolist()

    model = SentenceTransformer('all-MiniLM-L6-v2')

    question_embeddings = model.encode(questions)
    answers_embeddings = model.encode(answers)
    return np.array(question_embeddings), np.array(answers_embeddings)
    

# def float_to_binary(fp32_vectors):
#         binary_vectors = np.packbits((fp32_vectors > 0).astype(np.uint8), axis=1)
#         return binary_vectors

# def float_to_int8(fp32_vectors):
#     int8_max = 255
#     normalized = np.clip(fp32_vectors, 0, 1) * int8_max
#     int8_quantized = np.round(normalized).astype(np.uint8)
#     return int8_quantized

def float_to_int8(fp32_vectors):
    int8_max = 127  
    int8_min = -128 
    # Normalize the input vectors (assumed to be in the range 0 to 1) to the range -128 to 127
    normalized = np.clip(fp32_vectors, 0, 1) * (int8_max - int8_min) + int8_min
    int8_quantized = np.round(normalized).astype(np.int8)
    return int8_quantized


# def float_to_int4(fp32_vectors):
#     int4_max = 15  
#     normalized = np.clip(fp32_vectors, 0, 1) * int4_max
#     int4_quantized = np.round(normalized).astype(np.uint8)
#     # num_bits = 4
#     # int4_quantized_packed = np.packbits(int4_quantized.reshape(-1, int4_quantized.shape[1] // num_bits * num_bits), axis=1)
#     # return int4_quantized_packed
#     return int4_quantized
def float_to_int4(fp32_vectors):
    int4_max = 15  
    min_value, max_value = np.min(fp32_vectors), np.max(fp32_vectors)  # Get min/max for range normalization
    normalized = (fp32_vectors - min_value) / (max_value - min_value) * int4_max
    int4_quantized = np.round(normalized).astype(np.uint8)  # Ensure quantized to integer values
    return int4_quantized



def float_to_binary(fp32_vectors):
    # Ensure binary vectors are a multiple of 8 bits
    binary_vectors = (fp32_vectors > 0).astype(np.uint8)
    return binary_vectors

def create_faiss_index(vectors, index_type):
    dimension = vectors.shape[1]

    if index_type == 'int8':
        index = faiss.IndexHNSWFlat(dimension, 32)  # HNSW for INT8
        int8_vectors = float_to_int8(vectors).astype(np.float32)  
        index.add(int8_vectors) 
    elif index_type == 'int4':
        index = faiss.IndexHNSWFlat(dimension, 32)  # HNSW for INT8
        int4_vectors = float_to_int4(vectors).astype(np.float32)  
        index.add(int4_vectors) 
    # elif index_type == 'int4':
    #     # INT4 handling - no direct support in FAISS, but for benchmarking, use similar processing
    #     int4_vectors = float_to_int4(vectors)  # Pack int4 vectors
    #     # You would need a custom index or workaround here; for now, using a placeholder
    #     index = faiss.IndexHNSWFlat(dimension, 32)  # Placeholder, not true INT4
    #     index.add(np.float32(int4_vectors))  # Add placeholder vectors for demonstration

    elif index_type == 'binary':
        binary_vectors = float_to_binary(vectors)  # Ensure this function converts properly
        dimension_binary = binary_vectors.shape[1] * 8  # Convert dimension for binary indexing
        index = faiss.IndexBinaryHNSW(dimension_binary, 32)  # Binary HNSW index
        index.add(binary_vectors)  # Add binary vectors to the index

    elif index_type == 'fp32':
        # Standard FP32 index using HNSW
        index = faiss.IndexHNSWFlat(dimension, 32)  # HNSW index for FP32 precision
        vectors = vectors.astype(np.float32)  # Ensure vectors are float32
        index.add(vectors)  # Add vectors to the index

    return index


def benchmark_index(index, query_vectors, k=5, precision='fp32'):
    if isinstance(index, faiss.IndexBinaryHNSW):
        query_vectors = float_to_binary(query_vectors)
    elif isinstance(index, faiss.IndexHNSWFlat):
        if precision == 'int8':  
            query_vectors = float_to_int8(query_vectors).astype(np.float32)
        elif precision == 'int4':
            query_vectors = float_to_int4(query_vectors).astype(np.float32)
        else:
            query_vectors = query_vectors.astype(np.float32)  

    start_time = time.time()
    _, indices = index.search(query_vectors, k)
    latency = time.time() - start_time
    return latency, indices


def compute_ground_truth(answer_embeddings, query_embeddings, k=5):
    # Create an exact L2 index for the full-precision FP32 context embeddings
    index_fp32 = faiss.IndexFlatL2(answer_embeddings.shape[1])
    index_fp32.add(answer_embeddings)

    # Perform search with FP32 query embeddings
    _, ground_truth_indices = index_fp32.search(query_embeddings, k)
    return ground_truth_indices


def compute_recall(predicted_indices, ground_truth_indices, k):
    correct = 0
    total_queries = len(ground_truth_indices)

    for i in range(total_queries):
        correct += len(set(predicted_indices[i]).intersection(set(ground_truth_indices[i])))

    recall = correct / (total_queries * k)
    return recall

# Helper function to plot latency and recall
def plot_results(precisions, latencies, recalls):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Precision')
    ax1.set_ylabel('Latency (seconds)', color=color)
    ax1.bar(precisions, latencies, color=color, alpha=0.6, label='Latency')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Recall', color=color)
    ax2.plot(precisions, recalls, color=color, marker='o', label='Recall')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('INT8 vs Binary vs FP32 Precision Performance')
    plt.show()


def main():
    question_embeddings, answer_embeddings = load_embeddings()
    # question_embeddings = question_embeddings[:500]
    # answer_embeddings = answer_embeddings[:500]

    int8_index = create_faiss_index(answer_embeddings, 'int8')
    int4_index = create_faiss_index(answer_embeddings, 'int4')
    binary_index = create_faiss_index(answer_embeddings, 'binary')
    fp32_index = create_faiss_index(answer_embeddings, 'fp32')

    query_vectors_fp32 = question_embeddings[:10]  # for ground truth
    int8_queries = float_to_int8(query_vectors_fp32)
    int4_queries = float_to_int4(query_vectors_fp32)
    binary_queries = (query_vectors_fp32 > 0).astype(np.uint8) 

    # fp32_queries = query_vectors_fp32.astype(np.float32)  # Ensure queries are FP32

    ground_truth_indices = compute_ground_truth(answer_embeddings, query_vectors_fp32)

    print("Benchmarking INT8 Index")
    int8_latency, int8_indices = benchmark_index(int8_index, int8_queries, 5, 'in8')
    print(f"INT8 Latency: {int8_latency:.4f} seconds")
    int8_recall = compute_recall(int8_indices, ground_truth_indices, k=5)
    print(f"INT8 Recall: {int8_recall:.4f}")

    print("\nBenchmarking INT4 Index")
    int4_latency, int4_indices = benchmark_index(int4_index, int4_queries, 5, 'int4')
    print(f"INT4 Latency: {int4_latency:.4f} seconds")
    int4_recall = compute_recall(int4_indices, ground_truth_indices, k=5)
    print(f"INT4 Recall: {int4_recall:.4f}")

    print("\nBenchmarking Binary Index")
    binary_latency, binary_indices = benchmark_index(binary_index, binary_queries)
    print(f"Binary Latency: {binary_latency:.4f} seconds")
    binary_recall = compute_recall(binary_indices, ground_truth_indices, k=5)
    print(f"Binary Recall: {binary_recall:.4f}")

    # print("\nBenchmarking FP32 Index")
    # fp32_latency, fp32_indices = benchmark_index(fp32_index, fp32_queries)
    # print(f"FP32 Latency: {fp32_latency:.4f} seconds")
    # fp32_recall = compute_recall(fp32_indices, ground_truth_indices, k=5)
    # print(f"FP32 Recall: {fp32_recall:.4f}")

    # Plot results
    precisions = ['INT8', 'INT4' 'Binary']
    latencies = [int8_latency, int4_latency, binary_latency]
    recalls = [int8_recall, int4_recall, binary_recall]

    # plot_results(precisions, latencies, recalls)

if __name__ == "__main__":
    main()
