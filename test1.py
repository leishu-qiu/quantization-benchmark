import numpy as np
import faiss
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import recall_score
from sentence_transformers import SentenceTransformer


faiss.omp_set_num_threads(1)

def load_embeddings():
    #dataset1
    df = pd.read_parquet("hf://datasets/rachid16/rag_finetuning_data/data/train-00000-of-00001.parquet")
    #dataset2
    # df = pd.read_parquet("hf://datasets/philschmid/finanical-rag-embedding-dataset/data/train-00000-of-00001.parquet")
    questions = df['question'].tolist()
    answers = df['answer'].tolist()
    # answers = df['context'][:10000].tolist()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    question_embeddings = model.encode(questions)
    answers_embeddings = model.encode(answers)
    return np.array(question_embeddings), np.array(answers_embeddings)


def quantize_vector(fp32_vectors, target_precision=''):
    # fp32_vectors = fp32_vectors.astype(np.float32)
    if target_precision == 'int8':
        scale_factor = 127.0 / np.max(np.abs(fp32_vectors)) 
        int8_quantized = np.clip(np.round(fp32_vectors * scale_factor), -128, 127).astype(np.int8)
        return int8_quantized
    
    elif target_precision == 'int4':
        scale_factor = 7.0 / np.max(np.abs(fp32_vectors))
        int4_quantized = np.clip(np.round(fp32_vectors * scale_factor), -8, 7).astype(np.int8)
        # int4_as_int8 = int4_quantized.astype(np.int8)
        # return int4_as_int8
        return int4_quantized
        
    elif target_precision == 'binary':
        binary_vectors = (fp32_vectors > 0).astype(np.uint8)
        return binary_vectors
    
    else:
        TypeError('err')


def create_faiss_index(vectors, index_type):
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

    return index


def benchmark_index(index, query_vectors, k=5, precision=''):
    query_vectors = quantize_vector(query_vectors, precision)
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


def plot_results(results):
    plt.figure(figsize=(10, 6))

    # Extract latency and recall for each method
    methods = ['INT8', 'INT4', 'Binary']
    latencies = [result['latency'] for result in results]
    recalls = [result['recall'] for result in results]

    # Create the plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Latency (s)', color=color)
    ax1.bar(methods, latencies, color=color, alpha=0.6, label='Latency')
    ax1.tick_params(axis='y', labelcolor=color)

    # Instantiate a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Recall', color=color)
    ax2.plot(methods, recalls, color=color, marker='o', label='Recall')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # Ensures proper spacing between plots
    plt.title('Benchmark Results: Latency and Recall Comparison')
    plt.show()


def main():
    question_embeddings, answer_embeddings = load_embeddings()

    int8_index = create_faiss_index(answer_embeddings, 'int8')
    int4_index = create_faiss_index(answer_embeddings, 'int4')
    binary_index = create_faiss_index(answer_embeddings, 'binary')
    # fp32_index = create_or_load_faiss_index(answer_embeddings, 'fp32', 'fp32_index.faiss')

    random_queries = np.random.permutation(len(question_embeddings))
    num_queries = 100
    selected_indices = random_queries[:num_queries]
    query_vectors_fp32 = question_embeddings[selected_indices]

    int8_queries = quantize_vector(query_vectors_fp32, 'int8')
    int4_queries = quantize_vector(query_vectors_fp32, 'int4')
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

    # results = []

    # print("Benchmarking INT8 Index")
    # int8_latency, int8_indices = benchmark_index(int8_index, query_vectors_fp32, 5, 'int8')
    # int8_recall = compute_recall(int8_indices, ground_truth_indices, k=5)
    # results.append({'method': 'INT8', 'latency': int8_latency, 'recall': int8_recall})

    # print("Benchmarking INT4 Index")
    # int4_latency, int4_indices = benchmark_index(int4_index, query_vectors_fp32, 5, 'int4')
    # int4_recall = compute_recall(int4_indices, ground_truth_indices, k=5)
    # results.append({'method': 'INT4', 'latency': int4_latency, 'recall': int4_recall})

    # print("Benchmarking Binary Index")
    # binary_latency, binary_indices = benchmark_index(binary_index, query_vectors_fp32, 5, 'binary')
    # binary_recall = compute_recall(binary_indices, ground_truth_indices, k=5)
    # results.append({'method': 'Binary', 'latency': binary_latency, 'recall': binary_recall})

    # plot_results(results)

if __name__ == "__main__":
    main()
