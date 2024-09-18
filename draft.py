
# def main():
#     question_embeddings, context_embeddings = load_embeddings()

#     int8_embeddings = float_to_int8(context_embeddings)
#     binary_embeddings = float_to_binary(context_embeddings)

#     int8_index = create_faiss_index(int8_embeddings, 'int8')
#     binary_index = create_faiss_index(binary_embeddings, 'binary')

#     # Use the first 10 questions as queries
#     query_vectors_fp32 = question_embeddings[:10]  # FP32 query vectors

#     ground_truth_indices = compute_ground_truth(context_embeddings, query_vectors_fp32, k=5)

#     #int8
#     query_vectors_int8 = float_to_int8(query_vectors_fp32)
#     int8_latency, int8_indices = benchmark_index(int8_index, query_vectors_int8)
#     print(f"INT8 Latency: {int8_latency:.4f} seconds")


#     #binary
#     query_vectors_binary = float_to_binary(query_vectors_fp32)
#     binary_latency, binary_indices = benchmark_index(binary_index, query_vectors_binary)
#     print(f"Binary Latency: {binary_latency:.4f} seconds")


#     # Compute recall by comparing search results to ground truth
#     int8_recall = compute_recall(int8_indices, ground_truth_indices, k=5)
#     binary_recall = compute_recall(binary_indices, ground_truth_indices, k=5)

#     print(f"INT8 Recall: {int8_recall:.4f}")
#     print(f"Binary Recall: {binary_recall:.4f}")


#  # Plotting results
#     precisions = ['INT8', 'Binary']
#     latencies = [int8_latency, binary_latency]
#     recalls = [int8_recall, binary_recall]

#     fig, ax1 = plt.subplots()

#     color = 'tab:blue'
#     ax1.set_xlabel('Precision')
#     ax1.set_ylabel('Latency (seconds)', color=color)
#     ax1.bar(precisions, latencies, color=color, alpha=0.6, label='Latency')
#     ax1.tick_params(axis='y', labelcolor=color)

#     ax2 = ax1.twinx()
#     color = 'tab:red'
#     ax2.set_ylabel('Recall', color=color)
#     ax2.plot(precisions, recalls, color=color, marker='o', label='Recall')
#     ax2.tick_params(axis='y', labelcolor=color)

#     fig.tight_layout()
#     plt.title('INT8 vs Binary Precision Performance')
#     plt.show()

# if __name__ == "__main__":
#     main()

# #1. Encode Text Data
# model = SentenceTransformer('all-MiniLM-L6-v2') #why this model?

# questions = df['question'].tolist()
# contexts = df['context'].tolist()

# question_embeddings = model.encode(questions)
# context_embeddings = model.encode(contexts)

# embeddings = question_embeddings #use either question or context embeddings ???


# #2. Initialize FAISS index
# dimension = embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(embeddings)

# query_vector = embeddings[0] # Define a query vector (e.g., use the first vector from the dataset as a query)


#2. Define benchmarking function
# def benchmark_search(index, query_vector, k=5):
#     start_time = time.time()
#     D, I = index.search(np.array([query_vector]), k)
#     latency = time.time() - start_time
    
#     # Simulated ground truth for recall (replace with actual ground truth if available)
#     ground_truth_indices = np.array([1, 2, 3, 4, 5])  # Example ground truth
#     retrieved_indices = I[0]
#     recall = np.intersect1d(ground_truth_indices, retrieved_indices).size / ground_truth_indices.size
    
#     return D, I, latency, recall




# # Function to build and benchmark index for given precision with HNSW method
# def build_and_benchmark_index_hnsw(precision, query_vector, k=5):
#     dimension = embeddings.shape[1]
    
#     # Initialize HNSW index
#     index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is the number of neighbors for the HNSW graph

#     if precision == 'int8':
#         # Example for INT8 precision
#         index = faiss.IndexHNSWFlat(dimension, 32)
#         index.train(embeddings)
#         index.add(embeddings.astype(np.int8))  # Quantize to INT8
#     # elif precision == 'int4':
#     #     # Example placeholder for INT4 (actual implementation might need additional steps)
#     #     index = faiss.IndexHNSWFlat(dimension, 32)
#     #     index.train(embeddings)
#     #     # Apply INT4 quantization
#     #     # Note: FAISS does not directly support INT4; consider using binarization methods if needed
#     elif precision == 'binary':
#         # Example for binary precision using a binary quantizer
#         binary_index = faiss.IndexBinaryHNSW(dimension)
#         binary_embeddings = np.packbits(embeddings > 0, axis=1)  # Example binary quantization
#         binary_index.add(binary_embeddings)
#         query_vector = np.packbits(query_vector > 0)  # Binary quantization of the query vector
#         index = binary_index
#     else:
#         raise ValueError(f"Unsupported precision: {precision}")
    
#     D, I, latency, recall = benchmark_search(index, query_vector, k)
#     return D, I, latency, recall



# Perform the benchmark
# k = 5
# D, I, latency, recall = benchmark_search(query_vector, k)

# Output results
# print(f"Indices of nearest neighbors: {I}")
# print(f"Distances of nearest neighbors: {D}")
# print(f"Latency (seconds): {latency}")
# print(f"Recall: {recall}")



# Create FAISS index with scalar quantization for INT8
# def create_faiss_int8_index(vectors, nlist=100):
#     dimension = vectors.shape[1]
    
#     # Scalar quantization (SQ) with IVF
#     quantizer = faiss.IndexFlatL2(dimension)  # Coarse quantizer
#     index = faiss.IndexIVFSQ(quantizer, dimension, nlist)  # IVF with SQ (Scalar Quantization)
    
#     # Train the index
#     index.train(vectors)
#     index.add(vectors)
    
#     return index

# Create FAISS index for binary vectors
# def create_faiss_binary_index(vectors):
#     dimension = vectors.shape[1]
#     index = faiss.IndexBinaryHNSW(dimension, 32)  # Binary index using HNSW
#     index.add(vectors)
#     return index



def create_faiss_index(vectors, index_type):
    dimension = vectors.shape[1]

    if index_type == 'int8':
        # Use IVF with Product Quantization for INT8 precision
        quantizer = faiss.IndexFlatL2(dimension)  # Coarse quantizer for IVF
        index = faiss.IndexIVFPQ(quantizer, dimension, 100, 8, 8)  # IVF with PQ
        vectors = vectors.astype(np.float32)  # Ensure vectors are float32
        index.train(vectors)  # Train on vectors
        index.add(vectors)  # Add vectors to index

    elif index_type == 'binary':
        # Convert vectors to binary
        vectors = (vectors > 0).astype(np.uint8)
        index = faiss.IndexBinaryHNSW(dimension, 32)  # HNSW method for binary vectors
        index.add(vectors)

    return index


def create_faiss_index(vectors, index_type):
    dimension = vectors.shape[1]

    # if index_type == 'int8':
    #     # Use HNSW with FlatL2 for INT8 precision
    #     vectors = vectors.astype(np.float32)  # FAISS expects float32 for INT8 indices
    #     index = faiss.IndexHNSWFlat(dimension, 32)  # HNSW method
    #     index.add(vectors)
            # Use IVF with Product Quantization for INT8 precision

    if index_type == 'int8':
        # Use IVF with Product Quantization for INT8 precision
        # vectors = vectors.astype(np.float32)  # FAISS expects float32 for INT8 indices
        print('debugging start')
        quantizer = faiss.IndexFlatL2(dimension)  # Coarse quantizer for IVF
        index = faiss.IndexIVFPQ(quantizer, dimension, 100, 8, 8)  # IVF with PQ
        index.train(vectors)  # Train on vectors
        index.add(vectors)

    elif index_type == 'binary':
        # Use HNSW with binary vectors
        vectors = (vectors > 0).astype(np.uint8)  # Convert FP32 to binary
        index = faiss.IndexBinaryHNSW(dimension, 32)  # HNSW method for binary vectors
        index.add(vectors)
    print('debugging ended')
    

    return index