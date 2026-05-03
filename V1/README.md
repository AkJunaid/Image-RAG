# Image-to-Image RAG for Fashion

This project implements an improved Image-to-Image Retrieval-Augmented Generation (RAG) pipeline tailored for fashion images. It combines visual similarity searches with explicit attribute extraction (color, category, style) to provide highly relevant search results.

## Features

* **Zero-Shot Metadata Extraction**: Leverages OpenAI's CLIP model to automatically classify the dominant color, category, and style of clothing items without requiring labeled training datasets.
* **Hybrid Image Search**: Combines pure visual similarity (CLIP embeddings) with metadata scoring to rank results. Offers hard and soft filters (e.g., search for a similar dress, but strictly enforce the color "red").
* **Textual Natural Language Search**: Supports searching the indexed fashion database using natural language queries like "girl wearing red dress".
* **Fast Vector Retrieval**: Uses FAISS (Facebook AI Similarity Search) to handle fast nearest-neighbor lookups.

## Project Structure

* `v1.ipynb`, `v2.ipynb`, `image-rag.ipynb`: Jupyter notebooks containing the pipeline code. The latest pipeline with metadata-aware searching is in `v2.ipynb`.
* `train_images/`: Directory containing the dataset of fashion images to index.
* `test_images/`: Directory for query images.

## Usage

1. Place your dataset images (PNG, JPG, WEBP) inside the `train_images/` directory.
2. Open `v2.ipynb` in your notebook editor.
3. Run the cells sequentially to install dependencies, load the CLIP model, extract metadata, and build the FAISS index.
4. Save the generated FAISS index and metadata to disk using the provided cell to avoid re-encoding on restart.
5. Use the `search_by_image()` or `search_by_text()` functions provided in the notebook to query your dataset.

## Architecture Details

The pipeline works in two stages:
1. **Indexing**: Each dataset image is passed through CLIP to generate a visual embedding. Concurrently, CLIP zero-shot classification infers attributes (color, style, category) for each image. Embeddings are saved to a FAISS index (`faiss.index`), and metadata is saved to `metadata.json`.
2. **Retrieval**: When queried with an image or text, the system fetches an initial over-sized pool of visually similar candidates from FAISS. It then applies hard-filters based on metadata constraints (e.g., matching colors) and re-ranks the candidates using a combined visual and metadata score to yield the final output.