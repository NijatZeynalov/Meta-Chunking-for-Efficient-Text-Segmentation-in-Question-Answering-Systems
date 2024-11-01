# Meta-Chunking for Efficient Text Segmentation in Question Answering Systems

## Overview
This project implements the **Meta-Chunking** framework for efficient text segmentation in knowledge-intensive tasks such as **question answering (QA)**. The goal is to develop a system that optimally chunks long documents by leveraging linguistic and logical connections between sentences, thus improving the performance of large language models (LLMs) in **retrieval-augmented generation (RAG)** systems.

## Key Features
- **Margin Sampling and Perplexity Chunking**:
  - **Margin Sampling**: Uses LLMs for binary classification, segmenting text based on probability differences between consecutive sentences.
  - **Perplexity Chunking**: Identifies chunk boundaries using changes in perplexity across sentences.
- **Dynamic Chunk Merging**:
  - Merges chunks dynamically based on a maximum length threshold, ensuring manageable chunk sizes.

## Getting Started
### Requirements
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Usage
To run the question answering script with Meta-Chunking:

```bash
python examples/run_qa.py --document path/to/document.txt --question "Your question here" --chunker margin
```

Arguments:
- `--document`: Path to the text file containing the document to be processed.
- `--question`: The question to be answered based on the document.
- `--chunker`: Chunking strategy to use (`margin` or `perplexity`). Default is `margin`.

