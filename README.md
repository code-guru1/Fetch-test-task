# Multi-Task Learning with Transformers

This repository implements a multi-task learning (MTL) framework using transformer models. It covers sentence embedding generation, classification, named entity recognition (NER), and a structured training loop.

## Project Structure

```

├── task1.py         # Sentence Transformer implementation
├── task2.py         # Multi-Task Learning (MTL) model for classification & NER
├── task3.docx       # Discussion on training strategies (Task 3)
├── task4.py         # Training loop for MTL model
├── task4.docx       # Explanation for Task 4
├── requirements.txt # Dependencies for running the project
├── README.md        # Project documentation
```

## Tasks Overview

### Task 1: Sentence Transformer
- Loads a pre-trained transformer.
- Encodes sentences into fixed-length embeddings using mean pooling.

### Task 2: Multi-Task Learning Model
- Adds separate heads for classification and NER.
- Uses CLS token for classification and token-level outputs for NER.

### Task 3: Training Considerations
- Discusses freezing/unfreezing layers in different training scenarios.
- Covers transfer learning strategies.

### Task 4: Training Loop Implementation
- Implements an adaptive training loop for classification and/or NER.
- Optimizes only relevant parameters based on selected tasks.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/code-guru1/Fetch-test-task
cd Fetch-test-task
pip install -r requirements.txt
```

## Usage

Run individual scripts as needed:

```bash
python task1.py  # Generate sentence embeddings
python task2.py  # Run MTL model on test inputs
python task4.py  # Execute the training loop
```

## Notes
- The model uses dummy data for demonstration.
- Explanations for Task 3 and Task 4 are in the `.docx` files.

## License
This project is open-source and free to use.
