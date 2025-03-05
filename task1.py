import torch
from transformers import AutoTokenizer, AutoModel
from typing import Tuple
from torch import Tensor

def load_sentence_transformer(model_name:str="sentence-transformers/all-MiniLM-L6-v2")-> Tuple[AutoTokenizer, AutoModel]:
    """
    Loads a pre-trained sentence transformer model.
    Args:
        model_name(str): The name of the pre-trained sentence transformer model to load.Defaults to "sentence-transformers/all-MiniLM-L6-v2".
    Returns:
        Tuple[AutoTokenizer,AutoModel]: A tuple containing the tokenizer and the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def encode_sentences(sentences:list[str], tokenizer:AutoTokenizer, model:AutoModel)->Tensor:
    """
    Encodes a list of sentences into fixed-length embeddings.
    Args:
        sentences(List[str]): A list of sentences to encode.
        tokenizer(AutoTokenizer): The tokenizer corresponding to the sentence transformer model.
        model(AutoModel): The pre-trained sentence transformer model.
    Returns:
        Tensor: A tensor containing the sentence embeddings with shape (batch_size, embedding_dim).
    """

    # Tokenization Handling:
    # - Applied padding and truncation to ensure consistent input lengths.
    # - This prevents errors when processing variable-length sentences.
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    
    # Computation Optimization:
    # - Used torch.no_grad() to disable gradients.
    # - This reduces memory usage and speeds up inference.
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Pooling Strategy:
    # - Chose mean pooling instead of relying on the CLS token.
    # - Ensures embeddings capture contextual information from all tokens.
    embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings

if __name__ == "__main__":
    # Load model and tokenizer
    tokenizer, model = load_sentence_transformer()
    
    # Test sentences
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "BERT is a powerful model for NLP tasks."
    ]
    
    # Get sentence embeddings
    embeddings = encode_sentences(sentences, tokenizer, model)
    
    # Print embeddings
    for i, embedding in enumerate(embeddings):
        print(f"Sentence {i+1}: {sentences[i]}")
        print(f"Embedding: {embedding[:5]}...\n")  # Showing only first 5 values
