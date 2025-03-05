import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Tuple

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

class MultiTaskModel(nn.Module):
    """
    A multi-task model that utilizes a transformer backbone for both sentence classification 
    and named entity recognition (NER).
    The model consists of:
    - A shared transformer encoder.
    - A sentence classification head (using [CLS] token representation).
    - A token classification head for Named Entity Recognition (NER).
    
    Attributes:
        transformer(PreTrainedModel): The shared transformer backbone.
        classification_head(nn.Linear): A fully connected layer for sentence classification.
        ner_head(nn.Linear): A fully connected layer for token classification (NER).
    """
    def __init__(self, transformer_model:AutoModel)->None:
        super(MultiTaskModel, self).__init__()
        self.transformer = transformer_model
        hidden_size = transformer_model.config.hidden_size
        
        # Change 1: Added separate task-specific heads
        # - Task A: Sentence Classification (sentence-level output)
        # - Task B: NER (token-level output)
        self.classification_head = nn.Linear(hidden_size, 3)  # Example: 3 classes (adjust as needed)
        self.ner_head = nn.Linear(hidden_size, 5)  # Example: 5 entity types (adjust as needed)

    def forward(self, input_ids:torch.Tensor, attention_mask: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        # Get token-level representations from the transformer
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # Token-level representation
        
        # Change 2: Using [CLS] token for classification
        # - Sentence classification requires a single sentence-level output.
        pooled_output = sequence_output[:, 0, :]  # Extracting CLS token representation
        
        # Task A: Sentence classification prediction
        classification_logits = self.classification_head(pooled_output)
        
        # Change 2: Using raw token-level outputs for NER
        # - Unlike classification, NER requires token-level predictions, so we avoid pooling.
        ner_logits = self.ner_head(sequence_output)

        # Change 3: Returning separate outputs for each task
        # - Task A: Classification logits (sentence-level)
        # - Task B: NER logits (token-level)
        return classification_logits, ner_logits

def encode_sentences(sentences:list[str], tokenizer:AutoModel, model:AutoModel)->Tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes input sentences and passes them through a multi-task learning (MTL) model.
    Args:
        sentences(List[str]): A list of sentences to encode.
        tokenizer(AutoTokenizer): The tokenizer corresponding to the transformer model.
        model(AutoModel): A pre-trained transformer-based multi-task learning model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - classification_logits: Sentence classification logits of shape (batch_size, num_classes).
            - ner_logits: Token-level NER logits of shape (batch_size, seq_length, num_entity_types).
    """
    # Tokenize input sentences
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    
    # Disable gradient calculations for inference (optimization)
    with torch.no_grad():
        classification_logits, ner_logits = model(input_ids, attention_mask)
    
    return classification_logits, ner_logits

if __name__ == "__main__":
    # Load model and tokenizer
    tokenizer, transformer = load_sentence_transformer()
    mtl_model = MultiTaskModel(transformer)
    
    # Test sentences
    sentences = [
        "John Doe lives in New York.",
        "Artificial intelligence is transforming the world.",
        "Elon Musk founded SpaceX."
    ]
    
    # Get outputs for both tasks
    classification_logits, ner_logits = encode_sentences(sentences, tokenizer, mtl_model)
    
    # Print outputs
    print("Sentence Classification Predictions:")
    print(classification_logits)
    print("\nNER Token-Level Predictions:")
    print(ner_logits)
