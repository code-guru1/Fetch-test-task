import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from typing import Tuple,Optional

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
    def __init__(self, transformer_model:AutoModel)->None:
        super(MultiTaskModel, self).__init__()
        self.transformer = transformer_model
        hidden_size = transformer_model.config.hidden_size
        

        for param in self.transformer.parameters():
            param.requires_grad = False  # Prevents BERT from updating

        # Task A: Sentence Classification Head
        self.classification_head = nn.Linear(hidden_size, 3)  # Example: 3 classes
        
        # Task B: NER Head (Token-Level Classification)
        self.ner_head = nn.Linear(hidden_size, 5)  # Example: 5 entity types
    
    def forward(self, input_ids:torch.Tensor, attention_mask:torch.Tensor, task:str="both"):
        """Handles forward pass based on the task type."""
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # Token-level representation
        pooled_output = sequence_output[:, 0, :]  # CLS token for classification

        classification_logits = None
        ner_logits = None
        
        if task in ["both", "classification"]:
            classification_logits = self.classification_head(pooled_output)
        
        if task in ["both", "ner"]:
            ner_logits = self.ner_head(sequence_output)
        
        return classification_logits, ner_logits
    
def train_multi_task_model(model:MultiTaskModel, train_data:list[dict[str,torch.tensor]], epochs:int=3, learning_rate:float=2e-5, train_classification:bool=True, train_ner:bool=True):
    """
    Trains the multi-task learning model selectively based on the task.
    Args:
        model(MultiTaskModel): The multi-task model to be trained.
        train_data(list[dict[str, torch.Tensor]]): A list of dictionaries containing training batches.
        epochs(int,optional): Number of training epochs. Defaults to 3.
        learning_rate(float,optional): Learning rate for the optimizer. Defaults to 2e-5.
        train_classification(bool,optional): Whether to train the classification head. Defaults to True.
        train_ner(bool,optional): Whether to train the NER head. Defaults to True.
    Returns:
        None
    """
    # Set optimizer to update only required parameters
    params_to_optimize = []
    if train_classification:
        params_to_optimize += list(model.classification_head.parameters())
    if train_ner:
        params_to_optimize += list(model.ner_head.parameters())

    optimizer = optim.AdamW(params_to_optimize, lr=learning_rate)
    classification_loss_fn = nn.CrossEntropyLoss()
    ner_loss_fn = nn.CrossEntropyLoss()

    model.train()
    
    for epoch in range(epochs):
        total_classification_loss = 0
        total_ner_loss = 0
        num_batches = len(train_data)

        for batch in train_data:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            classification_labels = batch["classification_labels"]
            ner_labels = batch["ner_labels"]
            
            optimizer.zero_grad()
            
            task_type = "both"
            if train_classification and not train_ner:
                task_type = "classification"
            elif train_ner and not train_classification:
                task_type = "ner"
            
            classification_logits, ner_logits = model(input_ids, attention_mask, task=task_type)
            
            loss = 0
            if train_classification:
                classification_loss = classification_loss_fn(classification_logits, classification_labels)
                total_classification_loss += classification_loss.item()
                loss += classification_loss

            if train_ner:
                ner_loss = ner_loss_fn(ner_logits.permute(0, 2, 1), ner_labels) 
                total_ner_loss += ner_loss.item()
                loss += ner_loss

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}:")
        if train_classification:
            print(f"  Classification Loss: {total_classification_loss / num_batches:.4f}")
        if train_ner:
            print(f"  NER Loss: {total_ner_loss / num_batches:.4f}")

if __name__ == "__main__":

    tokenizer, transformer = load_sentence_transformer()
    mtl_model = MultiTaskModel(transformer)
    
    # Dummy training data
    batch_size = 2
    seq_length = 10
    
    train_data = [
        {
            "input_ids": torch.randint(0, 30522, (batch_size, seq_length)),  
            "attention_mask": torch.ones((batch_size, seq_length)),
            "classification_labels": torch.randint(0, 3, (batch_size,)),  
            "ner_labels": torch.randint(0, 5, (batch_size, seq_length)),  
        }
    ] * 10 

    # Train model with classification only
    train_multi_task_model(mtl_model, train_data, epochs=10, train_classification=True, train_ner=False)

    # Train model with NER only
    train_multi_task_model(mtl_model, train_data, epochs=10, train_classification=False, train_ner=True)

    # Train model with both classification and NER
    train_multi_task_model(mtl_model, train_data, epochs=10, train_classification=True, train_ner=True)
