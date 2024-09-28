import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CosineEmbeddingLoss
from spellchecker import SpellChecker
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import logging
from tqdm import tqdm

# LyricsMidiDataset class for processing lyrics and associated MIDI features
class LyricsMidiDataset(Dataset):
    def __init__(self, lyrics_df, max_seq_length=32, tokenizer=None, word2idx=None,
                 embedding_matrix=None, features_by_song=None, compressed_instruments_features_by_song=None,
                 feature_min=None, feature_max=None, instrument_feature_min=None, instrument_feature_max=None):
        """
        Initializes the dataset with the provided parameters.
        
        Args:
        - lyrics_df: DataFrame containing the lyrics and song information.
        - max_seq_length: The maximum length of token sequences (for chunking lyrics).
        - tokenizer: Tokenizer to encode the lyrics.
        - word2idx: A dictionary that maps words/tokens to their respective indices.
        - embedding_matrix: Pre-trained word embeddings.
        - features_by_song: Dictionary of regular features by song.
        - compressed_instruments_features_by_song: Dictionary of instrument features by song.
        - feature_min, feature_max: Min and max values for regular feature normalization.
        - instrument_feature_min, instrument_feature_max: Min and max values for instrument feature normalization.
        """
        self.lyrics_df = lyrics_df
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.word2idx = word2idx
        self.embedding_matrix = embedding_matrix
        self.features_by_song = features_by_song
        self.compressed_instruments_features_by_song = compressed_instruments_features_by_song
        self.feature_min = feature_min
        self.feature_max = feature_max
        self.instrument_feature_min = instrument_feature_min
        self.instrument_feature_max = instrument_feature_max

    def __len__(self):
        """Returns the total number of songs in the dataset."""
        return len(self.lyrics_df)

    def __getitem__(self, idx):
        """Retrieves a specific song by index, processes its lyrics and features."""
        
        # Retrieve the row corresponding to the song at index `idx`
        row = self.lyrics_df.iloc[idx]
        
        # Generate a unique song ID using the artist and song name
        song_id = f"{row['Artist']}_{row['Song Name']}"

        # Tokenize the lyrics using the provided tokenizer
        tokens = self.tokenizer.encode(row['Lyrics']).tokens

        # Determine the number of full chunks (of length max_seq_length) that can be made from the tokens
        num_full_chunks = len(tokens) // self.max_seq_length

        # Calculate the start indices for each chunk
        starts = list(range(0, num_full_chunks * self.max_seq_length, self.max_seq_length))
        
        # If there's a remaining partial chunk, include its start index
        if len(tokens) % self.max_seq_length != 0:
            starts.append(num_full_chunks * self.max_seq_length)

        sequence_chunks = []
        
        # Iterate through each start index to create chunks of lyrics, features, and targets
        for start in starts:
            end = start + self.max_seq_length
            
            # Get the current chunk of tokens
            chunk_tokens = tokens[start:end]
            
            # Convert the tokens to their corresponding word indices (unknown words become `<unk>`)
            chunk_ids = [self.word2idx.get(token, self.word2idx['<unk>']) for token in chunk_tokens]

            # Shift the chunk to generate the target sequence (next token prediction)
            target_ids = chunk_ids[1:] + [self.word2idx['<pad>']]  # Target is shifted and padded at the end

            # Fetch the pre-trained embeddings for each token in the chunk
            embeddings = [self.embedding_matrix[self.word2idx.get(token, self.word2idx['<unk>'])] for token in chunk_tokens]
            
            # Retrieve song-specific regular and instrument features for the chunk
            features_chunk = self.features_by_song.get(song_id, np.zeros((0, 143)))[start:end]
            instruments_chunk = self.compressed_instruments_features_by_song.get(song_id, np.zeros((0, 384)))[start:end]

            # Normalize features using Min-Max scaling
            features_chunk = (features_chunk - self.feature_min) / (self.feature_max - self.feature_min)
            instruments_chunk = (instruments_chunk - self.instrument_feature_min) / (self.instrument_feature_max - self.instrument_feature_min)

            # Handle any NaN or infinite values that may occur during normalization
            if np.isnan(features_chunk).any() or np.isinf(features_chunk).any():
                print(f"NaN or Inf found in features_chunk for song_id: {song_id}")
            if np.isnan(instruments_chunk).any() or np.isinf(instruments_chunk).any():
                print(f"NaN or Inf found in instruments_chunk for song_id: {song_id}")

            # Pad all sequences to ensure they are the same length (max_seq_length)
            embeddings = np.pad(embeddings, ((0, self.max_seq_length - len(embeddings)), (0, 0)), 'constant')
            features_chunk = np.pad(features_chunk, ((0, self.max_seq_length - len(features_chunk)), (0, 0)), 'constant')
            instruments_chunk = np.pad(instruments_chunk, ((0, self.max_seq_length - len(instruments_chunk)), (0, 0)), 'constant')

            # Concatenate the embeddings, regular features, and instrument features into a single tensor
            chunk_tensor = torch.tensor(np.concatenate([embeddings, features_chunk, instruments_chunk], axis=1), dtype=torch.float32)
            
            # Create a tensor for the target sequence, padded to max_seq_length
            target_tensor = torch.tensor(np.pad(target_ids, (0, self.max_seq_length - len(target_ids)), 'constant',
                                                 constant_values=self.word2idx['<pad>']), dtype=torch.long)

            # Append the chunk (input and target) to the list of sequence chunks
            sequence_chunks.append((chunk_tensor, target_tensor))

        return sequence_chunks

# Custom collate function for batching in DataLoader
def custom_collate_fn(batch):
    """
    Custom function for collating a batch of data, stacking the input feature tensors 
    and target tensors into batches.
    
    Args:
    - batch: List of (feature tensor, target tensor) pairs returned by the Dataset.
    
    Returns:
    - batch_features: A tensor containing the batched input features.
    - batch_targets: A tensor containing the batched target sequences.
    """
    
    # Flatten the nested batch (list of chunks) into a single list of feature and target tensors
    feature_list, target_list = zip(*[item for sublist in batch for item in sublist])
    
    # Stack feature tensors and target tensors into batches
    batch_features = torch.stack(feature_list)
    batch_targets = torch.stack(target_list)

    return batch_features, batch_targets

class CombinedLoss(nn.Module):
    def __init__(self, embedding_matrix, vocab_size, alpha=0.5):
        """
        Initializes the CombinedLoss module.
        
        Args:
        - embedding_matrix: The pre-trained embedding matrix for the vocabulary (used for cosine loss).
        - vocab_size: The size of the vocabulary (used in cross-entropy loss).
        - alpha: Weight parameter to balance cross-entropy loss and cosine loss.
        """
        super(CombinedLoss, self).__init__()
        
        # Define cross-entropy loss for classification (logits to correct word index)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # CosineEmbeddingLoss computes the cosine similarity between predicted and actual word embeddings
        self.cosine_loss = CosineEmbeddingLoss()

        # The embedding matrix, used to look up embeddings for the predicted and actual words
        self.embedding_matrix = embedding_matrix
        
        # Vocabulary size, used to reshape outputs for cross-entropy loss
        self.vocab_size = vocab_size
        
        # Alpha controls the weight of the cosine similarity loss in the total loss
        self.alpha = alpha

    def forward(self, outputs, targets):
        # Reshape outputs and targets for cross-entropy loss computation
        # Cross-Entropy expects shape [batch_size * seq_len, vocab_size] for logits and [batch_size * seq_len] for targets
        ce_loss = self.cross_entropy_loss(outputs.view(-1, self.vocab_size), targets.view(-1))
        
        # Cosine Similarity Loss
        # Get the predicted word indices by taking the argmax of the outputs across the vocabulary dimension
        predicted_indices = outputs.argmax(dim=-1).view(-1).cpu()  # Move indices to CPU
        
        # Get the actual word indices (target words)
        actual_indices = targets.view(-1).cpu()  # Move indices to CPU
        
        # Retrieve the embeddings for the predicted and actual words from the embedding matrix
        predicted_embeddings = self.embedding_matrix[predicted_indices]
        actual_embeddings = self.embedding_matrix[actual_indices]
        
        # Create label tensor for cosine embedding loss (1 for positive similarity)
        # CosineEmbeddingLoss expects a label of 1 to indicate that the vectors should be similar
        cosine_labels = torch.ones(predicted_embeddings.size(0)).to(outputs.device)
        
        # Compute the cosine similarity loss between the predicted and actual word embeddings
        cos_loss = self.cosine_loss(predicted_embeddings, actual_embeddings, cosine_labels)
        
        # Combine the cross-entropy loss and the cosine embedding loss, weighted by alpha
        total_loss = ce_loss + self.alpha * cos_loss
        
        return total_loss
    
# Simple Lyrics Generator using an LSTM-based architecture
class LyricsGenerator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=2, dropout=0.3):
        """
        Initialize the LyricsGenerator model.
        
        Args:
        - embedding_dim: The size of the input word embeddings.
        - hidden_dim: The size of the hidden state in the LSTM.
        - vocab_size: The size of the vocabulary (number of possible output words).
        - num_layers: Number of layers in the LSTM.
        - dropout: Dropout probability for regularization.
        """
        super(LyricsGenerator, self).__init__()
        
        # LSTM layer: Takes word embeddings as input and outputs hidden states
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, 
                            bidirectional=False, dropout=dropout)

        # Fully connected layer: First transformation of the LSTM output
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layer: Outputs the final predictions for each time step
        self.fc_2 = nn.Linear(hidden_dim, vocab_size)

        # Dropout layer for regularization to prevent overfitting
        self.dropout = nn.Dropout(dropout)

        # Layer normalization to normalize the output of the LSTM
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
        - x: Input tensor (batch_size, seq_len, embedding_dim).
        
        Returns:
        - output: Predicted logits for each word in the sequence.
        """
        # Pass input through the LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Normalize the LSTM output and apply dropout
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        # Pass through the fully connected layers
        output = torch.relu(self.fc_1(lstm_out))  # Apply ReLU activation
        output = self.fc_2(output)  # Final prediction logits for each word in the vocabulary
        
        return output

# Attention mechanism to focus on different parts of the sequence
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        """
        Initialize the Attention module.
        
        Args:
        - hidden_dim: The size of the LSTM hidden states (input dimension for attention).
        """
        super(Attention, self).__init__()
        
        # Linear layer to compute attention scores from LSTM output and hidden state
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Linear layer to compute the final attention weights (no bias)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
        # Placeholder to store the attention weights for visualization
        self.attention_weights = None
    
    def forward(self, lstm_out, hidden, mask=None):
        """
        Forward pass for computing attention weights.
        
        Args:
        - lstm_out: LSTM output for the entire sequence (batch_size, seq_len, hidden_dim).
        - hidden: Current hidden state (batch_size, hidden_dim).
        - mask: Optional mask to ignore certain time steps (for future steps in sequence generation).
        
        Returns:
        - attn_weights: Attention weights for each time step.
        """
        timestep = lstm_out.size(1)  # Get the length of the sequence (number of time steps)
        
        # Repeat the hidden state across all time steps to compare against the LSTM output
        h = hidden.unsqueeze(1).repeat(1, timestep, 1)  # (batch_size, seq_len, hidden_dim)

        # Concatenate the LSTM output and the hidden state to compute attention scores
        combined = torch.cat((lstm_out, h), dim=2)  # (batch_size, seq_len, hidden_dim * 2)

        # Pass through a linear layer and apply a tanh activation function
        attn_weights = torch.tanh(self.attn(combined))

        # Compute attention weights using a linear layer and remove the last dimension
        attn_weights = self.v(attn_weights).squeeze(2)  # (batch_size, seq_len)

        # Apply the mask to ignore certain time steps (e.g., future steps in autoregressive generation)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e10)  # Large negative values for masked positions

        # Apply softmax to get attention weights (probabilities over time steps)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Store the attention weights for later visualization
        self.attention_weights = attn_weights.detach().cpu().numpy()

        return attn_weights

# Advanced Lyrics Generator with LSTM and Attention Mechanism
class LyricsGeneratorAdvanced(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=2, dropout=0.1):
        """
        Initialize the advanced LyricsGenerator model with attention mechanism.
        
        Args:
        - embedding_dim: The size of the input word embeddings.
        - hidden_dim: The size of the hidden state in the LSTM.
        - vocab_size: The size of the vocabulary (number of possible output words).
        - num_layers: Number of layers in the LSTM.
        - dropout: Dropout probability for regularization.
        """
        super(LyricsGeneratorAdvanced, self).__init__()
        
        # LSTM layer for processing input sequences
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, 
                            bidirectional=False, dropout=dropout)
        
        # Attention mechanism to focus on important parts of the sequence
        self.attention = Attention(hidden_dim)
        
        # Fully connected layer to generate output logits
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        # Layer normalization for stabilizing the training process
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Initialize the LSTM weights
        self.init_weights()

        # Placeholder to store attention weights for each step
        self.attention_weights_per_step = []

    def init_weights(self):
        """
        Initialize the weights of the LSTM with proper initialization.
        """
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights
                nn.init.xavier_uniform_(param.data)  # Xavier initialization for input weights
            elif 'weight_hh' in name:  # Hidden-hidden weights
                nn.init.orthogonal_(param.data)  # Orthogonal initialization for hidden weights
            elif 'bias' in name:  # Bias terms
                param.data.fill_(0)  # Initialize bias to zero

    def forward(self, x):
        """
        Forward pass through the model with attention mechanism.
        
        Args:
        - x: Input tensor (batch_size, seq_len, embedding_dim).
        
        Returns:
        - outputs: Logits for each time step (batch_size, seq_len, vocab_size).
        """
        # Pass input through the LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Normalize and apply dropout to LSTM output
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)

        # Initialize a list to store the output logits for each time step
        outputs = []
        
        # Get the length of the sequence and batch size
        seq_len = lstm_out.size(1)
        batch_size = lstm_out.size(0)
        
        # Create a mask to ignore future time steps (used for autoregressive generation)
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(x.device)

        # Loop over each time step in the sequence
        for t in range(seq_len):
            # Use the hidden state corresponding to the current time step
            hidden = lstm_out[:, t, :]  # (batch_size, hidden_dim)
            
            # Compute the attention weights for the current hidden state
            attn_weights = self.attention(lstm_out, hidden, mask[t])
            
            # Store the attention weights for visualization later
            self.attention_weights_per_step.append(self.attention.attention_weights)
            
            # Compute the context vector as a weighted sum of the LSTM outputs
            context_vector = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch_size, hidden_dim)
            
            # Pass the context vector through the fully connected layer to get output logits
            output = self.fc(context_vector)
            outputs.append(output.unsqueeze(1))  # Add output for this time step to the list
        
        # Concatenate the outputs for all time steps into a single tensor
        outputs = torch.cat(outputs, dim=1)  # (batch_size, seq_len, vocab_size)
        
        return outputs
    
def train_model(
    model,
    dataloader,
    validation_dataloader,
    criterion,
    optimizer,
    num_epochs,
    patience,
    vocab_size,
    model_save_path,
    results_save_path,
    device=None
):
    """
    Train the model with early stopping, logging, and checkpoint saving. Includes validation after each epoch.

    Parameters:
    model (nn.Module): The model to be trained.
    dataloader (DataLoader): DataLoader for training data.
    validation_dataloader (DataLoader): DataLoader for validation data.
    criterion (nn.Module): Loss function.
    optimizer (optim.Optimizer): Optimizer.
    num_epochs (int): Number of epochs to train.
    patience (int): Patience for early stopping.
    vocab_size (int): Size of the vocabulary.
    model_save_path (str): Path to save the best model.
    results_save_path (str): Path to save the training results.
    device (Optional[torch.device]): Device to run the model on. If None, uses CUDA if available, otherwise CPU.

    Returns:
    Dict[str, List[float]]: A dictionary containing training loss and accuracy for each epoch.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model.to(device)
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    best_loss = float('inf')
    patience_counter = 0
    results = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        accuracy = 0

        for batch_idx, batch in tqdm(enumerate(dataloader), desc=f'Epoch {epoch + 1} (Training)', total=len(dataloader)):
            combined_features, target_tokens = batch
            combined_features = combined_features.to(device)
            target_tokens = target_tokens.to(device)
            
            output = model(combined_features)
            
            # Flatten the outputs and targets for loss calculation
            loss = criterion(output.view(-1, vocab_size), target_tokens.view(-1))
            
            # Calculate accuracy
            non_pad_elements = target_tokens != 1
            accuracy += (output.argmax(-1)[non_pad_elements] == target_tokens[non_pad_elements]).sum().item() / non_pad_elements.sum().item()

            optimizer.zero_grad()
            loss.backward()
            
            # plot_grad_flow(model.named_parameters())
            optimizer.step()
            
            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        average_accuracy = accuracy / len(dataloader)
        
        # Perform validation
        model.eval()
        val_loss = 0
        val_accuracy = 0

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(validation_dataloader), desc='Validation', total=len(validation_dataloader)):
                combined_features, target_tokens = batch
                combined_features = combined_features.to(device)
                target_tokens = target_tokens.to(device)
                    
                output = model(combined_features)
                loss = criterion(output.view(-1, vocab_size), target_tokens.view(-1))
                
                # Calculate accuracy
                non_pad_elements = target_tokens != 1
                val_accuracy += (output.argmax(-1)[non_pad_elements] == target_tokens[non_pad_elements]).sum().item() / non_pad_elements.sum().item()
                
                val_loss += loss.item()

        average_val_loss = val_loss / len(validation_dataloader)
        average_val_accuracy = val_accuracy / len(validation_dataloader)
        
        logger.info(f'Epoch {epoch + 1}, Training Loss: {average_loss:.4f}, Training Accuracy: {average_accuracy:.4f}')
        logger.info(f'Epoch {epoch + 1}, Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {average_val_accuracy:.4f}')
        
        results.append({
            'epoch': epoch + 1,
            'train_loss': average_loss,
            'train_accuracy': average_accuracy,
            'val_loss': average_val_loss,
            'val_accuracy': average_val_accuracy
        })

        # Early stopping and checkpoint saving
        if average_val_loss < best_loss:
            best_loss = average_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            logger.info('Model checkpoint saved.')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info('Early stopping.')
                break

    # Save results
    with open(results_save_path, 'w') as f:
        json.dump(results, f, indent=4)

    return results
    