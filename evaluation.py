import torch
import torch.nn.functional as F
from spellchecker import SpellChecker
import torch.nn as nn
from tqdm import tqdm

def generate_lyrics(model, initial_word, midi_features, instrument_features, embedding_matrix, idx2word, word2idx, valid_words, max_length=10000, device='cpu', temperature=1.0):
    # Set the model in evaluation mode (no gradient computation)
    model.eval()
    
    # Set device to CUDA if available, else use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to the appropriate device
    model.to(device)

    # Initialize the location index for midi and instrument features
    idx_location = 0
    
    # Start the generated lyrics list with the initial word
    generated_lyrics = [initial_word]

    # Check if the initial word is in the vocabulary; if not, use <unk> (unknown token)
    if initial_word not in word2idx:
        initial_word = '<unk>'
    
    # Get the embedding of the initial word and prepare it for input (1x1xEmbedding_Dim)
    word_embedding = torch.tensor(embedding_matrix[word2idx[initial_word]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Generate lyrics without calculating gradients 
    with torch.no_grad():
        for _ in range(max_length):
            # Ensure the feature index does not exceed the length of MIDI or instrument features
            if idx_location >= len(midi_features) or idx_location >= len(instrument_features):
                break

            # Fetch the corresponding MIDI and instrument feature vectors and prepare them for input (1x1xFeature_Dim)
            midi_feature_vector = torch.tensor(midi_features[idx_location], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            instrument_feature_vector = torch.tensor(instrument_features[idx_location], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            # Concatenate the word embedding with MIDI and instrument features along the last dimension
            combined_input = torch.cat((word_embedding, midi_feature_vector, instrument_feature_vector), dim=2)
            
            # Pass the combined input through the model to get the output
            output = model(combined_input)
            
            # Sample the next word index from the output using the sampling function
            next_word_index = sample_word(output[0, -1], temperature)
            next_word = idx2word[next_word_index]

            # Stop generating if the model predicts the 'terminate' token
            if next_word == 'terminate':
                generated_lyrics.append(next_word)
                generated_lyrics.append('.')
                print(f"Reached <END>")
                break

            # Handle subword tokens (those starting with "##") and check if it forms a valid word
            if next_word.startswith('##'):
                # Combine with the previous word to create a full word
                potential_word = generated_lyrics[-1] + next_word[2:]  # Try to merge the subword
                potential_word_2 = generated_lyrics[-1] + '_' + next_word[2:]
                
                # If the merged word is valid, replace the last word in the lyrics with the combined word
                if potential_word in valid_words or potential_word_2 in valid_words:
                    generated_lyrics[-1] = potential_word
                else:
                    generated_lyrics.append('')
            else:
                # If it's a regular word, add it to the lyrics
                generated_lyrics.append(next_word)
            
            # Update the word embedding for the next iteration
            word_embedding = torch.tensor(embedding_matrix[word2idx.get(next_word, word2idx['<unk>'])], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            # Move to the next set of features
            idx_location += 1

    # Join the generated lyrics into a single string and return it
    return ' '.join(generated_lyrics)

def sample_word(output, temperature=1.0):
    """
    Samples a word from the output distribution using temperature scaling.
    
    Args:
    - output: The output logits from the model for the current step.
    - temperature: A hyperparameter that controls the randomness of sampling.
    
    Returns:
    - The index of the sampled word.
    """
    # Apply temperature scaling to the output logits
    output = output / temperature
    
    # Convert logits to probabilities using softmax
    probabilities = F.softmax(output, dim=-1)
    
    # Sample a word index from the probability distribution
    return torch.multinomial(probabilities, 1).item()

def generate_lyrics_advanced(model, initial_word, midi_features, instrument_features, embedding_matrix, idx2word, word2idx, valid_words, max_length=1000, max_context_length=50, device='cpu', temperature=1.0):
    """
    Generates lyrics based on an initial word, MIDI features, and instrument features using a pre-trained model.
    
    Args:
    - model: The trained model used for lyric generation.
    - initial_word: The starting word for generating the lyrics.
    - midi_features: A sequence of MIDI features associated with the song.
    - instrument_features: A sequence of instrument-specific features.
    - embedding_matrix: Pre-trained embedding matrix for the vocabulary.
    - idx2word: A dictionary mapping indices to words.
    - word2idx: A dictionary mapping words to indices.
    - max_length: Maximum number of tokens to generate (default is 1000).
    - max_context_length: Maximum number of previous tokens to consider as context (default is 50).
    - device: Device to run the model on ('cpu' or 'cuda').
    - temperature: Temperature for sampling (controls the randomness of the output).
    
    Returns:
    - generated_lyrics: A string of the generated lyrics.
    - attention_weights_per_word: Attention weights for each generated word.
    """
    
    # Set the model to evaluation mode (disables dropout and gradient calculations)
    model.eval()
    
    # Ensure the model is using the appropriate device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Initialize the list of generated lyrics with the initial word
    generated_lyrics = [initial_word]
    
    # List to store attention weights for each generated word
    attention_weights_per_word = []

    # Handle case where the initial word is not in the vocabulary
    if initial_word not in word2idx:
        initial_word = '<unk>'  # Use the unknown token if the word is not found in the vocabulary

    # Convert the initial word into its embedding and add it to the list of word embeddings
    word_embeddings = [torch.tensor(embedding_matrix[word2idx[initial_word]], dtype=torch.float32).to(device)]
    
    # Begin generating lyrics without computing gradients (to save memory and improve performance)
    with torch.no_grad():
        for idx_location in range(max_length):
            # If we've run out of MIDI or instrument features, stop generating
            if idx_location >= len(midi_features) or idx_location >= len(instrument_features):
                break

            # Calculate the start of the context window, limiting it to the last `max_context_length` words
            context_start = max(0, idx_location - max_context_length + 1)

            # Extract the corresponding MIDI and instrument features for the current context window
            midi_feature_vector = torch.tensor(midi_features[context_start:idx_location+1], dtype=torch.float32).to(device)
            instrument_feature_vector = torch.tensor(instrument_features[context_start:idx_location+1], dtype=torch.float32).to(device)

            # Limit the word embeddings to the same context window
            limited_word_embeddings = word_embeddings[context_start:]

            # Ensure the padded word embeddings match the length of the feature vectors
            seq_len = midi_feature_vector.size(0)
            padded_word_embeddings = torch.zeros(seq_len, word_embeddings[0].size(0), dtype=torch.float32).to(device)
            padded_word_embeddings[:len(limited_word_embeddings)] = torch.stack(limited_word_embeddings)
            
            # Expand the dimensions of word embeddings and feature vectors to match the model's input requirements
            padded_word_embeddings = padded_word_embeddings.unsqueeze(0)  # Shape: [1, seq_len, embedding_dim]
            midi_feature_vector = midi_feature_vector.unsqueeze(0)  # Shape: [1, seq_len, midi_feature_dim]
            instrument_feature_vector = instrument_feature_vector.unsqueeze(0)  # Shape: [1, seq_len, instrument_feature_dim]

            # Concatenate word embeddings, MIDI features, and instrument features along the last dimension
            combined_input = torch.cat((padded_word_embeddings, midi_feature_vector, instrument_feature_vector), dim=2)
            
            # Pass the concatenated input through the model to get the output logits
            output = model(combined_input)
            
            # Sample the next word index from the model's output using temperature-controlled sampling
            next_word_index = sample_word(output[0, -1], temperature)
            next_word = idx2word[next_word_index]

            # Record the attention weights for the current step (assuming the model uses attention)
            attention_weights_per_word.append(model.attention_weights_per_step[-1])
            
            # Stop generating if the model predicts the 'terminate' token
            if next_word == 'terminate':
                generated_lyrics.append(next_word)
                break

            # Handle subword tokens (those starting with "##")
            if next_word.startswith('##'):
                potential_word = generated_lyrics[-1] + next_word[2:]  # Combine subword with the previous word
                potential_word_2 = generated_lyrics[-1] + '_' + next_word[2:]
                
                # If the combined word is valid, update the last word in the generated lyrics
                if potential_word in valid_words or potential_word_2 in valid_words:
                    generated_lyrics[-1] = potential_word
                else:
                    generated_lyrics.append('')  # If not valid, append an empty string
            else:
                generated_lyrics.append(next_word)  # Add the next word to the generated lyrics
            
            # Add the embedding of the newly generated word to the list of embeddings
            word_embeddings.append(torch.tensor(embedding_matrix[word2idx.get(next_word, word2idx['<unk>'])], dtype=torch.float32).to(device))

    # Return the generated lyrics as a string and the attention weights for each word
    return ' '.join(generated_lyrics), attention_weights_per_word

def cosine_similarity(vec1, vec2):
    # Ensure vec1 and vec2 are PyTorch tensors; if not, convert them
    if not isinstance(vec1, torch.Tensor):
        vec1 = torch.tensor(vec1)
    if not isinstance(vec2, torch.Tensor):
        vec2 = torch.tensor(vec2)

    # Detach tensors from the computation graph to avoid gradients and clone them for safety
    vec1 = vec1.clone().detach()
    vec2 = vec2.clone().detach()
    
    # Calculate and return the cosine similarity between vec1 and vec2
    # The unsqueeze(0) adds a batch dimension (required by cosine_similarity)
    return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

def get_embedding(word_idx, embedding_matrix):
    # Retrieve the embedding vector for the word index from the embedding matrix
    return embedding_matrix[word_idx]

def evaluate_cosine_similarity_unigram(predictions, targets, embedding_matrix):
    similarities = []
    # Loop over each word index pair from predictions and targets
    for pred_word_idx, true_word_idx in zip(predictions, targets):
        # Get the embedding vectors for the predicted and actual words
        pred_vec = get_embedding(pred_word_idx, embedding_matrix)
        true_vec = get_embedding(true_word_idx, embedding_matrix)
        
        # Calculate the cosine similarity between the predicted and actual embeddings
        similarity = cosine_similarity(pred_vec, true_vec)
        similarities.append(similarity)
    
    # Return the average cosine similarity across all words
    return sum(similarities) / len(similarities)

def create_ngrams(sequence, n):
    # Create a list of n-grams from the sequence
    # Each n-gram is a list of n consecutive word indices
    return [sequence[i:i+n] for i in range(len(sequence) - n + 1)]

def average_ngram_embedding(ngram, embedding_matrix):
    # Get the embeddings for each word in the n-gram
    embeddings = [get_embedding(word_idx, embedding_matrix) for word_idx in ngram]
    
    # Convert the list of embeddings (which are numpy arrays) into PyTorch tensors
    embeddings = [torch.tensor(embedding).clone().detach() for embedding in embeddings]
    
    # Stack the embeddings into a single tensor and calculate the mean (average) vector for the n-gram
    return torch.mean(torch.stack(embeddings), dim=0)

def evaluate_cosine_similarity_ngram(predictions, targets, embedding_matrix, n):
    # Create n-grams for the predicted and target sequences
    pred_ngrams = create_ngrams(predictions, n)
    true_ngrams = create_ngrams(targets, n)
    
    similarities = []
    # Loop over each pair of predicted and true n-grams
    for pred_ngram, true_ngram in zip(pred_ngrams, true_ngrams):
        # Calculate the average embedding vectors for the predicted and true n-grams
        pred_ngram_vec = average_ngram_embedding(pred_ngram, embedding_matrix)
        true_ngram_vec = average_ngram_embedding(true_ngram, embedding_matrix)
        
        # Compute the cosine similarity between the two n-gram vectors
        similarity = cosine_similarity(pred_ngram_vec, true_ngram_vec)
        similarities.append(similarity)
    
    # Return the average cosine similarity for all n-grams, or 0 if no similarities were calculated
    return sum(similarities) / len(similarities) if similarities else 0

def semantic_coherence(predictions, word_to_idx, embedding_matrix):
    """
    Calculate the semantic coherence of the generated lyrics.
    """
    embeddings = [torch.tensor(embedding_matrix[word_to_idx[word]]) for word in predictions if word in word_to_idx]
    embeddings = torch.stack(embeddings)  # Convert to torch tensor for easier handling

    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)
    
    if not similarities:
        return 0.0
    
    return sum(similarities) / len(similarities)

def calculate_perplexity_on_generated(model, generated_lyrics, word2idx, embedding_matrix, midi_features, instrument_features, device):
    # Set the model to evaluation mode (disables dropout and gradient computation)
    model.eval()
    total_loss = 0  # Variable to accumulate the total loss
    total_words = len(generated_lyrics)  # Total number of words in the generated lyrics
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Define the loss function (cross-entropy)

    with torch.no_grad():  # No gradients needed for perplexity calculation
        # Initialize the first input as the embedding for the first word in the generated lyrics
        initial_word_idx = word2idx[generated_lyrics[0]]
        input_indices = [initial_word_idx]
        
        # Convert the rest of the generated lyrics to indices (using <unk> for unknown words)
        target_indices = [word2idx[word] if word in word2idx else word2idx['<unk>'] for word in generated_lyrics[1:]]

        # Loop through each word in the generated lyrics (excluding the last one)
        for i in range(len(generated_lyrics) - 1):
            # Ensure that the feature index doesn't exceed the length of midi or instrument features
            if i >= len(midi_features) or i >= len(instrument_features):
                break

            # Retrieve the word embedding for the current word
            word_embedding = torch.tensor(embedding_matrix[input_indices[-1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            # Retrieve the corresponding MIDI and instrument feature vectors
            midi_feature_vector = torch.tensor(midi_features[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            instrument_feature_vector = torch.tensor(instrument_features[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            # Concatenate the word embedding, MIDI, and instrument features to form the model input
            combined_input = torch.cat((word_embedding, midi_feature_vector, instrument_feature_vector), dim=2)
            # Pass the combined input through the model to get the predicted output
            output = model(combined_input)
            # Reshape the output to have the shape (batch_size, vocab_size)
            output = output.view(-1, len(embedding_matrix))

            # Prepare the target word for this step
            target_tensor = torch.tensor([target_indices[i]], dtype=torch.long).to(device)
            # Calculate the loss for the predicted word compared to the actual word
            loss = criterion(output, target_tensor)
            total_loss += loss.item()  # Accumulate the loss for perplexity calculation

            # Append the next target word to the input list for the next iteration
            input_indices.append(target_indices[i])

    # Calculate the average loss over all the words
    avg_loss = total_loss / total_words
    # Convert the average loss to perplexity (exponent of the average loss)
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()  

def evaluate_generated_lyrics(model, test_df, features_by_song, all_instrument_features, embedding_matrix, idx2word, word2idx, device, model_type='base', max_length=100):
    """
    Evaluates generated lyrics by calculating perplexity, cosine similarity (unigram, bigram, trigram, fivegram), 
    and semantic coherence, using both the generated lyrics and the true lyrics from the test set.
    
    Args:
    - model: The trained model to evaluate.
    - test_df: DataFrame containing the test data (lyrics, song IDs, etc.).
    - features_by_song: Dictionary containing MIDI features for each song.
    - all_instrument_features: Dictionary containing instrument features for each song.
    - embedding_matrix: Pre-trained embedding matrix for the vocabulary.
    - idx2word: Dictionary mapping indices to words.
    - word2idx: Dictionary mapping words to their indices.
    - device: The device to run the model on ('cpu' or 'cuda').
    - model_type: Specifies the type of model ('base' or 'advanced') used for generation.
    - max_length: Maximum number of tokens to generate per song.
    
    Returns:
    - A dictionary containing evaluation metrics: unigram, bigram, trigram, fivegram cosine similarity,
      semantic coherence, and perplexity.
    """

    # Set model to evaluation mode (disables gradient computations and dropout)
    model.eval()

    # Lists to store various evaluation metrics for each song in the test set
    all_perplexities = []
    all_unigram_cosine_similarities = []
    all_bigram_cosine_similarities = []
    all_trigram_cosine_similarities = []
    all_fivegram_cosine_similarities = []
    all_semantic_coherences = []

    # Disable gradient calculations for efficiency during evaluation
    with torch.no_grad():
        # Iterate over each song in the test dataset
        for _, row in tqdm(test_df.iterrows(), desc='Generating Lyrics', total=len(test_df)):
            
            generated_lyrics_list = []  # Store generated lyrics for the current song
            all_predictions = []  # Store all predicted words for evaluation
            all_targets = []  # Store all true words for evaluation

            # Use the first word of the true lyrics as the initial word for generation
            initial_word = row['Lyrics'].split()[0]  
            true_lyrics = row['Lyrics'].split()  # Split the true lyrics into tokens
            song_name = row['id']  # Get the song ID

            # Retrieve MIDI and instrument features for the current song
            midi_features = features_by_song[song_name]
            instrument_features = all_instrument_features[song_name]

            # Generate lyrics based on the model type (either 'base' or 'advanced')
            if model_type == 'base':
                generated_lyrics = generate_lyrics(
                    model, initial_word, midi_features, instrument_features, embedding_matrix, 
                    idx2word, word2idx, max_length=max_length, device=device
                )
            else:
                generated_lyrics, _ = generate_lyrics_advanced(
                    model, true_lyrics, initial_word, midi_features, instrument_features, embedding_matrix, 
                    idx2word, word2idx, max_length=max_length, device=device
                )

            # Add the generated lyrics (as a list of words) to the results list
            generated_lyrics_list.append(generated_lyrics.split())

            # Collect predictions (generated words) and true targets (actual lyrics)
            all_predictions.extend(generated_lyrics.split())  # Generated lyrics
            all_targets.extend(true_lyrics[:len(generated_lyrics.split())])  # True lyrics (match generated length)

            # Convert predicted words and true words into indices (for evaluation)
            prediction_indices = [word2idx[word] if word in word2idx else word2idx['<unk>'] for word in generated_lyrics.split()]
            target_indices = [word2idx[word] if word in word2idx else word2idx['<unk>'] for word in all_targets]

            # Calculate perplexity for the generated lyrics
            perplexity_score = calculate_perplexity_on_generated(
                model, generated_lyrics.split(), word2idx, embedding_matrix, midi_features, instrument_features, device
            )
            all_perplexities.append(perplexity_score)

            # Evaluate cosine similarity between generated and true lyrics (unigram, bigram, trigram, fivegram)
            unigram_cosine_similarity = evaluate_cosine_similarity_unigram(prediction_indices, target_indices, embedding_matrix)
            bigram_cosine_similarity = evaluate_cosine_similarity_ngram(prediction_indices, target_indices, embedding_matrix, 2)
            trigram_cosine_similarity = evaluate_cosine_similarity_ngram(prediction_indices, target_indices, embedding_matrix, 3)
            fivegram_cosine_similarity = evaluate_cosine_similarity_ngram(prediction_indices, target_indices, embedding_matrix, 5)

            # Append cosine similarity scores for this song
            all_unigram_cosine_similarities.append(unigram_cosine_similarity)
            all_bigram_cosine_similarities.append(bigram_cosine_similarity)
            all_trigram_cosine_similarities.append(trigram_cosine_similarity)
            all_fivegram_cosine_similarities.append(fivegram_cosine_similarity)

            # Calculate semantic coherence for the generated lyrics
            semantic_coherence_score = semantic_coherence(all_predictions, word2idx, embedding_matrix)
            all_semantic_coherences.append(semantic_coherence_score)

    # Calculate average perplexity across all songs
    avg_perplexity = sum(all_perplexities) / len(all_perplexities)

    # Calculate average cosine similarities across all songs for different n-grams
    unigram_cosine_similarity = sum(all_unigram_cosine_similarities) / len(all_unigram_cosine_similarities)
    bigram_cosine_similarity = sum(all_bigram_cosine_similarities) / len(all_bigram_cosine_similarities)
    trigram_cosine_similarity = sum(all_trigram_cosine_similarities) / len(all_trigram_cosine_similarities)
    fivegram_cosine_similarity = sum(all_fivegram_cosine_similarities) / len(all_fivegram_cosine_similarities)

    # Calculate average semantic coherence across all songs
    semantic_coherence_score = sum(all_semantic_coherences) / len(all_semantic_coherences)

    # Print the evaluation metrics for the test set
    print(f"Unigram Cosine Similarity: {unigram_cosine_similarity:.4f}")
    print(f"Bigram Cosine Similarity: {bigram_cosine_similarity:.4f}")
    print(f"Trigram Cosine Similarity: {trigram_cosine_similarity:.4f}")
    print(f"Fivegram Cosine Similarity: {fivegram_cosine_similarity:.4f}")
    print(f"Semantic Coherence: {semantic_coherence_score:.4f}")
    print(f"Perplexity: {avg_perplexity:.4f}")

    # Return evaluation results as a dictionary
    return {
        'unigram_cosine': unigram_cosine_similarity,
        'bigram_cosine': bigram_cosine_similarity,
        'trigram_cosine': trigram_cosine_similarity,
        'fivegram_cosine': fivegram_cosine_similarity,
        'semantic_coherence': semantic_coherence_score,
        'perplexity': avg_perplexity
    }

def filter_real_words(text):
    """
    Filters real words from the given text based on a dictionary.
    
    Args:
    - text (str): A string containing the text to be processed.
    
    Returns:
    - real_words (str): A string containing only the valid (real) words found in the text.
    - non_real_words (str): A string containing the invalid (non-real) words found in the text.
    
    This function splits the input text into words, then checks each word against the spell checker
    dictionary. Words recognized as real are returned in one string, and non-recognized words are 
    returned in another string.
    """
    spell = SpellChecker()  # Initialize the spell checker
    words = text.split()  # Split the input text into individual words
    real_words = [word for word in words if word in spell]  # Collect words that are recognized as real
    non_real_words = [word for word in words if word not in spell]  # Collect words that are not recognized
    return ' '.join(real_words), ' '.join(non_real_words)  # Return both real and non-real words as strings

def post_process_lyrics(generated_lyrics):
    """
    Post-processes generated lyrics by replacing special tokens with appropriate symbols.
    
    Args:
    - generated_lyrics (str): A string containing generated lyrics as tokens.
    
    Returns:
    - processed_lyrics (str): A string with the processed lyrics, where special tokens are replaced 
      with symbols like line breaks, periods, etc.
    
    This function handles specific tokens that represent formatting or punctuation, such as 
    'break' for line breaks, 'period' for '.', and 'terminate' for ending the lyrics with a period. 
    It cleans and joins the processed lyrics into a final string.
    """
    # Split the generated lyrics into individual tokens/words
    generated_lyrics = generated_lyrics.split(" ")
    
    # Replace 'break' tokens with newline characters
    generated_lyrics = [token if token != 'break' else '\n' for token in generated_lyrics]
    
    # Replace 'period' tokens with actual periods
    generated_lyrics = [token if token != 'period' else '.' for token in generated_lyrics]
    
    # Replace 'terminate' tokens with periods (indicating the end of the lyrics)
    generated_lyrics = [token if token != 'terminate' else '.' for token in generated_lyrics]
    
    # Join the processed tokens back into a single string
    generated_lyrics = ' '.join(generated_lyrics)
    
    # Strip any leading or trailing whitespace
    generated_lyrics = generated_lyrics.strip()
    
    return generated_lyrics  

def generate_lyrics_for_songs(test_set, initial_words_list, models,
                              features_by_song, reduced_song_features, instrument_features, reduced_instrument_features,
                              embedding_matrix, idx2word, word2idx):
    """
    Generate lyrics for each song in the test set using multiple models and return the results.
    
    Args:
    - test_set: The test set containing song metadata.
    - initial_words_list: A list of tuples, where each tuple contains the initial words for generating lyrics.
    - models: Dictionary of models.
    - features_by_song: Dictionary of full features.
    - reduced_song_features: Dictionary of PCA-reduced song features.
    - instrument_features: Dictionary of full instrument features.
    - reduced_instrument_features: Dictionary of PCA-reduced instrument features.
    - embedding_matrix: The embedding matrix used by the models.
    - idx2word: A dictionary mapping indices to words.
    - word2idx: A dictionary mapping words to indices.

    Returns:
    - all_generated_lyrics: A dictionary containing the generated lyrics for all songs.
    """
    # Dictionary to store generated lyrics for all songs
    all_generated_lyrics = {}

    # Loop through each song in the test set
    for song_index, initial_words in enumerate(initial_words_list):
        # Get the song ID
        song_id = test_set.iloc[song_index]['id']
        print(f"Generating lyrics for song {song_id}...")

        # Generate lyrics for the song
        generated_lyrics = generate_lyrics_for_song(
            song_id, initial_words, models,  # Pass the dictionary of models
            features_by_song, reduced_song_features,
            instrument_features, reduced_instrument_features,
            embedding_matrix, idx2word, word2idx
        )
        
        # Save generated lyrics in a dictionary
        all_generated_lyrics[f'{song_index}'] = generated_lyrics

    return all_generated_lyrics

def generate_lyrics_for_song(song_id, initial_words, models,
                             features_by_song, reduced_song_features, instrument_features, reduced_instrument_features,
                             embedding_matrix, idx2word, word2idx):
    """
    Generate lyrics for a specific song using multiple models from the dictionary.
    
    Args:
    - song_id: The ID of the song.
    - initial_words: A tuple of initial words to start the lyric generation.
    - models: Dictionary of models.
    - features_by_song: Dictionary of full features.
    - reduced_song_features: Dictionary of PCA-reduced song features.
    - instrument_features: Dictionary of full instrument features.
    - reduced_instrument_features: Dictionary of PCA-reduced instrument features.
    - embedding_matrix: The embedding matrix used by the models.
    - idx2word: A dictionary mapping indices to words.
    - word2idx: A dictionary mapping words to indices.
    
    Returns:
    - generated_lyrics: A dictionary containing generated lyrics for each model and initial word.
    """
    generated_lyrics = {}

    # Loop through each initial word and each model configuration
    for initial_word in initial_words:
        for model_name in models:
            model = models[model_name]  # Access the model from the dictionary

            # Select the appropriate features for the model
            if 'with_pca' in model_name:
                song_features_model = reduced_song_features[song_id]
                instrument_features_model = reduced_instrument_features[song_id]
            else:
                song_features_model = features_by_song[song_id]
                instrument_features_model = instrument_features[song_id]

            # Generate lyrics based on model type (advanced or base)
            if 'advanced' in model_name:
                lyrics, _ = generate_lyrics_advanced(
                    model,
                    initial_word,
                    song_features_model,
                    instrument_features_model,
                    embedding_matrix,
                    idx2word,
                    word2idx,
                    max_length=1000,
                    max_context_length=64
                )
            else:
                lyrics = generate_lyrics(
                    model,
                    initial_word,
                    song_features_model,
                    instrument_features_model,
                    embedding_matrix,
                    idx2word,
                    word2idx,
                    max_length=1000
                )

            # Post-process and filter non-real words from the generated lyrics
            lyrics = post_process_lyrics(lyrics)
            real_words_lyrics, _ = filter_real_words(lyrics)

            # Store the generated lyrics
            if model_name not in generated_lyrics:
                generated_lyrics[model_name] = {}
            generated_lyrics[model_name][initial_word] = real_words_lyrics

    return generated_lyrics

def save_generated_lyrics_to_file(all_generated_lyrics, output_file):
    """
    Save generated lyrics to a text file.
    
    Args:
    - all_generated_lyrics: A dictionary containing generated lyrics for all songs and models.
    - output_file: The path to the text file where lyrics will be saved.
    """
    with open(output_file, 'w') as f:
        for song_index, models in all_generated_lyrics.items():
            f.write(f"Song {song_index}:\n")
            for model_name, lyrics_dict in models.items():
                for initial_word, lyrics in lyrics_dict.items():
                    f.write(f"Initial word '{initial_word}', Model '{model_name}':\n{lyrics}\n\n")
            f.write("\n")


