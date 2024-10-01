import gensim
import json
import logging
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import pickle
import pretty_midi
import random
import seaborn as sns
import ssl
import string
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CosineEmbeddingLoss

from collections import Counter
from transformers import PreTrainedTokenizerFast
from typing import Dict, List, Optional, Tuple
from nltk.corpus import wordnet
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from nltk.translate.bleu_score import sentence_bleu
from gensim.models import KeyedVectors

from preprocess_data import *
from feature_extraction import extract_features_and_concatenate, reduce_features_with_pca
from models_and_dataset import *
from config import *
from evaluation import evaluate_generated_lyrics, generate_lyrics_for_songs, save_generated_lyrics_to_file

print("All modules imported successfully!")

# Download stop words
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Downloading stopwords...")
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

nltk.download('wordnet')
valid_words = set(wordnet.words())

def load_word2vec_from_bin(bin_path):
    if os.path.exists(bin_path):
        print("Loading embeddings from bin file...")
        word2vec_model = KeyedVectors.load_word2vec_format(bin_path, binary=True)
        return word2vec_model
    else:
        raise FileNotFoundError("Bin file not found")

# Function to get the embedding for a word from the Word2Vec model
def get_word_embedding(word2vec_model, word):
    try:
        # Try to return the pre-trained embedding for the word
        return word2vec_model[word]
    except KeyError:
        # If the word is not found in the Word2Vec model, return a random vector as a fallback
        return np.random.normal(size=(300,))  # 300-dimensional random vector for unknown words

# Function to create an embedding matrix for the entire vocabulary
def create_embedding_matrix(vocab, word2vec_model):
    embedding_dim = 300  # Set the embedding dimension size
    embedding_matrix = np.zeros((len(vocab), embedding_dim))  # Initialize the matrix with zeros
    word2idx = {}  # Dictionary to map words to their index
    idx2word = {}  # Dictionary to map index to words

    # Loop through each word in the vocabulary and assign its corresponding embedding
    for i, word in enumerate(vocab):
        word2idx[word] = i  # Assign an index to each word
        idx2word[i] = word  # Store the word for each index
        embedding_matrix[i] = get_word_embedding(word2vec_model, word)  # Get word embedding for each word

    return embedding_matrix, word2idx, idx2word

if '__name__' == '__main__':

    # Load the all the necessary data
    word2vec_model = load_word2vec_from_bin(WORD_TO_VEC_PATH)
    midi_files = read_midi_files(MIDI_PATH)

    # Read and preprocess the data and split into train, validation, and test sets
    train_set, validation_set, test_set = read_data(TRAIN_PATH, TEST_PATH,valid_words)

    # Some of the midi files might be corrupted or not in the right format, make sure to remove them
    midi_files = remove_not_valid_midi_files(midi_files, train_set, validation_set, test_set)

    # Create a tokenizer and save it
    tokenizer = create_and_save_BPE_tokenizer(train_set, test_set, validation_set, VOCAB_SIZE, save_path='bpe_tokenizer.json')
    song_token_length = create_lyrics_by_song_id_dict(tokenizer, train_set, validation_set, test_set)

    # Extract features from the midi files
    # Start the feature extraction process
    features_by_song, instrument_features = extract_features_and_concatenate(midi_files, song_token_length)

    train_set = train_set[train_set['id'].isin(features_by_song.keys())]
    test_set = test_set[test_set['id'].isin(features_by_song.keys())]
    validation_set = validation_set[validation_set['id'].isin(features_by_song.keys())]

    # Reduce the dimensionality of the features using PCA
    reduced_song_features = reduce_features_with_pca(features_by_song.copy(), target_dim=40)
    reduced_instrument_features = reduce_features_with_pca(instrument_features.copy(), target_dim=60, features_name='Instrument')

    # Create the embedding matrix, word-to-index and index-to-word mappings
    vocab = tokenizer.get_vocab()
    embedding_matrix, word2idx, idx2word = create_embedding_matrix(vocab, word2vec_model)

    # Add special tokens to the vocabulary for unknown words and padding
    word2idx['<unk>'] = 1  # Unknown word token
    word2idx['<pad>'] = 1  # Padding token

    # Since the melody features may have different scales, it's important to normalize them
    # to ensure the model processes them more effectively, leading to better training and performance

    # Full Features Scaling
    feature_min, feature_max, instrument_feature_min, instrument_feature_max = compute_scaling(features_by_song, instrument_features)

    # Reduced Features Scaling
    feature_min_reduced, feature_max_reduced, instrument_feature_min_reduced, instrument_feature_max_reduced = compute_scaling(reduced_song_features, reduced_instrument_features)

    # Training the model
    # Full Features Scaling
    feature_min, feature_max, instrument_feature_min, instrument_feature_max = compute_scaling(features_by_song, instrument_features)

    # Reduced Features Scaling (PCA)
    feature_min_reduced, feature_max_reduced, instrument_feature_min_reduced, instrument_feature_max_reduced = compute_scaling(reduced_song_features, reduced_instrument_features)

    # Define model configurations
    model_configs = [
        {
            "model_type": "regular",
            "use_pca": False,
            "embedding_dim": 827,
            "hidden_dim": HIDDEN_DIM_REGULAR ,
            "num_layers": NUM_LAYERS_REGULAR ,
            "dropout": DROPOUT_REGULAR,
            "model_class": LyricsGenerator,
            "model_save_path": "model_regular_no_pca.pth",
            "results_save_path": "results_regular_no_pca.json"
        },
        {
            "model_type": "regular",
            "use_pca": True,
            "embedding_dim": 400,
            "hidden_dim": HIDDEN_DIM_REGULAR,
            "num_layers": NUM_LAYERS_REGULAR ,
            "dropout": DROPOUT_REGULAR,
            "model_class": LyricsGenerator,
            "model_save_path": "model_regular_with_pca.pth",
            "results_save_path": "results_regular_with_pca.json"
        },
        {
            "model_type": "advanced",
            "use_pca": False,
            "embedding_dim": 827,
            "hidden_dim": HIDDEN_DIM_ADVANCE ,
            "num_layers": NUM_LAYERS_ADVANCE ,
            "dropout": DROPOUT_ADVANCE ,
            "model_class": LyricsGeneratorAdvanced,
            "model_save_path": "model_advanced_no_pca.pth",
            "results_save_path": "results_advanced_no_pca.json"
        },
        {
            "model_type": "advanced",
            "use_pca": True,
            "embedding_dim": 400,
            "hidden_dim": HIDDEN_DIM_ADVANCE ,
            "num_layers": NUM_LAYERS_ADVANCE ,
            "dropout": DROPOUT_ADVANCE ,
            "model_class": LyricsGeneratorAdvanced,
            "model_save_path": "model_advanced_with_pca.pth",
            "results_save_path": "results_advanced_with_pca.json"
        }
    ]

    # Loop over each configuration and train the models
    for config in model_configs:
        # Select features and feature min/max scaling based on PCA usage
        if config["use_pca"]:
            feature_min = feature_min_reduced
            feature_max = feature_max_reduced
            instrument_feature_min = instrument_feature_min_reduced
            instrument_feature_max = instrument_feature_max_reduced
            features_by_song_model = reduced_song_features
            instrument_features_model = reduced_instrument_features
        else:
            features_by_song_model = features_by_song
            instrument_features_model = instrument_features

        # Create training and validation datasets with corresponding features
        train_dataset = LyricsMidiDataset(
            train_set, max_seq_length=32, tokenizer=tokenizer, word2idx=word2idx, embedding_matrix=embedding_matrix,
            features_by_song=features_by_song_model, compressed_instruments_features_by_song=instrument_features_model,
            feature_min=feature_min, feature_max=feature_max, instrument_feature_min=instrument_feature_min, instrument_feature_max=instrument_feature_max
        )

        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate_fn)

        validation_dataset = LyricsMidiDataset(
            validation_set, max_seq_length=32, tokenizer=tokenizer, word2idx=word2idx, embedding_matrix=embedding_matrix,
            features_by_song=features_by_song_model, compressed_instruments_features_by_song=instrument_features_model,
            feature_min=feature_min, feature_max=feature_max, instrument_feature_min=instrument_feature_min, instrument_feature_max=instrument_feature_max
        )

        validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)

        # Instantiate the model
        model = config["model_class"](config["embedding_dim"], config["hidden_dim"], VOCAB_SIZE, 
                                    num_layers=config["num_layers"], dropout=config["dropout"])

        # Define loss function and optimizer
        criterion = CombinedLoss(embedding_matrix, VOCAB_SIZE, alpha=0.5)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        print(f"Training {config['model_type']} model {'with PCA' if config['use_pca'] else 'without PCA'}...")

        # Train the model
        train_results = train_model(model, train_dataloader, validation_dataloader, criterion, optimizer, 
                                    num_epochs=100, patience=30, vocab_size=VOCAB_SIZE, 
                                    model_save_path=config["model_save_path"], results_save_path=config["results_save_path"])

        print(f"Training complete for {config['model_type']} model {'with PCA' if config['use_pca'] else 'without PCA'}. Model saved at {config['model_save_path']}.")

    # Evaluate the models
    # Paths to the saved models
    model_paths = {
        "base_no_pca": 'best_model_base_no_pca.pth',
        "base_with_pca": 'best_model_base_with_pca.pth',
        "advanced_no_pca": 'best_model_advanced_no_pca.pth',
        "advanced_with_pca": 'best_model_advanced_with_pca.pth'
    }

    # Define model parameters
    embedding_dim_with_pca = 400
    embedding_dim_no_pca = 827
    hidden_dim = 1024  
    base_num_layers = 1
    advanced_num_layers = 2

    # Initialize the models in a dictionary for easier looping
    models_config = {
        "base_no_pca": LyricsGenerator(embedding_dim_no_pca, hidden_dim, VOCAB_SIZE, num_layers=base_num_layers),
        "base_with_pca": LyricsGenerator(embedding_dim_with_pca, hidden_dim, VOCAB_SIZE, num_layers=base_num_layers),
        "advanced_no_pca": LyricsGeneratorAdvanced(embedding_dim_no_pca, hidden_dim, VOCAB_SIZE, num_layers=advanced_num_layers),
        "advanced_with_pca": LyricsGeneratorAdvanced(embedding_dim_with_pca, hidden_dim, VOCAB_SIZE, num_layers=advanced_num_layers)
    }

    # Load model weights from saved state_dicts and move models to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model_name, model in models_config.items():
        print(f"Loading model: {model_name}...")
        model.load_state_dict(torch.load(model_paths[model_name], map_location=device))
        model.to(device)

    # Evaluation for each model configuration
    results = []
    for model_name, model in models_config.items():
        print(f"Evaluating model: {model_name}...")
        
        if "with_pca" in model_name:
            # If using PCA, select reduced features
            song_features_model = reduced_song_features
            instrument_features_model = reduced_instrument_features
        else:
            # If not using PCA, select full features
            song_features_model = features_by_song
            instrument_features_model = instrument_features
        
        # Determine if the model is advanced or base
        model_type = 'advance' if 'advanced' in model_name else 'base'
        
        # Evaluate the model
        result = evaluate_generated_lyrics(
            model,
            test_set,
            song_features_model,
            instrument_features_model,
            embedding_matrix,
            idx2word,
            word2idx,
            device,
            model_type=model_type
        )
        
        # Add additional info to the result dictionary
        result['model'] = model_name
        result['pca'] = 'with_pca' in model_name
        
        # Append the result to the results list
        results.append(result)

    # Combine all results into a single DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file
    results_df.to_csv('evaluation_results.csv', index=False)

    print("Evaluation complete. Results saved to 'evaluation_results.csv'.")

    # Generate lyrics for each song in the test set using all models
    # Use the same initial words for all songs, as defined in the config
    test_initial_words = [tuple(INITIAL_WORDS) for _ in range(len(test_set))]

    # Call the function to generate lyrics for all songs
    all_generated_lyrics = generate_lyrics_for_songs(
        test_set,
        test_initial_words,
        models_config,  
        features_by_song,
        reduced_song_features,
        instrument_features,
        reduced_instrument_features,
        embedding_matrix,
        idx2word,
        word2idx
    )

    # Save the generated lyrics to a file
    save_generated_lyrics_to_file(all_generated_lyrics, GENERATED_LYRICS_PATH)

    print(f"Generated lyrics have been saved to: {GENERATED_LYRICS_PATH}")
