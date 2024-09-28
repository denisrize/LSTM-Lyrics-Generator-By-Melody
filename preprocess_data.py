import re
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import numpy as np

def read_midi_files(root_path):
    midi_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".mid"):
                midi_files.append(os.path.join(root, file))
    return midi_files

def clean_text(lyrics):
    """
    Clean the lyrics text for NLP preprocessing.
    """
    # Replace '&   ' with '<end>' first to avoid conflicts
    lyrics = re.sub(r'&\s{3}', 'terminate', lyrics)
    # Replace remaining '&' with '<pause>'
    lyrics = re.sub(r'&', 'break', lyrics)
    # Replace '.' with '<br>'
    lyrics = re.sub(r'\.', 'period', lyrics)
    # Remove the words 'verse' and 'chorus'
    lyrics = re.sub(r'\b(verse|chorus)\b', '', lyrics, flags=re.IGNORECASE)
    # Remove special characters except <end>, <pause>, and <br>
    lyrics = re.sub(r'[^a-zA-Z\s]', '', lyrics)
    # Convert to lowercase
    lyrics = lyrics.lower()
    # Replace multiple spaces with a single space
    lyrics = re.sub(r'\s+', ' ', lyrics).strip()
    return lyrics

def read_data(train_path, test_path,valid_words):
    """
    Read and preprocess train, validation, and test data from CSV files.
    """
    def read_df(path):
        """
        Read and preprocess a single CSV file.
        """
        # Read the CSV file
        df = pd.read_csv(path, header=None)
        # Set column names
        columns_length = len(df.columns)
        df.columns = ['Artist', 'Song Name', 'Lyrics'] + [f'Column {i}' for i in range(3, columns_length)]
        # Fill missing values with empty strings
        df.fillna("", inplace=True)
        # Drop duplicate rows
        df.drop_duplicates(inplace=True)
        # Reset index
        df.reset_index(inplace=True, drop=True)
        
        # Combine lyrics columns into a single column
        malformed_lyric_cols = df.iloc[:, 3:].copy()
        df['Lyrics'] = df['Lyrics'] + ' ' + malformed_lyric_cols.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        df.drop(columns=malformed_lyric_cols.columns, inplace=True)
        
        # Apply text cleaning function
        df['Lyrics'] = df['Lyrics'].apply(clean_text)
        df['Lyrics'] = df['Lyrics'].apply(
            lambda lyrics: ' '.join(word for word in lyrics.split(' ') if word in valid_words))
        return df
    
    # Read and preprocess the train and test datasets
    train_set = read_df(train_path)
    test_set = read_df(test_path)
    
    # Split train set into train and validation sets
    validation_set = train_set.sample(frac=0.1, random_state=42)
    train_set = train_set.drop(validation_set.index)
    
    # Normalize artist and song name columns
    def normalize_columns(df):
        df['Artist'] = df['Artist'].str.lower().str.strip()
        df['Song Name'] = df['Song Name'].str.lower().str.strip()
        return df
    
    train_set = normalize_columns(train_set)
    validation_set = normalize_columns(validation_set)
    test_set = normalize_columns(test_set)
    
    # Reset indexes
    train_set.reset_index(drop=True, inplace=True)
    validation_set.reset_index(drop=True, inplace=True)
    test_set.reset_index(drop=True, inplace=True)

    train_set['id'] = train_set['Artist'] + '_' + train_set['Song Name']
    test_set['id'] = test_set['Artist'] + '_' + test_set['Song Name']
    validation_set['id'] = validation_set['Artist'] + '_' + validation_set['Song Name']
    return train_set, validation_set, test_set

def extract_artist_song_name(file_path):
    file_name = os.path.basename(file_path)
    try:    
        artist,song_name = file_name.split('_-_')[:2]
    except:
        print(f" Problematic {file_name}")
    artist = artist.lower().replace('_', ' ').strip()
    song_name = song_name.lower().replace('_', ' ').strip().replace('.mid','') 
    return artist,song_name

def remove_not_valid_midi_files(midi_files, train_set, validation_set, test_set):
    """
    Remove MIDI files that are not in the dataset.
    """
    to_remove_from_df = []
    to_remove_from_midi_files = []

    for midi_file in midi_files:
        artist,song_name = extract_artist_song_name(midi_file)
    
        # If artist and song name are not in the dataset remove the file from the list
        if artist not in train_set['Artist'].values and artist not in test_set['Artist'].values and artist not in validation_set['Artist'].values:
            # print(f"Removing {midi_file} from the list because artist {artist} is not in the dataset")
            to_remove_from_midi_files.append(midi_file)
            to_remove_from_df.append((artist, song_name))
        elif song_name not in train_set['Song Name'].values and song_name not in test_set['Song Name'].values and song_name not in validation_set['Song Name'].values:
            # print(f"Removing {midi_file} from the list because song name {song_name} is not in the dataset")
            to_remove_from_midi_files.append(midi_file)
            to_remove_from_df.append((artist, song_name))

    to_remove_from_midi_files = list(set(to_remove_from_midi_files))

    for file in to_remove_from_midi_files:
        midi_files.remove(file)
    print(f"Removed Total of {len(to_remove_from_midi_files)} songs.")

    return midi_files

def create_and_save_BPE_tokenizer(train_set, test_set, validation_set, vocab_size, save_path):
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Set up a trainer for BPE
    trainer = BpeTrainer(vocab_size=vocab_size,
                        min_frequency=3,
                        special_tokens=['terminate','break','period'],
                        show_progress=True,
                        continuing_subword_prefix="##" )  # Add subword prefix ) 

    # Prepare the corpus
    corpus = train_set['Lyrics'].tolist() + test_set['Lyrics'].tolist() + validation_set['Lyrics'].tolist()
    # Train the tokenizer
    tokenizer.train_from_iterator(corpus, trainer)
    # Save the tokenizer
    tokenizer.save(save_path)

    return tokenizer

def create_lyrics_by_song_id_dict(tokenizer, train_set, validation_set, test_set):
    # Define a dictionary to store the token lengths for each song
    song_token_length = {}

    # Iterate through each row in the train set DataFrame
    for index, row in train_set.iterrows():
        # Construct the unique song identifier from artist and song name
        song_id = f"{row['Artist']}_{row['Song Name']}"
        # Tokenize the lyrics and calculate the length of the resulting tokens
        song_token_length[song_id] = len(tokenizer.encode(row['Lyrics']))
    for index, row in validation_set.iterrows():
        # Construct the unique song identifier from artist and song name
        song_id = f"{row['Artist']}_{row['Song Name']}"
        # Tokenize the lyrics and calculate the length of the resulting tokens
        song_token_length[song_id] = len(tokenizer.encode(row['Lyrics']))
    for index, row in test_set.iterrows():
        # Construct the unique song identifier from artist and song name
        song_id = f"{row['Artist']}_{row['Song Name']}"
        # Tokenize the lyrics and calculate the length of the resulting tokens
        song_token_length[song_id] = len(tokenizer.encode(row['Lyrics']))

    return song_token_length

# Function to compute min-max scaling and return min/max values for later use
def compute_scaling(features, instrument_features):
    """
    Computes min and max scaling for given features and instrument features,
    then checks for constant features, and returns the min/max values.
    """
    # Stack features vertically into arrays
    all_features = np.vstack(list(features.values()))
    all_instrument_features = np.vstack(list(instrument_features.values()))

    # Calculate min and max for features and instrument features
    feature_min = np.min(all_features, axis=0)
    feature_max = np.max(all_features, axis=0) + 1  # Add 1 to avoid zero difference
    instrument_feature_min = np.min(all_instrument_features, axis=0)
    instrument_feature_max = np.max(all_instrument_features, axis=0) + 1

    # Calculate the difference between max and min (range for normalization)
    feature_diff = feature_max - feature_min
    instrument_feature_diff = instrument_feature_max - instrument_feature_min

    # Identify features with zero difference (constant features)
    zero_diff_features = np.where(feature_diff == 0)[0]
    zero_diff_instrument_features = np.where(instrument_feature_diff == 0)[0]

    # Output any constant features detected
    if len(zero_diff_features) > 0:
        print(f"Zero difference found in features at indices: {zero_diff_features}")
    else:
        print("No zero difference found in features.")

    if len(zero_diff_instrument_features) > 0:
        print(f"Zero difference found in instrument features at indices: {zero_diff_instrument_features}")
    else:
        print("No zero difference found in instrument features.")

    # Return the min and max values for later use
    return feature_min, feature_max, instrument_feature_min, instrument_feature_max