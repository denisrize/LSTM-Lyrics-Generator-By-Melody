import pretty_midi
import numpy as np
from preprocess_data import extract_artist_song_name
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Function to adjust beats and downbeats to match the lyrics length
def beats_and_downbeats_features(beats,lyrics_length):
    # If the number of beats is greater than the number of tokens in the lyrics, apply moving average to compress them
    if len(beats) > lyrics_length:
        reduction_factor = len(beats) // lyrics_length
        # Compress beats using a moving average to fit the lyrics length
        beats = np.mean(beats[:lyrics_length * reduction_factor].reshape(-1, reduction_factor), axis=1)
    else:
        # If there are fewer beats, pad with zeros to match the lyrics length
        beats = np.pad(beats, (0, lyrics_length - len(beats)))
    return beats

# Extract features from each MIDI file and compress them to match the corresponding lyrics length
def extract_and_compress_features(midi_files,lyrics_length_dict=None):
    features_vectors = {}  # Dictionary to store feature vectors for each song
    program_id_list = list(range(128))  # List of 128 possible MIDI program IDs (standard instrument set)
    num_instruments = 128  # Maximum number of unique instruments

    # Process each MIDI file
    for midi_file in midi_files:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        # Determine the feature vector length by taking the maximum number of notes across all instruments
        feature_length = max(len(instrument.notes) for instrument in midi_data.instruments) if midi_data.instruments else 0
        # Initialize a feature vector for the song (3 features per instrument: program ID, pitch, velocity)
        feature_vector = np.zeros((feature_length, num_instruments * 3))  # Each instrument can have multiple notes

        for instrument in midi_data.instruments:
            index = program_id_list.index(instrument.program) * 3
            for i, note in enumerate(instrument.notes):
                # Store instrument program, pitch, and velocity for each note
                feature_vector[i, index:index+3] = [instrument.program, note.pitch, note.velocity]

        artist, song_name = extract_artist_song_name(midi_file)  # Extract artist and song name
        song_id = f"{artist}_{song_name}"  # Create a unique song ID
        features_vectors[song_id] = feature_vector  # Store the feature vector for this song
    
    # Compress the features for each song to match the length of the lyrics
    compressed_features = {}
    for song_id, feature_vector in features_vectors.items():
        lyrics_length = lyrics_length_dict[song_id]  # Get the lyrics length for the song
        # Compress the feature vector to fit the lyrics length
        compressed_feature = compress_features_adjusted(feature_vector, lyrics_length)
        compressed_features[song_id] = compressed_feature
    
    return compressed_features

# Function to compress features to match lyrics length
def compress_features_adjusted(feature_vectors, lyrics_length):
    # Find the actual length of non-zero data for compression
    actual_length = np.max([np.max(np.where(feature_vectors[:, i] != 0)[0]) + 1 if np.any(feature_vectors[:, i] != 0) else 0 for i in range(feature_vectors.shape[1])])
    
    # If the actual length is less than or equal to the lyrics length, just slice the feature vectors
    if actual_length <= lyrics_length:
        compressed_features = feature_vectors[:lyrics_length, :]
    else:
        # If the actual length is greater, compress the feature vectors
        compressed_features = np.zeros((lyrics_length, feature_vectors.shape[1]))
        step_size = actual_length / lyrics_length  # Calculate step size for compression
        
        for i in range(lyrics_length):
            # Average the data over the calculated step size
            start_index = int(i * step_size)
            end_index = int((i + 1) * step_size)
            compressed_features[i] = np.mean(feature_vectors[start_index:min(end_index, actual_length)], axis=0)
    
    return compressed_features

# Extract piano roll features and adjust to match lyrics length
def extract_piano_roll_feature(midi_data, lyrics_length):
    song_length = midi_data.get_end_time()  # Get the total song length in seconds
    fs = lyrics_length / song_length  # Calculate the sampling rate
    piano_roll = midi_data.get_piano_roll(fs=fs)  # Extract piano roll at the calculated sampling rate
    if piano_roll.shape[1] > lyrics_length:
        # If the piano roll has more frames than lyrics tokens, apply a moving average to compress it
        reduction_factor = piano_roll.shape[1] // lyrics_length
        piano_roll = np.mean(piano_roll[:, :lyrics_length * reduction_factor].reshape(piano_roll.shape[0], -1, reduction_factor), axis=2)
    else:
        # If fewer frames, pad with zeros
        piano_roll = np.pad(piano_roll, ((0, 0), (0, lyrics_length - piano_roll.shape[1])))
    return piano_roll

# Extract chroma features and adjust to match lyrics length
def extract_chroma_feature(midi_data, lyrics_length):
    song_length = midi_data.get_end_time()  # Get total song length
    fs = lyrics_length / song_length  # Calculate the sampling rate
    chroma = midi_data.get_chroma(fs=fs)  # Extract chroma features
    if chroma.shape[1] > lyrics_length:
        # If chroma has more frames than lyrics tokens, apply moving average to compress it
        reduction_factor = chroma.shape[1] // lyrics_length
        chroma = np.mean(chroma[:, :lyrics_length * reduction_factor].reshape(chroma.shape[0], -1, reduction_factor), axis=2)
    else:
        # If fewer frames, pad with zeros
        chroma = np.pad(chroma, ((0, 0), (0, lyrics_length - chroma.shape[1])))
    return chroma

# Create a tempo vector aligned with the length of the lyrics
def create_lyrics_tempo_vector(tempi, weights, lyrics_length):
    # Normalize the weights if they don't sum to 1 (e.g., to account for rounding errors)
    weights = weights / np.sum(weights)
    
    # Calculate how many tokens in the lyrics correspond to each tempo
    tokens_per_tempo = np.round(weights * lyrics_length).astype(int)
    
    # Adjust for rounding errors by adding/subtracting from the largest segment
    tokens_per_tempo[-1] += (lyrics_length - np.sum(tokens_per_tempo))
    
    # Create a tempo vector to match the lyrics
    lyrics_tempo_vector = np.zeros(lyrics_length, dtype=float)
    
    # Fill the tempo vector with the corresponding tempo values
    current_index = 0
    for tempo, count in zip(tempi, tokens_per_tempo):
        lyrics_tempo_vector[current_index:current_index + count] = tempo
        current_index += count
    
    return lyrics_tempo_vector

# Main function to extract and concatenate various features for each song
def extract_features_and_concatenate(midi_files, song_token_length):
    compressed_instruments_features_by_song = extract_and_compress_features(midi_files, song_token_length)

    features_by_song = {}
    
    # Process each MIDI file
    for midi_file_path in midi_files:
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file_path)  # Load the MIDI file
            artist, song_name = extract_artist_song_name(midi_file_path)  # Extract artist and song name
            song_id = f"{artist}_{song_name}"  # Create a unique song ID
            token_length = song_token_length[song_id]  # Get the number of tokens in the song's lyrics

            # Extract various musical features
            beats = midi_data.get_beats()
            beats_feature = beats_and_downbeats_features(beats, token_length)
            
            downbeats = midi_data.get_downbeats()
            downbeats_feature = beats_and_downbeats_features(downbeats, token_length)
            
            piano_roll_feature = extract_piano_roll_feature(midi_data, token_length)
            chroma_feature = extract_chroma_feature(midi_data, token_length)
            
            tempi = midi_data.estimate_tempi()  # Estimate the tempo of the song
            lyrics_tempo = create_lyrics_tempo_vector(tempi=tempi[0], weights=tempi[1], lyrics_length=token_length)
            
            # Concatenate all the extracted features
            full_feature_vector = np.hstack([
                beats_feature.reshape(-1, 1), 
                downbeats_feature.reshape(-1, 1), 
                piano_roll_feature.T, 
                chroma_feature.T, 
                lyrics_tempo.reshape(-1, 1),
            ])
            
            # Store the concatenated features for the song
            features_by_song[song_id] = full_feature_vector
        
        except Exception as e:
            print(f"Error processing {midi_file_path}: {e}")  # Handle any errors
    
    return features_by_song, compressed_instruments_features_by_song

def reduce_features_with_pca(original_features, target_dim=10):
    """
    Reduce the dimensionality of the features using PCA and plot the cumulative variance explained.

    Parameters:
    original_features (dict): Dictionary of song features with shape (seq_len, num_features).
    target_dim (int): The target number of dimensions after PCA.

    Returns:
    dict: Dictionary of reduced features.
    """
    scaler = MinMaxScaler()
    # Flatten the data
    all_features = np.concatenate([features for features in original_features.values()], axis=0)
    
    # Scale the data
    scaled_features = scaler.fit_transform(all_features)

    # Apply PCA
    pca = PCA(n_components=target_dim)
    reduced_features = pca.fit_transform(scaled_features)

    # Reshape and store back into a dictionary
    start = 0
    processed_features = {}
    for song, features in original_features.items():
        seq_len = features.shape[0]
        processed_features[song] = reduced_features[start:start + seq_len]
        start += seq_len

    return processed_features  
