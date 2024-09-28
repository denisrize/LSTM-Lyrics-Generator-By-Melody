
# Specify the path to the MIDI files and the training and testing datasets
MIDI_PATH = './midi_files' 
TRAIN_PATH = './lyrics_train_set.csv'
TEST_PATH = './lyrics_test_set.csv'

# Specify the Vocabulary size for the lyrics generation (BPE tokenizer)
VOCAB_SIZE = 4000

# Specify the initial words for the lyrics generation
INITIAL_WORDS = ["close", "small", "eyes"]

# Specify the output file where the generated lyrics will be saved
GENERATED_LYRICS_PATH = "generated_lyrics.txt"

# Specify the path to the pre-trained Word2Vec model
WORD_TO_VEC_PATH = './GoogleNews-vectors-negative300.bin'