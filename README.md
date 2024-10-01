# RNN-Lyrics-Generator-By-Melody

## Introduction
In this project, the primary goal is to explore the automatic generation of song lyrics based on provided melodies using deep learning techniques. The core idea is to combine the temporal dependencies in lyrics with the rich musical information embedded in melodies, such as rhythm, pitch, and instrumentation. The task is approached using Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) units, which are well-suited for sequential data. To further enhance the quality of the generated lyrics, we utilize an attention mechanism in advanced models to dynamically focus on relevant parts of the melody and lyrics sequences during the generation process.

## Pipeline Description

### Tokenization - Byte Pair Encoding (BPE)
For tokenizing the lyrics, we employed the Byte Pair Encoding (BPE) algorithm. BPE is a subword segmentation algorithm that helps build a more efficient and flexible vocabulary by breaking down words into subwords, enabling the model to handle out-of-vocabulary words and rare word instances better.

#### Why we chose BPE?
- **Efficient Vocabulary Size Reduction:** BPE dramatically reduces the vocabulary size by representing words as subwords, making the model more efficient.
- **Handling Out-of-Vocabulary Words:** It effectively manages rare and unseen words by breaking them into known subword segments, preventing the model from struggling with unknown tokens.
- **Improved Contextual Understanding:** BPE captures morphemes and root words, improving the model’s ability to generate contextually appropriate and coherent lyrics.

### Melody Feature Extraction
We extracted the following melody features from MIDI files:
- **Beats and Downbeats:** Rhythmic information that aligns with the structure of the lyrics.
- **Piano Roll:** Representation of which notes are played at each time step.
- **Chroma Features:** Capture harmonic content by representing the twelve pitch classes.
- **Tempo:** Indicates the speed of the music and helps the model adjust to different rhythmic patterns.
- **Instrument Notes:** Information about instrument pitch and velocity, contributing to a more detailed melodic context.

These features provide the model with crucial information about the musical structure, which helps in generating lyrics that are more rhythmically and harmonically aligned with the melody.

### Dimensionality Reduction using PCA
Given the high dimensionality of the melody features (527 features per token), Principal Component Analysis (PCA) is employed to reduce the feature space. By selecting the first 100 principal components, we preserve around 90% of the variance in the data while significantly reducing the model’s input size. This helps the model focus on the essential aspects of the melody while improving training efficiency and reducing overfitting.

### Model Architectures

#### Basic LSTM Model
This architecture consists of two stacked LSTM layers to handle the temporal dependencies in the lyrics and melody. Lyrics and melody features are concatenated at the input level to form vectors of size `(batch_size, sequence_length, 827)`, where 300 corresponds to the word embeddings, and 527 corresponds to the melody features.

#### Advanced LSTM with Attention Mechanism
In this variant, an attention layer is added after the LSTM layers to allow the model to dynamically focus on different parts of the input sequence during the lyrics generation process. This architecture helps the model better capture the interplay between the melody and the lyrics.

### Training Phase
The model is trained using the cross-entropy loss function to predict the next word in the lyrics sequence, given the current word and the corresponding melody features. The training dataset consists of 600 songs, with 10% used for validation.

Key training details include:
- **Batch sizes:** 32 and 64.
- **Hidden dimensions:** 64 and 128.
- **Max sequence lengths:** 32 and 64 tokens.
- **Learning rates:** 0.001 and 0.0001.

Regularization techniques such as dropout (to prevent overfitting) and early stopping (to halt training if validation accuracy does not improve) are applied. Additionally, Layer Normalization and the attention mechanism are used to stabilize training and improve performance.

## Evaluation Process

When evaluating a task like automatic song lyrics generation, traditional metrics like accuracy are not sufficient. Unlike classification tasks where accuracy measures the exact match between predictions and ground truth, lyrics generation involves creative and context-dependent outputs. For example, if the original lyric says “dance,” but the model generates “groove,” both are valid within the context of the song. Standard accuracy would incorrectly penalize such variations.

To overcome this limitation, we use a broader set of evaluation metrics that capture different aspects of the generated lyrics, such as their fluency, coherence, and semantic relevance.

### Perplexity
- **Definition:** Perplexity measures how well the probability model predicts a sample, specifically by calculating the exponentiated average negative log-likelihood of the predicted sequence.
- **Why it’s useful:** Perplexity provides insight into the model’s confidence in its predictions. Lower perplexity indicates that the model is better at predicting the next word in the sequence, suggesting that it generates more coherent and contextually appropriate lyrics.

### Cosine Similarity
- **Unigram Cosine Similarity:** Compares each predicted word with the corresponding word in the original lyrics using the cosine of the angle between the vectors representing the words. This metric helps quantify how similar the generated lyrics are to the original at a word-by-word level.
- **N-gram Cosine Similarity:** Evaluates longer word sequences (e.g., bigrams, trigrams) to check whether the model maintains a reasonable sequence of words, even if they are slightly reordered or differ from the original. N-gram similarity captures the flow of ideas across multiple words.

- **Why it’s useful:** Cosine similarity evaluates both the semantic content and structural similarity between generated and original lyrics. By using n-gram similarity in addition to unigram, the evaluation captures cases where words are reordered but still convey a similar meaning, offering a more lenient and realistic evaluation compared to strict word-matching accuracy.

### Semantic Coherence
- **Definition:** Semantic coherence measures the extent to which the generated lyrics maintain a meaningful and logical flow of information. This is crucial for ensuring that the output not only fits the melody but also forms cohesive and understandable sentences or phrases.
- **Why it’s useful:** In the context of generating song lyrics, it’s vital that the lyrics are not only grammatically correct but also make sense as a whole. Semantic coherence ensures that the generated lyrics have logical consistency, essential for creating lyrics that sound natural and fit within the context of the song.

## Result Analysis

### PCA Effect:
- Performing PCA on melody features improves the perplexity for both models, indicating better prediction capabilities.
- The effect on semantic coherence is mixed: it slightly decreases for the Base model but increases for the Advanced model.
- Cosine similarities generally decrease for the Base model but show improvements or minor changes for the Advanced model when using PCA.

### Advanced vs. Base Model:
- The Advanced model significantly outperforms the Base model in terms of perplexity and semantic coherence, indicating better overall performance.
- Cosine similarities are quite similar between the two models, indicating no significant difference in their semantic alignment and contextual relevance of the generated lyrics.

These insights suggest that the Advanced model with PCA is the most effective configuration, offering the best balance between predictive confidence and semantic coherence.
