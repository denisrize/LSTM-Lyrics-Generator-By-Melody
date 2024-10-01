<table>
  <tr>
    <td>
      <h1>LSTM-Lyrics-Generator-By-Melody</h1>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/e6a20f77-e9c4-40b4-b1d1-189768c6c93d" alt="LSTM generating lyrics" width="200">
    </td>
  </tr>
</table>

## Table of Contents
- [Introduction](#introduction)
- [Preprocess and Feature Extraction](#preprocess-and-feature-extraction)
  - [Tokenization - Byte Pair Encoding (BPE)](#tokenization---byte-pair-encoding-bpe)
  - [Melody Feature Extraction](#melody-feature-extraction)
  - [Dimensionality Reduction using PCA](#dimensionality-reduction-using-pca)
- [Model Architectures](#model-architectures)
  - [Basic LSTM Model](#basic-lstm-model)
  - [Advanced LSTM with Attention Mechanism](#advanced-lstm-with-attention-mechanism)
- [Training Phase](#training-phase)
  - [Training and Validation Process](#training-and-validation-process)
- [Evaluation Process](#evaluation-process)
  - [Perplexity](#perplexity)
  - [Cosine Similarity](#cosine-similarity)
  - [Semantic Coherence](#semantic-coherence)
- [Result Analysis](#result-analysis)
  - [PCA Effect](#pca-effect)
  - [Advanced vs. Base Model](#advanced-vs-base-model)
- [Generated Lyrics Analysis](#generated-lyrics-analysis)
- [Analysis and Conclusion: Impact of Melody and Seed Words on Generated Lyrics](#analysis-and-conclusion-impact-of-melody-and-seed-words-on-generated-lyrics)
- [Conclusion](#conclusion)

## Introduction
In this project, the primary goal is to explore the automatic generation of song lyrics based on provided melodies using deep learning techniques. The core idea is to combine the temporal dependencies in lyrics with the rich musical information embedded in melodies, such as rhythm, pitch, and instrumentation. The task is approached using Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) units, which are well-suited for sequential data. To further enhance the quality of the generated lyrics, we utilize an attention mechanism in advanced models to dynamically focus on relevant parts of the melody and lyrics sequences during the generation process.

## Preprocess and Feature Extraction

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
In the following plots, you can see the reduced song features on the left, and reduced instrument features on the right plot.

![Picture1](https://github.com/user-attachments/assets/8bedc3bf-866a-49cd-8acb-4375018e34f0)


### Model Architectures

#### Basic LSTM Model
This architecture consists of two stacked LSTM layers to handle the temporal dependencies in the lyrics and melody. Lyrics and melody features are concatenated at the input level to form vectors of size `(batch_size, sequence_length, 827)`, where 300 corresponds to the word embeddings, and 527 corresponds to the melody features.

#### Advanced LSTM with Attention Mechanism
In this variant, an attention layer is added after the LSTM layers to allow the model to dynamically focus on different parts of the input sequence during the lyrics generation process. This architecture helps the model better capture the interplay between the melody and the lyrics.

### Training Phase
To get the best out of our lyric generation model, we needed a loss function that could do more than just predict the next word—it had to understand the meaning behind those words too. That’s where the Combined Loss comes in, blending two powerful approaches:

- Cross-Entropy Loss ensures the model predicts the correct word from the vocabulary. This is crucial for accuracy, helping the model hit the right word at the right time.

- Cosine Embedding Loss takes things a step further by ensuring the predicted word is not only correct but also similar in meaning to the actual word. This helps the model capture the context and flow of the lyrics, making the output feel more natural and meaningful.

And then there’s Alpha, The balancing act. It lets me control the trade-off between these two objectives. If more focus needed on the exact word, Decrease Alpha or else for better semantic similarity increase it. It’s a flexible way to ensure the model generates lyrics that are both accurate and contextually relevant.
The model is trained using the cross-entropy loss function to predict the next word in the lyrics sequence, given the current word and the corresponding melody features. The training dataset consists of 600 songs, with 10% used for validation.

#### Training and Validation Process
Training a model to generate lyrics can be compared to teaching it to compose a song, one word at a time. The model is trained to predict the next word based on both the current word and the accompanying melody features. During training, we employ a technique called teacher forcing, where the model is given the actual next word as a guide. This approach accelerates the learning process and helps the model stay aligned with the intended sequence, reducing the risk of generating irrelevant or incoherent lyrics.

Key hyperparameters include:
- **Batch sizes:** 32 and 64.
- **Hidden dimensions:** 64 and 128.
- **Max sequence lengths:** 32 and 64 tokens.
- **Learning rates:** 0.001 and 0.0001.

Regularization techniques such as dropout (to prevent overfitting) and early stopping (to halt training if validation accuracy does not improve) are applied. Additionally, Layer Normalization and the attention mechanism are used to stabilize training and improve performance.

Base Model – Melody features approach # 1
Num Layers: 1, Hidden Dim: 64, Dropout = 0.1, Loss: CrossEntropyLoss, LR: 0.001, Weight Decay: 0.0001, Optimizer: Adam, Max Sequence Length: 32, Batchsize = 64

![basemodelapp1](https://github.com/user-attachments/assets/5b468caf-0ae0-4a54-902e-ae1e261d7fe7)


Base Model – Melody features approach # 2
Num Layers: 1, Hidden Dim: 64, Dropout = 0.1, Loss: CrossEntropyLoss, LR: 0.001, Weight Decay: 0.0001, Optimizer: Adam, Max Sequence Length: 32, Batchsize = 64

![basemodelapp2](https://github.com/user-attachments/assets/b8f3793a-2d98-47af-97b1-386e965d81f0)

Advanced Model – Melody features approach # 1
Num Layers: 2, Hidden Dim: 64, Dropout = 0.2, Loss: CrossEntropyLoss, LR: 0.001, Weight Decay: 0.0001, Optimizer: Adam, Max Sequence Length: 32, Batchsize = 64

![advancemodelapp1](https://github.com/user-attachments/assets/e577bdee-0a0e-434c-b1cf-99efa637c8c6)

Advanced Model – Melody features approach # 2
Num Layers: 2, Hidden Dim: 64, Dropout = 0.2, Loss: CrossEntropyLoss, LR: 0.001, Weight Decay: 0.0001, Optimizer: Adam, Max Sequence Length: 32, Batchsize = 64

![advancemodelapp2](https://github.com/user-attachments/assets/eba5bd5e-ab93-4c34-aaab-c5328118d324)


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
For each melody in the test set, we generated lyrics using both architectural variants using the PCA-reduced melody features. The process was repeated three times with different initial words for each melody.
The result for each of the model architecture and melody representation (best are in bold):

![Screenshot 2024-10-01 113942](https://github.com/user-attachments/assets/454a13cd-a513-467c-bc7e-01eeebd3a04a)

Results Visualization -

![cosine_ngrams](https://github.com/user-attachments/assets/1094332b-4811-42e0-8c96-83de6a9f6caa)


![perplexity_semantic](https://github.com/user-attachments/assets/a1b8e76b-b8de-4781-a9b8-9b108b425aa3)


### PCA Effect:
- Performing PCA on melody features improves the perplexity for both models, indicating better prediction capabilities.
- The effect on semantic coherence is mixed: it slightly decreases for the Base model but increases for the Advanced model.
- Cosine similarities generally decrease for the Base model but show improvements or minor changes for the Advanced model when using PCA.

### Advanced vs. Base Model:
- The Advanced model significantly outperforms the Base model in terms of perplexity and semantic coherence, indicating better overall performance.
- Cosine similarities are quite similar between the two models, indicating no significant difference in their semantic alignment and contextual relevance of the generated lyrics.

These insights suggest that the Advanced model with PCA is the most effective configuration, offering the best balance between predictive confidence and semantic coherence.

### Generated Lyrics Analysis

We won't include all the generated lyrics in this README as the output is quite extensive, but let's take a look at a few examples:

**1.Song 'bangles eternal flame':**

**Initial word 'close', Model 'base_no_pca':** close question walk alone heaven wanting earl below pack shore vulnerability message aim better try rebel s cheek order desert spinning forget star bad hearts top maybe gentle music windows moving anywhere wonderful repeat amazed glimpse north painful prove north bottle plan let.

**Initial word 'small', Model 'base_no_pca':** small fade lonely sabotage cell ban ahead trunk mesmerized passing fame weep gentle saw stray gang presence truly pound felt monkey raining pound farm beth command alien nerve wine band cause wheel cred north short hands fade single leg two game don catch all redeemed hi criminal perfect meter important bright mean someday returning corner dumb north in fence.

**Initial word 'eyes', Model 'base_no_pca':** eyes nice pave ins memory try point ell karma tumbling hes bang turned opened dim gentle silk saw breeze hearts fresh yell taking whey penny longer grease better drunk raining con hell worry chill clap fields power must ting instinctively brown flavor nice slim pave wit game not should telephone.

**Initial word 'close', Model 'base_with_pca':** close harmony girlfriend hug no noise pave prove sin man certain check right proof rolling embrace charm precious fame gray pa faster prof sick rightly history knee game three draw ship sweat kiss heaven high slam genie sugar boogie sky drown boogie either haunted draw blowing dime boogie saw.

**Initial word 'small', Model 'base_with_pca':** small para south peter top sock gentle chain win nice top sick desert garden stick bare ans yule slam perfect hardly busy em race harmony fighting wire therefore nuts backward ask myrrh unfurl unlimited ad sleigh stand prom cant brown join punk u paved knock hug liquor.

**Initial word 'eyes', Model 'base_with_pca':** eyes date charm nation slut desert embrace boogie care gates shit nation try blast coat chin join glimpse fade wonderful brand stone wont grieving rent prove weep wonderful hands easily try shoot asleep joint sh now should follow ins business neither pan bond staring hurry ain bond silence bear.

**Initial word 'close', Model 'advanced_no_pca':** close fort able defend f north bright my bitter more in fade gentle test pinch north longer ink in stage north pissed stumble north precious differ pave type prove holiday how silence hail turn embrace longer names depressed sock star fig.

**Initial word 'small', Model 'advanced_no_pca':** small sat corvette cute hurting rap folks dick north ss midnight fade vibes book dirty bitter social criminal nice acting story gentle in better fashion naught fruitless ask try sock silence pissed settle in in blues nice sock sock fade.

**Initial word 'eyes', Model 'advanced_no_pca':** eyes ice hearts shepherd order f physic thirty tie seem fade pinch tied hush thous least depressed weep stupid cruel pack nice ass in heaven ask lady tie fade ken taste pave pave strength number fade sherry ground longer control.

**Initial word 'close', Model 'advanced_with_pca':** close no ting carry window color nice aware no gaze chain ale instead nice nice bad sentiment craze pa covered rap dawn hair control nice reflect sandman baa billy presence flash defeat coff painful ill nice roof.

**Initial word 'small', Model 'advanced_with_pca':** small short single cover par choice nice enough cred nice sabotage gentle boardwalk three attention frosty hear yester cast fourth bad magazine foggy date nothing worry north wonderful fashion crowded hero york levee gar join thirty off blast station starting couple shades drown rap.

**Initial word 'eyes', Model 'advanced_with_pca':** eyes shadow draw nice channel gentle pull busy acting stick hardly sock role nice brown fate dad new list gentle joy steps new try par game stem eye maybe no skip order make candlelight youth haste private liberty nation miser signed offer don fashion magazine hair bad.

**2.Song 'Aqua barbie girl':**

**Initial word 'close', Model 'base_no_pca':** close question walk alone heaven wanting earl below pack shore vulnerability message aim better try rebel s cheek order desert spinning forget star bad hearts top maybe gentle music windows moving anywhere wonderful repeat amazed glimpse north painful prove north bottle plan let be.

**Initial word 'small', Model 'base_no_pca':** small fade lonely sabotage cell ban ahead trunk mesmerized passing fame weep gentle saw stray gang presence truly pound felt monkey raining pound farm beth command alien nerve wine band cause wheel cred north short hands fade single leg two game don catch all redeemed hi criminal perfect meter important bright mean someday returning corner dumb north in fence.

**Initial word 'eyes', Model 'base_no_pca':** eyes nice pave ins memory try point ell karma tumbling hes bang turned opened dim gentle silk saw breeze hearts fresh yell taking whey penny longer grease better drunk raining con hell worry chill clap fields power must ting instinctively brown flavor nice slim pave wit game not should telephone.

**Initial word 'close', Model 'base_with_pca':** close harmony girlfriend hug no noise pave prove sin man certain check right proof rolling embrace charm precious fame gray pa faster prof sick rightly history knee game three draw ship sweat kiss heaven high slam genie sugar boogie sky drown boogie either haunted draw blowing dime boogie saw.

**Initial word 'small', Model 'base_with_pca':** small para south peter top sock gentle chain win nice top sick desert garden stick bare ans yule slam perfect hardly busy em race harmony fighting wire therefore nuts backward ask myrrh unfurl unlimited ad sleigh stand prom cant brown join punk u paved knock hug liquor.

**Initial word 'eyes', Model 'base_with_pca':** eyes date charm nation slut desert embrace boogie care gates shit nation try blast coat chin join glimpse fade wonderful brand stone wont grieving rent prove weep wonderful hands easily try shoot asleep joint sh now should follow ins business neither pan bond staring hurry ain bond silence bear.

**Initial word 'close', Model 'advanced_no_pca':** close fort able defend f north bright my bitter more in fade gentle test pinch north longer ink in stage north pissed stumble north precious differ pave type prove holiday how silence hail turn embrace longer names depressed sock star fig.

**Initial word 'small', Model 'advanced_no_pca':** small sat corvette cute hurting rap folks dick north ss midnight fade vibes book dirty bitter social criminal nice acting story gentle in better fashion naught fruitless ask try sock silence pissed settle in in blues nice sock sock fade.

**Initial word 'eyes', Model 'advanced_no_pca':** eyes ice hearts shepherd order f physic thirty tie seem fade pinch tied hush thous least depressed weep stupid cruel pack nice ass in heaven ask lady tie fade ken taste pave pave strength number fade sherry ground longer control.

**Initial word 'close', Model 'advanced_with_pca':** close no ting carry window color nice aware no gaze chain ale instead nice nice bad sentiment craze pa covered rap dawn hair control nice reflect sandman baa billy presence flash defeat coff painful ill nice roof.

**Initial word 'small', Model 'advanced_with_pca':** small short single cover par choice nice enough cred nice sabotage gentle boardwalk three attention frosty hear yester cast fourth bad magazine foggy date nothing worry north wonderful fashion crowded hero york levee gar join thirty off blast station starting couple shades drown rap.

**Initial word 'eyes', Model 'advanced_with_pca':** eyes shadow draw nice channel gentle pull busy acting stick hardly sock role nice brown fate dad new list gentle joy steps new try par game stem eye maybe no skip order make candlelight youth haste private liberty nation miser signed offer don fashion magazine hair bad.

## Analysis and Conclusion: Impact of Melody and Seed Words on Generated Lyrics
The generated lyrics are mostly unintelligible and frequently contain words that are very common in the dataset. For instance, the word "quiet" appears over 100 times in the generated lyrics, which is a common word in rap songs. We observed that once a word appeared for the first time, it tended to reappear frequently.
Additionally, our models generated a significant amount of cursing, and many of the lyrics resemble rap songs. This likely occurs because our dataset includes a relatively large portion of rap and hip-hop songs, which typically contain more words. As a result, the model learns more words related to rap. We even noticed that the word "marshall" was generated, which is Eminem's nickname in his songs.

### Initial word effect on the generated lyrics:
The choice of the initial word significantly influences the words produced in the generated lyrics. We calculated the average number of unique words shared between lyrics that start with different initial words ('close', 'small', and 'eyes') for each model. The maximum percentage of shared words observed was 16.93%, indicating that the majority of the generated words differ when varying the initial word. This highlights the substantial impact that the choice of the initial word has on the generated lyrics.

![word_overlap_initial](https://github.com/user-attachments/assets/d769b2b1-d2e3-4138-bcb9-dd7f7a04745f)

### Melody effect on the generated lyrics:
To analyze the effect of melody on the uniqueness of the generated lyrics, we calculated the average number of unique words shared between different melodies with the same initial word for the same model. We observed that the maximum average percentage of shared words was 16.8%, indicating that the melody significantly influences the generated words.

![melodies_words_overlap](https://github.com/user-attachments/assets/d4066124-a8c5-4255-8e0a-6d978c96b96e)

### Conclusion

In this project, we successfully implemented and trained a neural network to generate song lyrics based on melodies. The integration of melodic information proved beneficial, and the sampling-based approach for word selection enhanced the quality of the generated lyrics. Additionally, the attention mechanism used in the advanced models improved the coherence and overall results of the generated lyrics.

We also observed that the melody significantly influenced the lyrics generated using the features we extracted. This underscores the importance of incorporating melodic features into the model to produce more contextually appropriate and varied lyrics.

Moreover, the use of the Byte Pair Encoding (BPE) approach helped us not only reduce the vocabulary size by more than half but also handle out-of-vocabulary words during inference. This reduction in vocabulary size facilitated more efficient training and improved the model's ability to generate relevant words even when encountering previously unseen terms.

Future work will focus on further improving the model and exploring additional methods for melody-lyric integration.



