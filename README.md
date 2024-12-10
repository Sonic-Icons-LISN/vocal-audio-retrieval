# Weekly Progress Report

## Week 1

I explored the possibilities of the newest open-source audio generation model called **Stable Audio Open**. I read the associated paper and discussed the reasons for choosing this model. I also reviewed preceding approaches to audio generation, such as:
- Auto and non-autoregressive models
- End-to-end diffusion
- Spectrogram diffusion
- Latent diffusion models

Since this new model offers exceptional sound quality, I decided to download the model, test it, and perform fine-tuning to understand some of the necessary steps and parameters of modern audio generation models.

## Week 2

I took approximately 14,000 audio samples along with their descriptions and wrote custom model and dataset configurations to fine-tune the Stable Audio Open model. After many failures, including null or infinite training losses, I managed to complete 13 epochs of training, which took around 10 hours. During the first few epochs, the results were not clear or improved at all. However, after around 7-8 epochs, the model started generating sounds that were present only in the fine-tuning dataset.

I used **Weights & Biases** to monitor the training process. I chose four demo prompts to make it easier to assess the training performance. These demo prompts were included in the fine-tuning dataset. We could clearly see the improvment in the generated audio sample relevance to the input text.

## Week 3

After ensuring that custom configurations were correctly fine-tuning the model, we moved on to the next step. Our goal is to generate sound effects, particularly with the help of vocal imitations. We need to complete the step of vocal + text embeddings before continuing with sound synthesis.

Fortunately, contrastive language-audio pretraining allows us to associate labels with corresponding audio samples. However, our requirements include using vocal imitations as inputs. Thus, we need to create a different model with three inputs for training: audio label, vocal imitation, and the original sound effect. For inference, we would only provide the audio label and the vocal imitation.

For this process, we need three encoders:
1. **Text Encoder**: Embeds the original audio caption.
2. **Vocal Imitation Encoder**: Maps the vocal imitation to a latent space.
3. **Original Audio Encoder**: Maps the original sound effect to a different latent space.

These encoders MUST be pretrained, although they do not have to be the same, since one is applied to voices and another to environmental sounds.

### Additional Exploration

I also explored the **Encodec** and **Descript Audio Codec** models. However, they were not a good fit for this task for the following reasons:

**Encodec**:
- **Compression Focus**: Primarily designed for audio compression, aiming to reduce the size of audio files while maintaining quality, which limits its applicability for nuanced audio feature extraction and generation.
- **Lack of Pre-trained Models for Fine-Tuning**: Encodec lacks the necessary pre-trained models for fine-tuning on diverse datasets like vocal imitations and environmental sounds.

**Descript Audio Codec**:
- **Designed for Editing**: Built for audio editing, focusing on transcription and waveform manipulation, lacking the deep learning capabilities required for multimodal tasks.
- **Suboptimal Feature Representation**: Not as rich or adaptable as transformer-based models, limiting its effectiveness in capturing complex audio patterns.

### BYOL-A Exploration

I considered **BYOL-A** as well, but it wasn't a good fit due to its sensitivity to noise. Although the authors of the paper tried using k-means clustering and vector quantization to mitigate this sensitivity, these methods add complexity and may not completely resolve the issue. Since we're focusing on audio retrieval, HTSAT is a better fit due to its advanced understanding of long-range dependencies and hierarchical structures in audio data.

### Proposed Architecture

To achieve our goal, I propose the following architecture:

1. **Three Encoders**:
   - A text encoder for audio labels.
   - A specific encoder well-fitted for vocal imitations, leveraging a pre-trained model known for its efficacy with voice samples (HTSAT, wav2vec 2.0, Deep Speech 2).
   - HTSAT for encoding the original audio samples.

2. **Fusion Mechanism**:
   - Implementing a robust fusion layer to combine embeddings from vocal imitations and text descriptions using cross-modal attention.

3. **Projection to Latent Space**:
   - Afterwards we would need to project vocal + text embeddings and original sample embeddings, both to the same dimentional latent space. Research shows that 2 layer MLP is a good fit.

4. **Contrastive Learning Framework**:
   - Using a contrastive loss to align fused embeddings with original audio embeddings, ensuring effective learning from our multimodal dataset.

### Fusion Mechanism: Cross-Modal Attention

The fusion mechanism is crucial for effectively combining the embeddings from the vocal imitation and text encoders. We will use a cross-modal attention mechanism, which allows the model to focus on relevant parts of each modality's embedding while combining them into a unified representation.

**Cross-Modal Attention Details**:
- **Attention Mechanism**: Attention mechanisms, particularly self-attention used in transformers, enable the model to weigh the importance of different parts of the input. Cross-modal attention extends this concept to multiple modalities.
- **Query, Key, Value**: In cross-modal attention, one modality (e.g., text) is used to generate the queries (Q), while the other modality (e.g., vocal imitation) provides the keys (K) and values (V).
- **Attention Calculation**: The attention scores are calculated using the dot-product of the queries and keys, scaled by the square root of the key dimension. These scores are then passed through a softmax function to obtain the attention weights.
- **Fusion Layer**: The output of the attention mechanism, which is a weighted sum of the values, provides a fused representation that incorporates information from both modalities.

### Data Augmentation

Given the limited number of vocal imitation samples, data augmentation is crucial to expand our training dataset and improve model robustness. We will employ several data augmentation techniques:

- **Pitch Shifting**: Altering the pitch of vocal imitations to create variations.
- **Time Stretching**: Changing the speed of vocal imitations without affecting the pitch.
- **Adding Noise**: Introducing background noise to simulate different recording conditions.
- **Vocal Effects**: Applying effects such as reverb, echo, and distortion to diversify the dataset.
- **Synthetic Vocal Imitations**: Generating synthetic vocal imitations using advanced Text-to-Speech (TTS) systems trained to mimic various sounds.

### Text/Caption/Label Augmentation

To further enhance our dataset, we will use the newly released **Llama 3.1 70B** model for text augmentation. This model will help us generate additional captions and labels to enrich our dataset:

- **Synonym Replacement**: Replacing words in the original captions with their synonyms to create new variations.
- **Paraphrasing**: Using Llama 3.1 to generate paraphrased versions of existing captions.
- **Extended Descriptions**: Generating more detailed descriptions of the audio samples
