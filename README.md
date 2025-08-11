# LLM---Detect-AI-Generated-Text

_Farhad de Sousa and Sanat Mulay_ 

We investigate various approaches to detect AI generated text. Given a prompt and an essay as input, we want to determine whether the essay was written by a human or by an LLM. 

We explore a baseline strategy based on beaming, and conduct a survey of the effect of temperature on next-word generation. We find that our most successful strategy uses an LLM to embed the text and then trains a classifier to classify the embeddings. 

## Datasets:
- Kaggle dataset (smaller, and contains very few human-written samples) 
- DAIGT dataset (larger and more diverse) 

## Methods:
- Output next word probabilities from pretrained LLM and compare to true next word 
  - probability associated to the true next word is the probability that an LLM generated the text 
- Extract last hidden state from pretrained LLM and feed it through a Random Forest Classifier 
  - Self-attention works such that the last hidden state will contain information about the whole text 

## Results: 
### Aproach 1: Next word probablilities 
 Using only **gpt-2** and **10 split points** per text, we get the following results: 
 ```
 Test Accuracy: 0.6225 
 Classification Report: 
 
               precision    recall  f1-score   support 
 
            0     0.7027    0.5752    0.6326       226
            1     0.5535    0.6839    0.6118       174

     accuracy                         0.6225       400
    macro avg     0.6281    0.6296    0.6222       400
 weighted avg     0.6378    0.6225    0.6236       400
 ```

 ### Approach 2: Extract last hidden state 
 - **Model: gpt-2** \
 Results:
```
 Test Accuracy: 0.9500

Confusion Matrix:
            Pred Human  Pred AI
True Human         223        3
True AI             17      157
```

- **Model: llama_3_2_1B** \
  Results:
```
Test Accuracy: 0.9725

Confusion Matrix:
            Pred Human  Pred AI
True Human         223        3
True AI              8      166
```

- **Model: gemma-2-2b** \
  Results:
```
Test Accuracy: 0.9750

Confusion Matrix:
            Pred Human  Pred AI
True Human         224        2
True AI              8      166
```

## Additional Observations:
- gpt-2 identified ~750 features, out of which 6 had an importance score > 0.015
- llama identified ~2000 features, only 4 of which had an importance score > 0.015
- the accuracy roughly decreases with respect to text length:
```
(statistics shown for gpt-2)
   Accuracy by Text-Length Category:
    Category '300-399':
      Number of samples = 118
      Accuracy = 0.9492
    Category '400-499':
      Number of samples = 72
      Accuracy = 0.9583
    Category '<300':
      Number of samples = 131
      Accuracy = 0.9618
    Category '≥500':
      Number of samples = 79
      Accuracy = 0.9241
```

## Future Directions: 
- Directly use a classification model (eg Mistral for text classification) and train it to classify human written vs AI generated
- Test more models
- Ensemble models: classify using last hidden states from multiple models and take a weighted vote
- True next word (baseline startegy) but with more split points, and compare with multiple models if computational resources allow









