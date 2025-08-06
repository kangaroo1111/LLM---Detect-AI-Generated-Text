# LLM---Detect-AI-Generated-Text

Farhad de Sousa and Sanat Mulay 

We investigate various approaches to detect AI generated text. Given a prompt and an essay as input, we want to determine whether the essay was written by a human or by an LLM. 

We explore a baseline strategy based on beaming, and conduct a survey of the effect of temperature on next-word generation. We find that our most successful strategy uses an LLM to embed the text and then trains a classifier to classify the embeddings. 

Datasets:
-Kaggle dataset (smaller, and contains very few human-written samples)
-DAIGT dataset (larger and more diverse)

Methods:
-Output next word probabilities from pretrained LLM and compare to true next word
--probability associated to the true next word is the probability that an LLM generated the text
-Directly use a classification model (eg Mistral for text classification) and train it to classify human written vs AI generated
-Extract last hidden state from pretrained LLM and feed it through a neural net for classification
--Self-attention works such that the last hidden state will contain information about the whole text

Results:
Aproach 1: Next word probablilities
 Using only gpt-2 and 10 split points per text, we get the following results: Test Accuracy: 0.6225
 Classification Report:
               precision    recall  f1-score   support
 
            0     0.7027    0.5752    0.6326       226
            1     0.5535    0.6839    0.6118       174
 
     accuracy                         0.6225       400
    macro avg     0.6281    0.6296    0.6222       400
 weighted avg     0.6378    0.6225    0.6236       400

