# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv("C:/llm-det/llm-detect-ai-generated-text/train_essays.csv")
pf = pd.read_csv("C:/llm-det/llm-detect-ai-generated-text/train_prompts.csv")

# Prepare the features and labels
# Initialize an empty list to store features
features = []

# Loop over each entry in the dataset
for i in range(len(df)):
    # Get the text and prompt details
    text = df['text'][i]
    prompt_id = df['prompt_id'][i]
    source_text = pf['source_text'][prompt_id]
    instructions = pf['instructions'][prompt_id]
    combined_prompt = source_text + instructions

    # Generate features using predict_v2
    feature = predict_v2(text, combined_prompt, model1)
    features.append(feature)

# Extract labels
labels = df['generated']

# Pad the sequences
features_padded = pad_sequences(
    features, maxlen=10000, padding='post', truncating='post', value=0.0
)

# Convert features and labels into a DataFrame
train_df = pd.DataFrame({
    'features': list(features_padded),
    'labels': labels
})

# Save the prepared data
train_df.to_csv("C:/llm-det/llm-detect-ai-generated-text/train_df.csv", index=False)

# Proceed with data splitting and model training

# Convert features to NumPy array
X = np.array(list(train_df['features']))
y = train_df['labels'].values  # Ensure labels are in NumPy array format

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Determine the input shape based on the number of features
input_shape = (X_train.shape[1],)

# Build the neural network model
model = Sequential()
# Input layer with 64 neurons and ReLU activation
model.add(Dense(64, activation='relu', input_shape=input_shape))
# Output layer with 1 neuron and sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model with optimizer, loss function, and evaluation metric
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# Train the model
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2  # Use 20% of training data for validation
)

# Save the trained model to a file
model.save("C:/llm-det/model.keras")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC AUC: {roc_auc}')