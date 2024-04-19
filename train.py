X = train_df.drop('labels', axis=1)
y = train_df['labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the regression model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(max_sequence_length,)))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['roc_auc'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

model.save("C:/llm-det/model.keras")

# Evaluate the model using ROC AUC
y_pred = model.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC AUC: {roc_auc}')