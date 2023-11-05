#%%
import os
import autokeras as ak
from kerastuner.engine import hyperparameters
TRAIN_DIR = 'D:/Workspace_Python/DeepLearning/Fruit_classification/dataset/train'
VALIDATION_DIR = 'D:/Workspace_Python/DeepLearning/Fruit_classification/dataset/validation'

# Bilinear, bicubic, lanczos5, mitchellcubic.
INTERPOLATION = "bilinear"

# Open the log file in write mode. The 'a' flag is used to append to the file if it exists, or create a new file if it doesn't.
log_file = open("D:/Workspace_Python/DeepLearning/Fruit_classification/model/autokeras.log", "a")

# Adding new line for better readability in the log file
log_file.write("Load datasets\n")
print("Load datasets")
train_dataset = ak.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(64, 64),
    color_mode="rgb",
    batch_size=32,
    shuffle=True,
    seed=42,
    validation_split=None,
    interpolation=INTERPOLATION,
)

test_dataset = ak.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=(64, 64),
    color_mode="rgb",
    batch_size=32,
    shuffle=True,
    seed=42,
    validation_split=None,
    interpolation=INTERPOLATION,
)

# Initialize the image classifier.
log_file.write("Initialize the image classifier\n")
print("Initialize the image classifier")

clf = ak.ImageClassifier(
    project_name="fruit_classifier", 
    directory='model/',
    objective="val_accuracy",
    seed=42,
    overwrite=False,
    max_trials=1,
)

#%%
# Feed the image classifier with training data.
log_file.write("Start fitting\n")
print("Start fitting")

clf.fit(
    train_dataset,
    epochs=1,
    validation_split=0.20,
)
# # Export the model
log_file.write("Export the model\n")
print("Export the model")
clf.export_model()
log_file.write("Model exported\n")
# Summarize the model architecture
clf.summary()

# Access the model's layers
for layer in clf.layers:
    print(layer)
    print("Layer name:", layer.name)
    print("Layer type:", type(layer))
    print("Layer weights:", layer.get_weights())
    print()

# Save the model's architecture to a file
with open('model_architecture.json', 'w') as f:
    f.write(clf.to_json())
print(clf.to_json())
# Access the KerasTuner object
tuner = clf.tuner

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters()[0]
log_file.write(best_hps.values)
# # Predict with the best model.
# log_file.write("Predict\n")
# print("Predict")
# #predicted_y = clf.predict(test_dataset, batch_size=32)

# #print(predicted_y)

# # Evaluate the best model with testing data.
# log_file.write("Evaluate the model\n")
# print("Evaluate the model")
# res = clf.evaluate(test_dataset, batch_size=32)

# accuracy = res["accuracy"]
# precision = res["precision"]
# recall = res["recall"]
# f1_score = res["f1_score"]

# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1_score}")

# log_file.write(f"Accuracy: {accuracy}\n")
# log_file.write(f"Precision: {precision}\n")
# log_file.write(f"Recall: {recall}\n")
# log_file.write(f"F1 Score: {f1_score}\n")



# # It's good practice to close the file when you're done with it
# log_file.close()
# # Value             |Best Value So Far |Hyperparameter
# # vanilla           |vanilla           |image_block_1/block_type
# # True              |True              |image_block_1/normalize
# # False             |False             |image_block_1/augment
# # 3                 |3                 |image_block_1/conv_block_1/kernel_size
# # 1                 |1                 |image_block_1/conv_block_1/num_blocks
# # 2                 |2                 |image_block_1/conv_block_1/num_layers
# # True              |True              |image_block_1/conv_block_1/max_pooling
# # False             |False             |image_block_1/conv_block_1/separable
# # 0.25              |0.25              |image_block_1/conv_block_1/dropout
# # 32                |32                |image_block_1/conv_block_1/filters_0_0
# # 64                |64                |image_block_1/conv_block_1/filters_0_1
# # flatten           |flatten           |classification_head_1/spatial_reduction_1/reduction_type
# # 0.5               |0.5               |classification_head_1/dropout
# # adam              |adam              |optimizer
# # 0.001             |0.001             |learning_rate