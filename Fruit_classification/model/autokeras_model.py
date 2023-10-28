import autokeras as ak

TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/test'

#bilinear, bicubic, lanczos5, mitchellcubic.
INTERPOLATION="bilinear"

train_dataset = ak.image_dataset_from_directory(TRAIN_DIR,
                                image_size=(64, 64),
                                color_mode="rgb",
                                batch_size=32,
                                shuffle=True,
                                seed=42,
                                validation_split=None,
                                interpolation=INTERPOLATION,)


test_dataset = ak.image_dataset_from_directory(TEST_DIR,
                                image_size=(64,64),
                                color_mode="rgb",
                                batch_size=32,
                                shuffle=True,
                                seed=42,
                                validation_split=None,
                                interpolation=INTERPOLATION,)


# Initialize the image classifier.
clf = ak.ImageClassifier(
    project_name="logo_classifier",
    directory='/home/ubuntu/workspace/finovox_main/dl_project',
    objective="val_accuracy",
    seed=42,
    overwrite=False,
    max_trials=5)

# Feed the image classifier with training data.
clf.fit(train_dataset,
            epochs=3,
            validation_split=0.20,
        )

# Predict with the best model.
predicted_y = clf.predict(test_dataset, batch_size=32)

print(predicted_y)

res = clf.evaluate(test_dataset, batch_size=32)
# Evaluate the best model with testing data.
accuracy = res["accuracy"]
precision = res["precision"]
recall = res["recall"]
f1_score = res["f1_score"]

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")


clf.export_model()
