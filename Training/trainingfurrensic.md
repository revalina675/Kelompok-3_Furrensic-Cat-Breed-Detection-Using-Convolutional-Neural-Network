_For the information, we are using **kaggle notebook** to run this code._

### STEP 1: IMPORT LIBRARIES

```
!pip install protobuf==3.20.3

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Setting up the Path
image_dir = '/kaggle/input/the-oxfordiiit-pet-dataset/images/images'
CAT_BREEDS = [
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
    'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
    'Siamese', 'Sphynx'
]
```

### STEP 2: DATA PREPARATION & CHECKING

```
filenames = os.listdir(image_dir)
categories = []
valid_filenames = []

for filename in filenames:
    category = "_".join(filename.split("_")[:-1])
    if category in CAT_BREEDS:
        valid_filenames.append(filename)
        categories.append(category)

df = pd.DataFrame({'filename': valid_filenames, 'category': categories})

# Splitting Data (80:20)
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['category'])

# dataset detail
print("\n" + "="*40)
print("="*40)
print(f"Total Initial Data        : {len(df)} images")
print(f"Training Data (80%)       : {len(train_df)} images")
print(f"Validation Data (20%)     : {len(validate_df)} images")
print(f"Number of Classes         : {len(CAT_BREEDS)} Breeds")
print("Split Method               : Stratified Shuffle Split (balanced class distribution)")
```

### STEP 3: PREPROCESSING & GENERATOR

```
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# train & validation the generator
train_generator = train_datagen.flow_from_dataframe(
    train_df, image_dir, x_col='filename', y_col='category',
    target_size=(224, 224), class_mode='categorical', batch_size=32
)

validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, image_dir, x_col='filename', y_col='category',
    target_size=(224, 224), class_mode='categorical', batch_size=32
)
```

### STEP 4: TRAINING PHASE 1 (FROZEN)

```
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(12, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('model_terbaik_raw.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("\ntraining phase 1 (frozen) starting")
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop]
)
```

### STEP 5: TRAINING PHASE 2 (FINE TUNING)

```
base_model.trainable = True
fine_tune_at = 140
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

print("\ntraining phase 2 (fine tuning) starting")
history_fine = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop]
)
```

### STEP 6: REPORT (OPTIONAL)

```
best_model = tf.keras.models.load_model('model_terbaik_raw.keras')

eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
eval_generator = eval_datagen.flow_from_dataframe(
    validate_df,
    image_dir,
    x_col='filename',
    y_col='category',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)
Y_pred = best_model.predict(eval_generator, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
y_true = eval_generator.classes
labels = list(eval_generator.class_indices.keys())

report_dict = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()

df_classes = df_report.iloc[:-3].copy()

cols_metrics = ['precision', 'recall', 'f1-score']
df_classes[cols_metrics] = df_classes[cols_metrics] * 100

df_classes.reset_index(inplace=True)
df_classes.rename(columns={'index': 'Breed Name'}, inplace=True)

df_classes['Accuracy'] = df_classes['recall']

df_classes.rename(columns={
    'precision': 'Precision',
    'recall': 'Recall',
    'f1-score': 'F1-Score'
}, inplace=True)

final_cols = ['Breed Name', 'Accuracy', 'Precision', 'Recall', 'F1-Score']

print("\n" + "="*40)
print("="*40)
print(df_classes[final_cols].to_markdown(index=False, floatfmt=".2f"))

acc_val = report_dict['accuracy'] * 100
prec_val = report_dict['weighted avg']['precision'] * 100
rec_val = report_dict['weighted avg']['recall'] * 100
f1_val = report_dict['weighted avg']['f1-score'] * 100

summary_data = [
    {'Metrics': 'Accuracy', 'Value': f"{acc_val:.2f}%"},
    {'Metrics': 'Precision', 'Value': f"{prec_val:.2f}%"},
    {'Metrics': 'Recall', 'Value': f"{rec_val:.2f}%"},
    {'Metrics': 'F1-Score', 'Value': f"{f1_val:.2f}%"}
]
df_summary = pd.DataFrame(summary_data)

print("\n" + "="*40)
print("="*40)
print(df_summary.to_markdown(index=False))

acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.plot([20, 20], [0, 1], label='Start Fine Tuning', linestyle='--', color='lime')
plt.legend(loc='lower right')
plt.title('Training & Validation Accuracy')
plt.ylim([0.5, 1.0])

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.plot([20, 20], [0, 4], label='Start Fine Tuning', linestyle='--', color='lime')
plt.legend(loc='upper right')
plt.title('Training & Validation Loss')
plt.show()

print("\n" + "="*40)
print("="*40)

eval_generator.reset()
x_batch, y_batch = next(eval_generator)
y_pred_batch = best_model.predict(x_batch)
y_pred_ids = np.argmax(y_pred_batch, axis=1)
y_true_ids = np.argmax(y_batch, axis=1)

plt.figure(figsize=(15, 10))
for i in range(min(9, len(x_batch))):
    plt.subplot(3, 3, i+1)
    img = x_batch[i]
    img = img - img.min()
    img = img / img.max()
    plt.imshow(img)

    pred_lbl = labels[y_pred_ids[i]]
    true_lbl = labels[y_true_ids[i]]

    color = 'green' if pred_lbl == true_lbl else 'red'
    plt.title(f"Asli: {true_lbl}\nAI: {pred_lbl}", color=color)
    plt.axis('off')
plt.tight_layout()
plt.show()
```
