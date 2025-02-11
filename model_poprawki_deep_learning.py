#!/usr/bin/env python
# coding: utf-8

# # Opis projektu
# 
# 
# Celem projektu jest stworzenie modelu deep learningowego do klasyfikacji emocji na podstawie obrazów twarzy. Model będzie trenowany na zbiorze FER-2013, a jego skuteczność zostanie oceniona na nowym zestawie twarzy, np. z filmów.
# 
# W projekcie zostały wykorzystane konwolucyjne sieci neuronowe - CNN. CNN pozwalają na automatyczną ekstrakcję cech, takich jak kontury, tekstury czy układy twarzy, eliminując potrzebę ręcznej inżynierii cech. Dzięki warstwom konwolucyjnym, pooling i gęstym sieć efektywnie uczy się rozpoznawania wzorców związanych z emocjami.
# 
# ## Struktura modelu:
# 
# Warstwy konwolucyjne - wykrywają podstawowe cechy obrazu, np. krawędzie.
# 
# Pooling - zmniejsza rozmiar danych, redukując liczbę parametrów.
# 
# Warstwy gęste (Dense) - przekształcają wyekstrahowane cechy w klasy emocji.
# 
# Softmax - finalna warstwa klasyfikująca obrazy na kategorie emocji.
# 
# Model będzie trenowany przy użyciu optymalizatora Adam i funkcji straty categorical crossentropy. Ocena wyników zostanie przeprowadzona na podstawie dokładności oraz macierzy pomyłek.

# In[1]:


import numpy as np
import cv2
import tensorflow as tf

print("numpy version:", np.__version__)
print("OpenCV version:", cv2.__version__)
print("TensorFlow version:", tf.__version__)


# In[2]:


###Biblioteki
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import scipy
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[3]:


## Wczytanie datasetu

dataset_path = r"C:\Users\alaad\OneDrive\Pulpit\deep_learning projekt\faces_train"
train_dir =  f"{dataset_path}/train"
test_dir = f"{dataset_path}/test"


# In[4]:


## Dostosowanie parametrów obrazu
img_size = (48, 48)
batch_size = 32


# In[5]:


## Eksploracyjna analiza danych zbioru

#Sprawdzenie liczby próbek w danej klasie

labels = os.listdir(train_dir)
class_counts = [len(os.listdir(os.path.join(train_dir, label))) for label in labels]

plt.figure(figsize=(8,5))
sns.barplot(x=labels, y=class_counts)
plt.xlabel("Klasy emocji")
plt.ylabel("Liczba próbek")
plt.title("Rozkład liczby próbek w każdej klasie")
plt.xticks(rotation=30)
plt.show()



# In[6]:


##Wagi emocji

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Klasy emocji
class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Liczba próbek w każdej klasie
samples_per_class = np.array([4000, 500, 4000, 7200, 5000, 4900, 3200])

# Obliczenie wag klasowych
class_weights = compute_class_weight(class_weight="balanced", 
                                    classes=np.unique(np.arange(len(class_labels))),
                                    y=np.repeat(np.arange(len(class_labels)), samples_per_class))
#Konwersja
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

print("Obliczone wagi klas:", class_weights_dict)


# In[7]:


## Sprawdzenie wymiarów obrazów w zbiorze

heights, widths = [], []
for category in os.listdir(train_dir):
    category_path = os.path.join(train_dir, category)
    for img_name in os.listdir(category_path)[:10]:  # Pobieramy po 10 na próbę
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            h, w, _ = img.shape
            heights.append(h)
            widths.append(w)

print(f"Minimalna wysokość obrazów: {min(heights)}, Maksymalna wysokość: {max(heights)}")
print(f"Minimalna szerokość obrazów: {min(widths)}, Maksymalna szerokość: {max(widths)}")


# In[8]:


## Histogram wartości pikseli w jednym obrazie

example_img_path = os.path.join(train_dir, labels[0], os.listdir(os.path.join(train_dir, labels[0]))[0])
example_img = cv2.imread(example_img_path, cv2.IMREAD_GRAYSCALE)

plt.hist(example_img.ravel(), bins=50, color='gray')
plt.title("Histogram wartości pikseli przykładowego obrazu")
plt.xlabel("Wartość piksela")
plt.ylabel("Liczba pikseli")
plt.show()


# In[9]:


## Sprawdzenie rozmiarów obrazów
image_shapes = []

for category in os.listdir(train_dir):
    category_path = os.path.join(train_dir, category)
    for img_name in os.listdir(category_path)[:10]:  # 10 obrazów z każdej klasy
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            image_shapes.append(img.shape[:2])  # Tylko wysokość i szerokość

# DataFrame
import pandas as pd
df_shapes = pd.DataFrame(image_shapes, columns=["Wysokość", "Szerokość"])
print(df_shapes.describe())

# Wykres
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df_shapes["Szerokość"], y=df_shapes["Wysokość"])
plt.xlabel("Szerokość obrazów")
plt.ylabel("Wysokość obrazów")
plt.title("Rozkład rozmiarów obrazów")
plt.show()


# In[10]:


## Tworzenie generatorów danych

train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalizacja pikseli do zakresu [0,1]
    rotation_range=5,
#   width_shift_range=0.1,  # przesunięcie w poziomie
#    height_shift_range=0.1,  # przesunięcie w pionie
    horizontal_flip=True,  # Odbicie poziome dla zwiększenia różnorodności
#    shear_range=0.2,      #przesunięcie kątowe
#    zoom_range=0.2,       #Zoom o zakresie
#    brightness_range=[0.8, 1.2]  #Jasność obrazu
)

test_datagen = ImageDataGenerator(rescale=1./255)


# In[11]:


## Zbiór treningowy

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    batch_size=8,
    color_mode="grayscale",  #skala szarości
    class_mode="categorical"
)


# In[12]:


## Zbiór testowy

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48,48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical"
)


# In[13]:


import os

# Sprawdzenie pierwszych 10 plików w train/
print("Pliki w train/:", os.listdir(train_dir)[:10])

# Sprawdzenie pierwszych 10 plików w pierwszej kategorii
first_category = os.path.join(train_dir, os.listdir(train_dir)[0])
print("Pliki w pierwszej kategorii:", os.listdir(first_category)[:10])


# In[14]:


# Pobieranie ścieżki do pierwszego obrazu z kategorii "angry"
first_category = os.path.join(train_dir, "angry")
first_image_path = os.path.join(first_category, os.listdir(first_category)[0])

# Wczytanie obraz w skali szarości
img = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)

# Sprawdzenie czy obraz się wczytał
if img is None:
    print("Błąd")
else:
    # Wyświetlenie obrazu
    plt.imshow(img, cmap="gray")
    plt.title("Przykładowy obraz")
    plt.show()


# In[15]:


print(train_generator)


# In[16]:


print("Liczba obrazów w zbiorze treningowym:", train_generator.samples)
print("Liczba klas:", train_generator.num_classes)



# In[17]:


batch_x, batch_y = next(train_generator)
print("Kształt batcha:", batch_x.shape)


# In[18]:


import matplotlib.pyplot as plt

# Wyświetlenie kilku obrazów z batcha
fig, axes = plt.subplots(1, 5, figsize=(15,5))
for i in range(5):
    axes[i].imshow(batch_x[i].reshape(48, 48), cmap="gray")
    axes[i].axis("off")
plt.show()


# In[19]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization



# In[20]:


model = Sequential()


# In[21]:


model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))


# In[22]:


model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


# In[23]:


# Druga warstwa konwolucyjna
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))


# Druga warstwa MaxPooling
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Trzecia warstwa konwolucyjna
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))


# Trzecia warstwa MaxPooling
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))


# In[24]:


# Czwarta warstwa konwolucyjna
model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


# In[25]:


model.add(Flatten())


# In[26]:


model.add(Dense(units=256, activation='relu'))


# In[27]:


model.add(BatchNormalization())


# In[28]:


model.add(Dropout(0.3))


# In[29]:


model.add(Dense(units=7, activation='softmax'))


# In[30]:


## kompilacja modelu

from tensorflow.keras.optimizers import Adam

model.compile(
    loss='categorical_crossentropy',
#    optimizer=Adam(learning_rate=1e-4),
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    metrics=['accuracy']
)


# In[31]:


## Trening modelu
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=5,          
    restore_best_weights=True  
)


from tensorflow.keras.callbacks import ReduceLROnPlateau

#lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=0.00001)
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)

epochs = 30  # Liczba epok

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=epochs,
    class_weight=class_weights_dict,
#    callbacks=[reduce_lr,early_stopping]
    callbacks=[early_stopping]
)


# In[32]:


## Wizualizacja wyników
import matplotlib.pyplot as plt

# Rysowanie wykresu dokładności
plt.plot(history.history['accuracy'], label='Dokładność treningowa')
plt.plot(history.history['val_accuracy'], label='Dokładność walidacyjna')
plt.xlabel('Epoki')
plt.ylabel('Dokładność')
plt.legend()
plt.show()

# Rysowanie wykresu straty
plt.plot(history.history['loss'], label='Strata treningowa')
plt.plot(history.history['val_loss'], label='Strata walidacyjna')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




