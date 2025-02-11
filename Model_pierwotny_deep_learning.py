#!/usr/bin/env python
# coding: utf-8

# # Opis projektu
# 
# 
# Celem projektu jest stworzenie modelu deep learningowego do klasyfikacji emocji na podstawie obrazów twarzy. Model będzie trenowany na zbiorze FER-2013.
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


# In[6]:


###Biblioteki
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import scipy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
print("✅ scipy działa! Wersja:", scipy.__version__)


# In[7]:


## Wczytanie datasetu

dataset_path = r"C:\Users\alaad\OneDrive\Pulpit\deep_learning projekt\faces_train"
train_dir =  f"{dataset_path}/train"
test_dir = f"{dataset_path}/test"


# In[8]:


## Dostosowanie parametrów obrazu
img_size = (48, 48)
batch_size = 64


# In[9]:


## Tworzenie generatorów danych

train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalizacja pikseli do zakresu [0,1]
#    rotation_range=10,  # Mała rotacja do 10 stopni
#    width_shift_range=0.1,  # Delikatne przesunięcie w poziomie (10%)
#    height_shift_range=0.1,  # Delikatne przesunięcie w pionie (10%)
#    horizontal_flip=True  # Odbicie poziome dla zwiększenia różnorodności
)

test_datagen = ImageDataGenerator(rescale=1./255)


# In[10]:


## Zbiór treningowy

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    batch_size=8,
    color_mode="grayscale",  # Jeśli obrazy są w skali szarości
    class_mode="categorical"
)


# In[11]:


## Zbiór testowy

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical"
)


# In[12]:


import os

# Sprawdzenie pierwszych 10 plików w train/
print("Pliki w train/:", os.listdir(train_dir)[:10])

# Sprawdzenie pierwszych 10 plików w pierwszej kategorii
first_category = os.path.join(train_dir, os.listdir(train_dir)[0])
print("Pliki w pierwszej kategorii:", os.listdir(first_category)[:10])


# In[13]:


# Pobierz ścieżkę do pierwszego obrazu z kategorii "angry"
first_category = os.path.join(train_dir, "angry")  # Zmienna train_dir powinna być już zdefiniowana
first_image_path = os.path.join(first_category, os.listdir(first_category)[0])

# Wczytaj obraz w skali szarości
img = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)

# Sprawdzenie czy obraz się wczytał
if img is None:
    print("Błąd: Obraz nie został wczytany poprawnie!")
else:
    # Wyświetlenie obrazu
    plt.imshow(img, cmap="gray")
    plt.title("Przykładowy obraz")
    plt.show()


# In[14]:


print(train_generator)


# In[15]:


print("Liczba obrazów w zbiorze treningowym:", train_generator.samples)
print("Liczba klas:", train_generator.num_classes)



# In[16]:


batch_x, batch_y = next(train_generator)
print("✅ Batch został pobrany poprawnie!")
print("Kształt batcha:", batch_x.shape)


# In[17]:


import matplotlib.pyplot as plt

# Wyświetlenie kilku obrazów z batcha
fig, axes = plt.subplots(1, 5, figsize=(15,5))
for i in range(5):
    axes[i].imshow(batch_x[i].reshape(48, 48), cmap="gray")
    axes[i].axis("off")
plt.show()


# In[18]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# In[19]:


model = Sequential()


# In[20]:


model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))


# In[21]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[23]:


# Druga warstwa konwolucyjna
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))

# Druga warstwa MaxPooling
model.add(MaxPooling2D(pool_size=(2,2)))

# Trzecia warstwa konwolucyjna
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))

# Trzecia warstwa MaxPooling
model.add(MaxPooling2D(pool_size=(2,2)))



# In[24]:


model.add(Flatten())


# In[25]:


model.add(Dense(units=128, activation='relu'))


# In[26]:


model.add(Dropout(0.5))


# In[27]:


model.add(Dense(units=7, activation='softmax'))


# In[28]:


## kompilacja modelu
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# In[29]:


## Trening modelu

epochs = 20  # Liczba epok, możemy potem dostosować

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=epochs
)


# In[30]:


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




