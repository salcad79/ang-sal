import librosa
import tensorflow as tf
from tensorflow import keras
import numpy as np
import soundfile as sf
import sys

conversion = False
checkpoint = False

input_file = "/Users/angelo/Documents/230530LehmanPrediction/Bellshort.wav"
output_file = "/Users/angelo/Documents/230530LehmanPrediction/output_bell.wav"

data, sample_rate = sf.read(input_file, 11025, dtype='int16') #leggo il file a 16bit

data = np.asarray(data)

dati = [] ##converto 'data' da 16bit a 8bit
#print(max(data))
#print(min(data))

for valore in data:
    valore_normalizzato = (valore / (2**16) + 0.5) #tra 0 e 1
    nuovo_valore = chr(int(valore_normalizzato * (2**8) + 10000))
    dati.append(nuovo_valore)

#print(dati)


dati = np.asarray(dati)

#print(max(dati))
#print(min(dati))
#print(dati.shape)
Alphabet = np.asarray(np.unique(dati)) #creo l'alfabeto eliminando i duplicati da 'dati'

#print(Alphabet)

#print(dati.shape)

lenAlphabet = len(Alphabet)
lenDat = len(dati)

#print(lenAlphabet)
#print(lenDat)

#np.set_printoptions(threshold=sys.maxsize)
#np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

#print('Alfabeto ha lunghezza: ', lenAlphabet)

mappa = {}
for i in range(lenAlphabet):
    mappa[Alphabet[i]] = i  # Indice dell'elemento nell'alfabeto

def onehot(U):
    v = np.zeros(lenAlphabet)
    indice = mappa[U]
    v[indice] = 1
    return v

def sample2(prob):
    return np.argmax(np.random.multinomial(1, prob, 1))

x = []
y = []
N = 100

for i in range(lenDat - N): #crea una lista vuota per ogni elemento della lista fino a (lenDat - N) e...
    x_Nelementi = []
    for j in range(N): #... per ogni elemento della lista crea una sottolista composta da N elementi
        x_Nelementi.append(onehot(dati[i + j])) # [[...], [...], [...]] crea una lista di 50 liste che ciascuna contiene 256 valori onehot
    x.append(x_Nelementi) # rappresentazione one-hot delle lettere da i a i+N-1
    y.append(onehot(dati[i + N]))

x = np.array(x)
y = np.array(y)

print(y)

numSequences = x.shape[0]
x_train = x[:int(numSequences * 0.75)]
y_train = y[:int(numSequences * 0.75)]
x_val = x[int(numSequences * 0.75):]
y_val = y[int(numSequences * 0.75):]

model = keras.Sequential([
    keras.layers.Input(shape=(N, lenAlphabet)),
    keras.layers.LSTM(4 * lenAlphabet),
    keras.layers.Dense(lenAlphabet, activation="softmax")
])

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

if False:
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath='/Users/angelo/Documents/230530LehmanPrediction/alicebest.h5',
        monitor='accuracy',
        mode='max',
        save_best_only=True)

    model.fit(x_train, y_train,
        epochs=100,
        batch_size=256,
        validation_data=(x_val, y_val),
        callbacks=[model_checkpoint_callback])

model.load_weights('/Users/angelo/Documents/230530LehmanPrediction/alicebest.h5')

#Predizione
seed = dati[0:N] ## sono 6 lettere, e sono il "seme" iniziale per la mia generazione
seed = ''.join(seed)
#print(seed)

collezione = seed ## qui metto tutte le lettere collezionate
#voglio generare altre 100 lettere
for i in range(10000):
# un esempio solo ha forma (N, lenalphabet)
    xp = []
    for k in seed:
        xp.append(onehot(k))
    xp = np.array(xp)
    xp = np.reshape(xp, (1, N, lenAlphabet))
    yp = model.predict(xp)[0]
    yp = yp.astype('float64')
    yp = yp / np.sum(yp)
    next_index = sample2(yp) # o con sample2(yp)
    next_char = Alphabet[next_index]
    collezione = collezione + next_char ## ci salviamo il nuovo carattere nella collezione di quelli generati
    seed = seed[1:] + next_char
#    print(yp)
print(collezione)

valori_float = [(((ord(chr_val) - 10000) / 2**8) - 0.5)for chr_val in collezione]

print(valori_float)

with sf.SoundFile(output_file, 'w', 11025, 1, 'PCM_U8') as f:
    f.write(valori_float)

