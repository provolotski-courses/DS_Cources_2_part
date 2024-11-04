import numpy as np
from keras.src.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.layers import Attention

# загружаем данные из файла по-строчно
with open('data/input_text.txt', 'r', encoding='utf-8') as f:
    texts = f.readlines()

# Параметры
sequence_length = 3  # Количество слов в последовательности
embedding_dim = 50   # Размерность эмбеддингов
latent_dim = 64      # Размерность скрытого состояния

# Подготовка данных
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts) #формируем токены
total_words = len(tokenizer.word_index) + 1  #+1 поскольку счет с нуля
sequences = []
# формируем последовательность токенов
for line in texts:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[max(0, i - sequence_length):i + 1]
        sequences.append(n_gram_sequence)

# Дополнение последовательностей нулями для одинаковой длины
sequences = pad_sequences(sequences, maxlen=sequence_length + 1, padding='pre')
X, y = sequences[:, :-1], sequences[:, -1]

# One-hot представление целевых слов
y = to_categorical(y, num_classes=total_words)


# Энкодер
encoder_inputs = Input(shape=(sequence_length,))
x = Embedding(total_words, embedding_dim, input_length=sequence_length)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(x)


# Механизм внимания
attention = Attention()
context_vector = attention([encoder_outputs, encoder_outputs])

# Декодер
decoder_lstm = LSTM(latent_dim)
decoder_outputs = decoder_lstm(context_vector, initial_state=[state_h, state_c])
decoder_dense = Dense(total_words, activation='softmax')
output = decoder_dense(decoder_outputs)


# Создание модели
model = Model(encoder_inputs, output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
model.fit(X, y, epochs=1000, batch_size=8, verbose=1) # отрабатываем на 1000 эпох

# Функция для генерации следующего слова

def predict_next_word(model, tokenizer, text, sequence_length):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = token_list[-sequence_length:]
    token_list = pad_sequences([token_list], maxlen=sequence_length, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return ""


#Рекуррентный автоэнкодер с механизмом внимания для предсказания последующих слов в тексте
# Пример предсказания
input_text = "рекурентный автоэнкодер "
# чтобы не было слишком скучно генерируем последовательно 10 слов
# для данной входной фразы результат: "рекурентный автоэнкодер  состоит из двух основных частей кодировщика и декодировщика результаты случаях"
for i in range(0,10):
    predicted_word = predict_next_word(model, tokenizer, input_text, sequence_length)
    input_text = input_text+' '+ predicted_word

print(f"Итоговая фраза: {input_text}")