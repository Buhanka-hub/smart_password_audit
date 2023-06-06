#импортирование библиотек
import numpy as np 
import re
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, SimpleRNN, Dense, LSTM
from numba import njit, typed

def restore_NN():
    file = input("Разместите файл с паролями в директории исполняемого файла. После этого, введите название файла: ")
    #чтение файла с паролями
    with open('./'+file) as f:
        text = f.read()
        text = re.sub(r'[^a-zA-Z]', '', text)
    #инициализация начальных значений
    num_characters = 48
    inp_chars=2
    # кодирование символов паролей в OHE
    tokenizer = Tokenizer(num_words=num_characters, char_level=True, lower=False)
    tokenizer.fit_on_texts(text)
    data = tokenizer.texts_to_matrix(text)
    n = data.shape[0]-inp_chars
    #формирование обучающей и тестовой выборки методом скользящего окна
    @njit()
    def training_data():
        X = [data[j:j+inp_chars, :] for j in range(0,4)]
        Y = data[inp_chars:6]
        for i in range(6,n,6):
            
            X0 = [data[j:j+inp_chars:] for j in range(i,i+4)]
            X.extend(X0) 
            Y0 = data[i+inp_chars:i+6]
            Y = np.append(Y, Y0, axis=0)
        return X,Y
    X,Y = training_data()
    print(len(X), len(Y))
    #создание модели из 4 слоёв
    model = Sequential()
    model.add(Input((inp_chars, num_characters))) 
    model.add(LSTM(256, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(LSTM(64))
    model.add(Dense(num_characters, activation='softmax'))
    model.summary()
    #задание парметров обучения. Запуск обучения
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    history = model.fit(X, Y, batch_size=32, epochs=100, verbose=2, validation_split=0.25)
    #рисование графиков
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    #сохранение модели в файл
    model.save('./user_model.h5')

def calc_NN():
    model = load_model('./user_model.h5')
    #функция возвращает вероятность следующих символов
    def prob_next_char(inp_str):
        x = []
        for i in inp_str:
            x.append(tokenizer.texts_to_matrix(i))
        x = np.array(x)
        inp = x.reshape(1, inp_chars, num_characters)
        mas = []
        pred = model.predict( inp ) 

        for k in range(1,len(pred[0])):
            mas.append((tokenizer.index_word[k], pred[0][k]))    
        return mas
    #функция формирует массив паролей с символом на 1 больше
    @njit()
    def next_char(mas):
        next_mas = [("",0)]
        next_mas.clear()
        
        for i in mas:
            next_ch = prob_next_char(i[0][-2:]) #возращает результат НС
            tmp = [(i[0]+j[0], i[1]*j[1]) for j in next_ch]
            next_mas += tmp
        return next_mas
    user_pass = input("Введите проверяемый пароль: ")
    mas_2_char = [user_pass[:2],1]
    mas_3_char = next_char(mas_2_char)
    mas_4_char = next_char(mas_3_char)
    mas_5_char = next_char(mas_4_char)
    mas_6_char = next_char(mas_5_char)
    return mas_6_char, user_pass
#функция вычисляет результат оценки стойкости пароля и выводит на экран
def result(mas_6_char, user_pass):
    mas_6_char.sort(key=lambda i: -i[1]) #сортировка массива по уменьшению вероятностей появления
    index_NN = 0
    index_Template = 0
    for i in range(len(mas_6_char)): #вычисления позиции пользовательского пароля
        if mas_6_char[i][0] == user_pass:
            index_NN = i
            break
    pass_len=len(user_pass)
    alp = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(2, pass_len): #вычисления позиции пароля в списке для полного перебора
        for j in range(len(alp)):
            if user_pass[i] == alp[j]:
                index_Template += j*(52**(pass_len-i-1))
                break
    # блок вывода результатов
    print("Первые 20 паролей:")
    print(mas_6_char[:20])
    print("\nпослдение 5 паролей:")
    print(mas_6_char[-5:])
    print("В полученном списке пароль находится на",index_NN,"месте")
    print("При использовании списка для полного перебора пароль занимает", index_Template,"место")
    print("Если расчитывать, что скорость перебора пароля 1000 паролей/сек., \
          то подбор пароля занял бы в первом случае",index_NN/1000,"секунд")
    print("во втором случае",index_Template/1000,"секунд")
    print("Вероятности пароля:")
    print("в полученном списке: ",mas_6_char[index_NN][1])
    print("в списке для полного перебора: ",1/(52**len(user_pass)))
    file = ""
    for i in mas_6_char:
        file += i[0]+"\n"
    with open("password_list.txt","w") as f: #сохраняем пароли в файл
        f.write(file)
    print("Пароли сохранены в файл password_list.txt")
#точка начала выполнения программы
print("Данная программа вычисляет стойкость пароля")
restore = True
lp = True
while lp:
    tmp = input("Хотите переобучить модель на собственном наборе паролей?(Y/n): ")
    if tmp in ("Y","y"):
        restore = True
        lp = False
    elif tmp in ("N", "n"):
        restore = False
        lp = False
    else:
        print("Ответ введен некорректно.")
        lp = True
if restore:
    restore_NN()
recalc = True
while recalc:
    print("Подождите, пока идут вычисления. Это может занять несколько минут...")
    mas_6_char, user_pass = calc_NN()
    result(mas_6_char, user_pass)
    lp = True
    while lp:
        tmp = input("Желаете проверить другой пароль?(Y/n): ")
        if tmp in ("Y","y"):
            recalc = True
            lp = False
        elif tmp in ("N", "n"):
            recalc = False
            lp = False
        else:
            print("Ответ введен некорректно.")
            lp = True
