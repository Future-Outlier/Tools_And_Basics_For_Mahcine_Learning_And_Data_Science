#%% 製作長週期資料
from tensorflow.keras.utils import to_categorical

data = [0,1,1,1,1,1,1,2,2,2,2,3,3] * 5  #←週期為 13 的資料
data = to_categorical(data)  #←將資料轉成 one hot 編碼

# %% 處理時序資料
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

data_gen = TimeseriesGenerator(data, 
                               data,
                               length=1,  #←時間長度設定為 1
                               batch_size=1)  #←設定 batch_size 為 1, 這樣就能直接使用 fit() 來訓練

print(data_gen[0])

# %% 建立 Stateful RNN 模型
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

model = Sequential()
model.add(layers.SimpleRNN(10,
                           stateful=True,  #←啟用 Stateful RNN
                           batch_input_shape=(1,None,4)  #←(批次量大小, 時間長度, 特徵數)
                           ))
model.add(layers.Dense(4, activation='softmax'))
model.summary()

#%% 編譯、訓練模型
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

epochs = 50    #←設定訓練週期為 50 次


for i in range(epochs):    #←使用 for 迴圈來控制訓練週期
    print('Epoch', i + 1, '/', epochs)
    model.fit(data_gen,
              epochs=1,    #←訓練完一個週期即停止
              shuffle=False)   #←設定為 False, 避免時序被打亂
    model.reset_states()    #←重置 Stateful RNN 的狀態

#%% 輸出預測結果
model.reset_states()  #←一開始先重置狀態
out_put = data_gen[0][0]    #←只使用第一筆資料進行預測
 
for i in range(50):    #←循環預測 50 次
    prediction = model.predict_classes(out_put, batch_size=1)    #←預測類別
    print(prediction)    #←顯示預測結果
    out_put = to_categorical(prediction, num_classes=4).reshape(1,-1,4)   #←將類別轉成 one hot 編碼, 並將 shape 轉為 RNN 能接受的 3D 陣列

# %%
