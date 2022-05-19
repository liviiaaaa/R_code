#встановлюю бібліотеку keras
install.packages("keras")
library(keras)
install_keras()

#завантажую датасет
mnist = dataset_fashion_mnist()
str(mnist)
#розбиваю вибірку на тренувальну та тестову
c(c(x_train, y_train), c(x_test, y_test)) %<-% mnist
x_train[5,,]
image(x_train[5,,]) # Бачимо, що картинка повернута на бік
#повертаємо картинку так,як нам потрібно
plotImage = function(im) {
  image(t(apply(im, 2, rev)))
}
plotImage(x_train[5,,])
#Підготовка датасету
#приведемо значення пікселів в інтервал (0,1)
x_train = array_reshape(x_train, c(60000, 28, 28, 1))
x_train = x_train / 255
x_test  = array_reshape(x_test, c(10000, 28, 28, 1))
x_test  = x_test / 255
#перетворюємо відгук у one-hot вектори
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)
y_train
#Будуємо  щільну нейронну мережу
#Задаємо модель
denseModel = keras_model_sequential() %>%
  #задаємо різні шари нейронної мережі
  layer_flatten(input_shape = c(28,28,1)) %>%  #подаємо на вхід матриці, перетворюємо їх у масиви
  layer_dense(units = 64, activation = "relu") %>%  #додаємо 1 прихований шар до нейр.мережі
  layer_dense(units = 10, activation = "softmax")   #класифікуючий шар
denseModel
#компілюємо модель
denseModel %>% compile(
  optimizer = "adam",     #вибрали оптимізатор
  loss = "categorical_crossentropy",
  metrics = c("accuracy") 
)
#робимо підгонку моделі
denseModel %>% fit(
  x_train, y_train,
  epochs = 10, batch_size=64
)
#перевіряємо результат на тестових даних
result = evaluate(denseModel, x_test, y_test)
result
#пробуємо покращити модель,додавши ще один шар batch_normalization
denseModel = keras_model_sequential() %>%
  layer_flatten(input_shape = c(28,28,1)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dense(units = 10, activation = "softmax")
denseModel
compile(denseModel, optimizer="adam", loss="categorical_crossentropy", metrics=c("accuracy"))
fit(denseModel, x_train, y_train, epochs=20, batch_size=64)
#перевіряємо результат на тестових даних
result = evaluate(denseModel, x_test, y_test)
result
#пробуємо покращити модель,додавши ще один шар dropout
denseModel = keras_model_sequential() %>%
  layer_flatten(input_shape = c(28,28,1)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(units = 10, activation = "softmax")
denseModel
compile(denseModel, optimizer="adam", loss="categorical_crossentropy", metrics=c("accuracy"))
fit(denseModel, x_train, y_train, epochs=10, batch_size=64)
#перевіряємо результат на тестових даних
result = evaluate(denseModel, x_test, y_test)
result
#пробуємо покращити модель,змінивши batch_size до 128
denseModel = keras_model_sequential() %>%
  layer_flatten(input_shape = c(28,28,1)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(units = 10, activation = "softmax")
denseModel
compile(denseModel, optimizer="adam", loss="categorical_crossentropy", metrics=c("accuracy"))
fit(denseModel, x_train, y_train, epochs=10, batch_size=128)
#перевіряємо результат на тестових даних
result = evaluate(denseModel, x_test, y_test)
result
#в початковій моделі змінюємо learning rate 
denseModel = keras_model_sequential() %>%
  layer_flatten(input_shape = c(28,28,1)) %>%  #подаємо на вхід матриці, перетворюємо їх у масиви
  layer_dense(units = 64, activation = "relu") %>%  #додаємо 1 прихований шар до нейр.мережі
  layer_dense(units = 10, activation = "softmax")   #класифікуючий шар
denseModel
denseModel %>% compile(loss = "categorical_crossentropy", optimizer = optimizer_adam(lr = 1e-2), metrics = c("accuracy"))
denseModel %>% fit(x_train, y_train, epochs=10, batch_size=64)
#перевіряємо результат на тестових даних
result = evaluate(denseModel, x_test, y_test)
result
#в початковій моделі вибираємо інший оптимізатор 
denseModel = keras_model_sequential() %>%
  layer_flatten(input_shape = c(28,28,1)) %>%  #подаємо на вхід матриці, перетворюємо їх у масиви
  layer_dense(units = 64, activation = "relu") %>%  #додаємо 1 прихований шар до нейр.мережі
  layer_dense(units = 10, activation = "softmax")   #класифікуючий шар
denseModel
denseModel %>% compile(loss = "categorical_crossentropy", optimizer = optimizer_adadelta(lr = 1), metrics = c("accuracy"))
denseModel %>% fit(x_train, y_train, epochs=10, batch_size=64)
#перевіряємо результат на тестових даних
result = evaluate(denseModel, x_test, y_test)
result
#Будуємо згорткову мережу,скориставшись functional API
inputs = layer_input(shape=c(28,28,1))
z = layer_conv_2d(inputs, filters=32, kernel_size=c(3,3), activation="selu")
z = layer_max_pooling_2d(z, pool_size = c(2, 2))
z = layer_conv_2d(z, filters = 64, kernel_size = c(3, 3), activation = "relu")
z = layer_max_pooling_2d(z, pool_size = c(2, 2))
z = layer_conv_2d(z, filters = 64, kernel_size = c(3, 3), activation = "relu")
outputs = z
model = keras_model(inputs, outputs)
#додаємо останній шар для класифікації
z = layer_flatten(z)
z = layer_dense(z, units = 64, activation = "selu")
outputs = layer_dense(z, units = 10, activation = "softmax")
model = keras_model(inputs, outputs)
model
compile(model, optimizer="adam", loss="categorical_crossentropy", metrics=c("accuracy"))
fit(model, x_train, y_train, epochs=10, batch_size=64)
results = model %>% evaluate(x_test, y_test)
results


