library(mlbench) #підключаю необхіднi бібліотекu
library(class)
library(e1071)
library(fastAdaboost)
library(C50)
library(randomForest)

data_set<-read.csv("/Users/Livia/Desktop/amn-lab/lab2/abalone.data", header=TRUE) #експортуэмо дані з csv-файла
data<-data_set[,2:ncol(data_set)] #робимо класифікацію над числовими змінними(відкидаю факторну змінну Sex)
names(data)<-c("Length","Diameter","Height","Whole_weight","Shucked_weight","Viscera_weight","Shell_weight","Rings")
data
sum(is.na(data)) #перевіряю чи є пропущені дані
dataX = data.matrix(data[,1:7]) #матриця регресорів

#Відгук Rings розбиваю на 3 групи: Rings<8, Rings є [8;12], Rings > 12.
rings_groups<-cut(data$Rings,breaks = c(0,7,12,Inf),
         labels = c(0,1,2))
#розбила відгук на 3 групи і присвоїла кожній групі окреме зн-ня : 0,1 або 2
data$Rings <- rings_groups #тепер відгук Rings приймає нові зн-ня : 0,1,2 
Y = as.numeric(data$Rings=="0")
X = as.numeric(data$Rings=="1")
Z = as.numeric(data$Rings=="2")
length(Y)
sum(Y)
sum(X)
sum(Z)
#перевіряю, яке віднош-ня між трьома класами відгуку
#скільки даних потрапило в  кожен клас
sum(Y)/length(Y) #в клас 0 - 20%
sum(X)/length(Y) #в клас 1 - 63%
sum(Z)/length(Y) #в клас 2 - 17%
#результат непоганий

#attach(data)#приєднуємо дані
#data
#____розбиваю вибірку на тестову та тренувальну______

tr.index = sample(1:nrow(data), nrow(data)*0.8)
trSet = data[tr.index, ]  #тренувальна вибірка
testSet = data[-tr.index, ]  #тестова вибірка
trX = data.matrix(trSet[,1:7])  #тренувальна вибірка тільки з регресорів
trY = trSet$Rings       #відгук по тренувальній вибірці
testX = data.matrix(testSet[,1:7]) #тестова вибірка тільки з регресорів
testY = testSet$Rings    #відгук по тестовій вибірці
#перевіряю, яке віднош-ня між трьома класами відгуку на тренувальній і тестовій вибірці
#тобто скільки відсотків даних потрапило в кожен клас при розбитті
table(trSet$Rings)/sum(table(trSet$Rings))
table(testSet$Rings)/sum(table(testSet$Rings))
#результати вийшли майже однаковими,тому все добре. дані розбились правильно


#_______логістична регресія(мультиноміальна)________
#по тренувальній вибірці
#Встановлюю базовий клас
trSet$Rings <- relevel(trSet$Rings, ref = "0")
require(nnet) #підключаю пакет nnet
#будую модель на тренувальних даних
multinom.fit <- multinom(Rings ~ ., data = trSet)
#Перевіряю роботу моделі
summary(multinom.fit)
#Прогнозування значень для тренувальної вибірки
trSet$precticed <- predict(multinom.fit, newdata = trSet, "class")
#Будую таблицю класифікації
ctable <- table(trY, trSet$precticed)
ctable
#Обчислюю точність моделі на тренувальних даних
round((sum(diag(ctable))/sum(ctable))*100,2) #Accuracy : 76%

###########спробувала вилучити деякі регресори
trSet$Rings <- relevel(trSet$Rings, ref = "0")
require(nnet) #підключаю пакет nnet
#будую модель на тренувальних даних
multinom1.fit1 <- multinom(Rings ~ Length+Height+Whole_weight, data = trSet)
#Перевіряю роботу моделі
summary(multinom1.fit1)
#Прогнозування значень для тренувальної вибірки
trSet$precticed <- predict(multinom1.fit1, newdata = trSet, "class")
#Будую таблицю класифікації
c2table <- table(trY, trSet$precticed)
c2table
#Обчислюю точність моделі на тренувальних даних
round((sum(diag(c2table))/sum(c2table))*100,2) #Accuracy : 72.31%
###########
#перевіряю модель на тестових даних
testSet$precticed <- predict(multinom.fit, newdata = testSet, "class")
#Будую таблицю класифікації
c1table <- table(testY, testSet$precticed)
c1table
#Обчислюю точність моделі на тестових даних ##77.99%
round((sum(diag(c1table))/sum(c1table))*100,2)

#--------------класифікація методом k-nn---------------
result = knn(trX, testX, cl=trY)
#будую таблицю класифікації
conf.matrix = table(predicted=result, true=testY)
conf.matrix
#Обчислюю точність моделі для k=1
sum(diag(conf.matrix)/sum(conf.matrix))
#вибираю k=64
result1 = knn(trX, testX, cl=trY, k=64)
#будую таблицю класифікації
conf.matrix1 = table(predicted=result1, true=testY)
conf.matrix1
#Обчислюю точність моделі для k=64
sum(diag(conf.matrix1)/sum(conf.matrix1))
#вибираю k=15
result2 = knn(trX, testX, cl=trY, k=15)
#будую таблицю класифікації
conf.matrix2 = table(predicted=result2, true=testY)
conf.matrix2
#Обчислюю точність моделі для k=15
sum(diag(conf.matrix2)/sum(conf.matrix2))

#________НАЇВНИЙ БАЙЄСІВСЬКИЙ КЛАСИФІКАТОР_________
model = naiveBayes(trX, as.factor(trY))
hatYtr  = predict(model, trX) #роблю прогноз
hatYtest= predict(model, testX)
#будую талицю класиф. для тренувальної вибірки
table(hatYtr, trY)
#будую талицю класиф. для тестової вибірки
table(hatYtest, testY)
#обчислюю точність моделей на двох вибірках
print(sprintf("Training accuracy: %3.3f", sum(hatYtr==trY)/length(trY)))
print(sprintf("Test accuracy: %3.3f", sum(hatYtest==testY)/length(testY)))
dataY <- data$Rings
dataY
#___________МЕТОД svm________
# ----- Робота з усією вибіркою--------
model = svm(dataY ~ dataX, cost=0.75, kernel="radial"); #створюю модель використовуючи svm
prediction =model$fitted #прогноз
#перевіряю точність моделі на двох базових заданих параметрах kernel i cost
print(sprintf("Accuracy on the test set: %.3f", sum(as.numeric(prediction == dataY))/length(dataY)))
#стараюся підібрати параметри краще,за допомогою ф-ї tune()
tunedModel = tune(svm, dataY ~ dataX, ranges = list(cost = c(1, 5, 10,50), kernel=c("linear", "radial")));
tunedModel
tunedModel2 = tune(svm, dataY ~ dataX, ranges = list(cost = c(5, 10, 50), gamma=c(0.1, 1, 5)), kernel="radial");
tunedModel2
tunedModel3 = tune(svm, dataY ~ dataX, ranges = list(cost = c(5, 10, 50), degree=c(2, 3, 4, 5)), kernel="polynomial");
tunedModel3
# ------- Робота з тренувальною вибіркою
d = trX
tuned = tune(svm, trY ~ trX, ranges = list(cost = c(1,5, 10, 50,100), degree=c(2, 3, 4, 5)), kernel="polynomial");
tuned
model = svm(trY ~ d, cost=100, degree=3, kernel="polynomial")
#--------Перевіряю роботу моделі на тестовій вибірці
d = testX
hatYtest = predict(model, d)
print(sprintf("Accuracy on the test set: %3.3f", sum(as.numeric(as.numeric(hatYtest  == testY)))/length(testY)))
#--------- sampling="fix"
data = trX
model = tune(svm, trY ~ data, ranges = list(cost = c(5, 10, 50), degree=c(2, 3, 4, 5)), kernel="polynomial", tunecontrol=tune.control(sampling="fix"))
summary(model)
data = testX
hatY = predict(model$best.model, data)
print(sprintf("Accuracy on the test set: %.3f", sum(as.numeric(hatY == testY))/length(testY)))
#--------- sampling="fix" with given validation set size
control = tune.control(sampling="fix", fix=0.1)
data = trX
model = tune(svm, trY ~ data, ranges = list(cost = c(5, 10, 50), degree=c(2, 3, 4, 5)), kernel="polynomial", tunecontrol=control)
summary(model)
data = testX
hatY = predict(model$best.model, data)
print(sprintf("Accuracy on the test set: %.3f", sum(as.numeric(hatY == testY))/length(testY)))
#---------- sampling="bootstrap" -----------
control = tune.control(sampling="bootstrap", boot.size=0.9)
data = trX
model = tune(svm, trY ~ data, ranges = list(cost = c(5, 10,50), degree=c(2, 3, 4, 5)), kernel="polynomial", tunecontrol=control)
summary(model)
data = testX
hatY = predict(model$best.model, data)
print(sprintf("Accuracy on the test set: %.3f", sum(as.numeric(hatY == testY))/length(testY)))
#---------- sampling="cross" --------------
control = tune.control(sampling="cross", cross=10)
data = trX
model = tune(svm, trY ~ data, ranges = list(cost = c(50, 60), degree=c(2, 3, 4, 5)), kernel="polynomial", tunecontrol=control)
summary(model)
data = testX
hatY = predict(model$best.model, data)
print(sprintf("Accuracy on the test set: %.3f", sum(as.numeric(hatY == testY))/length(testY)))

plot(model) #будую графік моделі
# --------- C50 -----------дерева----
#будую модель дерев
model = C5.0(Rings ~., data=trSet)
summary(model)
prediction = predict(model, trSet)
table(prediction, trSet$Rings)
#перевіряю модель на тестових даних
prediction = predict(model, testSet)
table(prediction, testSet$Rings)
print(sprintf("Accuracy on the test set=%3.3f", sum(prediction==testSet$Rings)/length(prediction)))

#--------налaштовую bootstrap для моделі дерев, для цього задаю параметр trials
model = C5.0(Rings ~., data=trSet, trials=10)
summary(model)
prediction = predict(model, trSet)
table(prediction, trSet$Rings)
#перевіряю модель на тестових даних
prediction = predict(model, testSet)
table(prediction, testSet$Rings)
print(sprintf("Accuracy on the test set=%3.3f", sum(prediction==testSet$Rings)/length(prediction)))

#---------- randomForest --------
model = randomForest(Rings ~ ., data=trSet)
prediction = predict(model, trSet)
table(prediction, trSet$Rings)
prediction = predict(model, testSet)
table(prediction, testSet$Rings)
print(sprintf("Accuracy on the test set=%3.3f", sum(prediction==testSet$Rings, na.rm=TRUE)/length(prediction)))
###---------------------
#-------РОБОТА З ФАКТОРНОЮ ЗМІННОЮ----
data<-read.csv("/Users/Livia/Desktop/amn-lab/lab2/abalone.data", header=TRUE) #експортуэмо дані з csv-файла
names(data)<-c("Sex","Length","Diameter","Height","Whole_weight","Shucked_weight","Viscera_weight","Shell_weight","Rings")
data
#розбиваю відгук
rings_groups<-cut(data$Rings,breaks = c(0,7,12,Inf),
                  labels = c(0,1,2))
data$Rings <- rings_groups
#задаю,що змінна Sex - фактор
zf<-factor(data$Sex)
#хочу отримати коди,якими представлені класи фактора
unclass(zf)
#присвоюю ці числові значення змінній Sex
data$Sex <- unclass(zf)
#нормую атрибут Sex
library(scales)
norm <- c(scale(data$Sex, center=TRUE, scale = FALSE))
data$Sex <- norm 
#____розбиваю вибірку на тестову та тренувальну______
tr.index = sample(1:nrow(data), nrow(data)*0.8)
trSet = data[tr.index, ]  #тренувальна вибірка
testSet = data[-tr.index, ]  #тестова вибірка
trX = data.matrix(trSet[,1:7])  #тренувальна вибірка тільки з регресорів
trY = trSet$Rings       #відгук по тренувальній вибірці
testX = data.matrix(testSet[,1:7]) #тестова вибірка тільки з регресорів
testY = testSet$Rings    #відгук по тестовій вибірці
#перевіряю, яке віднош-ня між трьома класами відгуку на тренувальній і тестовій вибірці
#тобто скільки відсотків даних потрапило в кожен клас при розбитті
table(trSet$Rings)/sum(table(trSet$Rings))
table(testSet$Rings)/sum(table(testSet$Rings))
#результати вийшли майже однаковими,тому все добре. дані розбились правильно

#----------логістична регресія(мультиноміальна)-----------
#по тренувальній вибірці
#Встановлюю базовий клас
trSet$Rings <- relevel(trSet$Rings, ref = "0")
require(nnet) #підключаю пакет nnet
#будую модель на тренувальних даних
multinom.fit <- multinom(Rings ~ ., data = trSet)
#Перевіряю роботу моделі
summary(multinom.fit)
#Прогнозування значень для тренувальної вибірки
trSet$precticed <- predict(multinom.fit, newdata = trSet, "class")
#Будую таблицю класифікації
ctable <- table(trY, trSet$precticed)
ctable
#Обчислюю точність моделі на тренувальних даних
round((sum(diag(ctable))/sum(ctable))*100,2) #Accuracy : 76%
###########спробувала вилучити деякі регресори
trSet$Rings <- relevel(trSet$Rings, ref = "0")
require(nnet) #підключаю пакет nnet
#будую модель на тренувальних даних
multinom1.fit1 <- multinom(Rings ~ Length+Height+Whole_weight, data = trSet)
#Перевіряю роботу моделі
summary(multinom1.fit1)
#Прогнозування значень для тренувальної вибірки
trSet$precticed <- predict(multinom1.fit1, newdata = trSet, "class")
#Будую таблицю класифікації
c2table <- table(trY, trSet$precticed)
c2table
#Обчислюю точність моделі на тренувальних даних
round((sum(diag(c2table))/sum(c2table))*100,2) 
#перевіряю модель на тестових даних
testSet$precticed <- predict(multinom.fit, newdata = testSet, "class")
#Будую таблицю класифікації
c1table <- table(testY, testSet$precticed)
c1table
#Обчислюю точність моделі на тестових даних
round((sum(diag(c1table))/sum(c1table))*100,2)
#--------------класифікація методом k-nn---------------
result = knn(trX, testX, cl=trY)
#будую таблицю класифікації
conf.matrix = table(predicted=result, true=testY)
conf.matrix
#Обчислюю точність моделі для k=1
sum(diag(conf.matrix)/sum(conf.matrix))
#вибираю k=64
result1 = knn(trX, testX, cl=trY, k=64)
#будую таблицю класифікації
conf.matrix1 = table(predicted=result1, true=testY)
conf.matrix1
#Обчислюю точність моделі для k=64
sum(diag(conf.matrix1)/sum(conf.matrix1))
#вибираю k=15
result2 = knn(trX, testX, cl=trY, k=15)
#будую таблицю класифікації
conf.matrix2 = table(predicted=result2, true=testY)
conf.matrix2
#Обчислюю точність моделі для k=15
sum(diag(conf.matrix2)/sum(conf.matrix2))
#________НАЇВНИЙ БАЙЄСІВСЬКИЙ КЛАСИФІКАТОР_________
model = naiveBayes(trX, as.factor(trY))
hatYtr  = predict(model, trX) #роблю прогноз
hatYtest= predict(model, testX)
#будую талицю класиф. для тренувальної вибірки
table(hatYtr, trY)
#будую талицю класиф. для тестової вибірки
table(hatYtest, testY)
#обчислюю точність моделей на двох вибірках
print(sprintf("Training accuracy: %3.3f", sum(hatYtr==trY)/length(trY)))
print(sprintf("Test accuracy: %3.3f", sum(hatYtest==testY)/length(testY)))
dataY <- data$Rings
dataY
#___________МЕТОД svm________
# ----- Робота з усією вибіркою--------
model = svm(dataY ~ dataX, cost=0.75, kernel="radial"); #створюю модель використовуючи svm
prediction =model$fitted #прогноз
#перевіряю точність моделі на двох базових заданих параметрах kernel i cost
print(sprintf("Accuracy on the test set: %.3f", sum(as.numeric(prediction == dataY))/length(dataY)))
#стараюся підібрати параметри краще,за допомогою ф-ї tune()
tunedModel = tune(svm, dataY ~ dataX, ranges = list(cost = c(1, 5, 10,50), kernel=c("linear", "radial")));
tunedModel
tunedModel2 = tune(svm, dataY ~ dataX, ranges = list(cost = c(5, 10, 50), gamma=c(0.1, 1, 5)), kernel="radial");
tunedModel2
tunedModel3 = tune(svm, dataY ~ dataX, ranges = list(cost = c(5, 10, 50), degree=c(2, 3, 4, 5)), kernel="polynomial");
tunedModel3
# ------- Робота з тренувальною вибіркою
d = trX
tuned = tune(svm, trY ~ trX, ranges = list(cost = c(1,5, 10, 50,100), degree=c(2, 3, 4, 5)), kernel="polynomial");
tuned
model = svm(trY ~ d, cost=100, degree=3, kernel="polynomial")
#--------Перевіряю роботу моделі на тестовій вибірці
d = testX
hatYtest = predict(model, d)
print(sprintf("Accuracy on the test set: %3.3f", sum(as.numeric(as.numeric(hatYtest  == testY)))/length(testY)))
#--------- sampling="fix"
data = trX
model = tune(svm, trY ~ data, ranges = list(cost = c(5, 10, 50), degree=c(2, 3, 4, 5)), kernel="polynomial", tunecontrol=tune.control(sampling="fix"))
summary(model)
data = testX
hatY = predict(model$best.model, data)
print(sprintf("Accuracy on the test set: %.3f", sum(as.numeric(hatY == testY))/length(testY)))
#--------- sampling="fix" with given validation set size
control = tune.control(sampling="fix", fix=0.1)
data = trX
model = tune(svm, trY ~ data, ranges = list(cost = c(5, 10, 50), degree=c(2, 3, 4, 5)), kernel="polynomial", tunecontrol=control)
summary(model)
data = testX
hatY = predict(model$best.model, data)
print(sprintf("Accuracy on the test set: %.3f", sum(as.numeric(hatY == testY))/length(testY)))
#---------- sampling="bootstrap" -----------
control = tune.control(sampling="bootstrap", boot.size=0.9)
data = trX
model = tune(svm, trY ~ data, ranges = list(cost = c(5, 10,50), degree=c(2, 3, 4, 5)), kernel="polynomial", tunecontrol=control)
summary(model)
data = testX
hatY = predict(model$best.model, data)
print(sprintf("Accuracy on the test set: %.3f", sum(as.numeric(hatY == testY))/length(testY)))
#---------- sampling="cross" --------------
control = tune.control(sampling="cross", cross=10)
data = trX
model = tune(svm, trY ~ data, ranges = list(cost = c(50, 60), degree=c(2, 3, 4, 5)), kernel="polynomial", tunecontrol=control)
summary(model)
data = testX
hatY = predict(model$best.model, data)
print(sprintf("Accuracy on the test set: %.3f", sum(as.numeric(hatY == testY))/length(testY)))

plot(model) #будую графік моделі
# --------- C50 -----------дерева----
#будую модель дерев
model = C5.0(Rings ~., data=trSet)
summary(model)
prediction = predict(model, trSet)
table(prediction, trSet$Rings)
#перевіряю модель на тестових даних
prediction = predict(model, testSet)
table(prediction, testSet$Rings)
print(sprintf("Accuracy on the test set=%3.3f", sum(prediction==testSet$Rings)/length(prediction)))

#--------налaштовую bootstrap для моделі дерев, для цього задаю параметр trials
model = C5.0(Rings ~., data=trSet, trials=10)
summary(model)
prediction = predict(model, trSet)
table(prediction, trSet$Rings)
#перевіряю модель на тестових даних
prediction = predict(model, testSet)
table(prediction, testSet$Rings)
print(sprintf("Accuracy on the test set=%3.3f", sum(prediction==testSet$Rings)/length(prediction)))

#---------- randomForest --------
model = randomForest(Rings ~ ., data=trSet)
prediction = predict(model, trSet)
table(prediction, trSet$Rings)
prediction = predict(model, testSet)
table(prediction, testSet$Rings)
print(sprintf("Accuracy on the test set=%3.3f", sum(prediction==testSet$Rings, na.rm=TRUE)/length(prediction)))
