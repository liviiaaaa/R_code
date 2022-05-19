library(mlbench)
library(car)
library(glmnet)
require(glmnet)

tbl<-read.table(file="/Users/Livia/Desktop/amn-lab/servo.csv",
                header=T)  #експортуємо дані з файлу csv
tbl
sum(is.na(tbl))  #пропущених даних немає
#спочатку відкину факторні зміннні(із факторним змінними буду знову ділити на трейн і тест)
data<-tbl[,3:ncol(tbl)] 
data
#____розбиваю вибірку на тестову та тренувальну______

tr.index = sample(1:nrow(data), nrow(data)*0.8)
trSet = data[tr.index, ]  #тренувальна вибірка
testSet = data[-tr.index, ]  #тестова вибірка
trX = data.matrix(trSet[,1:2])  #тренувальна вибірка тільки з регресорів
trY = trSet$Class       #відгук по тренувальній вибірці
testX = data.matrix(testSet[,1:2]) #тестова вибірка тільки з регресорів
testY = testSet$Class   #відгук по тестовій вибірці

#______регресійний аналіз__________
scatterplotMatrix(data, diagonal="histogram",smoother=F)#діаграма розсіювання
model<-lm(Class~Pgain+Vgain, data=trSet)#початкова модель
summary(model)

plot(model$fitted.values,trSet$Class,xlab="Class forecast", ylab="true Class") #діаграма прогноз-відгук
abline(c(0,1),col="red")
plot(model$fitted.values,model$residuals,xlab="prediction", ylab="residuals") #діаграма прогноз-залишки
abline(0,0,col="red")
qqnorm(model$residuals)
qqline(model$residuals,col="red")

#покращення оцінки нелінійністю
model1<-lm(Class~(Pgain+Vgain)^2, data=trSet)#ввели нелінійність
summary(model1)
plot(model1$fitted.values,trSet$Class,xlab="Class forecast", ylab="true Class") #діаграма прогноз-відгук
abline(c(0,1),col="red")
plot(model1$fitted.values,model1$residuals,xlab="prediction", ylab="residuals") #діаграма прогноз-залишки
abline(0,0,col="red")
qqnorm(model1$residuals)
qqline(model1$residuals,col="red")

#рідж-регресія
y <- trSet$Class #визначаємо відгук
x <- data.matrix(trSet[, c('Pgain', 'Vgain')]) #матриця регресорів
model_la <- glmnet(x, y, alpha = 0)
summary(model_la)
cv_model <- cv.glmnet(x, y, alpha = 0) #підбір парам.lambda методом крос-валідації
summary(cv_model)
best_lambda <- cv_model$lambda.min
best_lambda
plot(cv_model) 
best_model <- glmnet(x, y, alpha = 0, lambda = best_lambda)
coef(best_model)
plot(model_la, xvar = "lambda")
y_predicted <- predict(model_la, s = best_lambda, newx = x)#прогнозовані дані
sst <- sum((y - mean(y))^2)
sse <- sum((y_predicted - y)^2)#якість прогнозу
sse
rsq <- 1 - sse/sst   #r^2 
rsq

#test
#перевірка якості прогнозу кожної з моделей на тест.вибірці
test_hat <- predict(model,testSet)
test_hat
sum((testSet-test_hat)^2)   #28901.78
test_hat2 <- predict(model1,testSet)
test_hat2
sum((testSet-test_hat2)^2)  ##28531.89; 
y_test <- testSet$Class #визначаємо відгук
x_test <- data.matrix(testSet[, c('Pgain', 'Vgain')])
test_hat3 <- predict(model_la, s = best_lambda, newx = x_test)
sum((testSet-test_hat3)^2)  ##28561.97


#введу факторні змінні

data<-read.table(file="/Users/Livia/Desktop/amn-lab/servo.csv",
                header=T)  #експортуємо дані з файлу csv
data
sum(is.na(tbl))  #пропущених даних немає
#задаю,що змінна Motor - фактор
zf<-factor(data$Motor)
#хочу отримати коди,якими представлені класи фактора
unclass(zf)
#присвоюю ці числові значення змінній Motor
data$Motor <- unclass(zf)
#нормую атрибут Motor
library(scales)
norm <- c(scale(data$Motor, center=TRUE, scale = FALSE))
data$Motor <- norm 
#виконую те ж саме над змінною Screw
xf<-factor(data$Screw)
#хочу отримати коди,якими представлені класи фактора
unclass(xf)
#присвоюю ці числові значення змінній Motor
data$Screw <- unclass(xf)
#нормую атрибут Motor
library(scales)
norm <- c(scale(data$Screw, center=TRUE, scale = FALSE))
data$Screw <- norm

#____розбиваю вибірку на тестову та тренувальну______

tr.index = sample(1:nrow(data), nrow(data)*0.8)
trSet = data[tr.index, ]  #тренувальна вибірка
testSet = data[-tr.index, ]  #тестова вибірка
trX = data.matrix(trSet[,1:4])  #тренувальна вибірка тільки з регресорів
trY = trSet$Class       #відгук по тренувальній вибірці
testX = data.matrix(testSet[,1:4]) #тестова вибірка тільки з регресорів
testY = testSet$Class   #відгук по тестовій вибірці

#______регресійний аналіз__________
scatterplotMatrix(data, diagonal="histogram",smoother=F)#діаграма розсіювання
model<-lm(Class~Motor+Screw+Pgain+Vgain, data=trSet)#початкова модель
summary(model)

plot(model$fitted.values,trSet$Class,xlab="Class forecast", ylab="true Class") #діаграма прогноз-відгук
abline(c(0,1),col="red")
plot(model$fitted.values,model$residuals,xlab="prediction", ylab="residuals") #діаграма прогноз-залишки
abline(0,0,col="red")
qqnorm(model$residuals)
qqline(model$residuals,col="red")

#покращення оцінки нелінійністю
model1<-lm(Class~(Motor+Screw+Pgain+Vgain)^2, data=trSet)#ввели нелінійність
summary(model1)
plot(model1$fitted.values,trSet$Class,xlab="Class forecast", ylab="true Class") #діаграма прогноз-відгук
abline(c(0,1),col="red")
plot(model1$fitted.values,model1$residuals,xlab="prediction", ylab="residuals") #діаграма прогноз-залишки
abline(0,0,col="red")
qqnorm(model1$residuals)
qqline(model1$residuals,col="red")

#рідж-регресія
y <- trSet$Class #визначаємо відгук
x <- data.matrix(trSet[, c('Motor','Screw','Pgain', 'Vgain')]) #матриця регресорів
model_la <- glmnet(x, y, alpha = 0)
summary(model_la)
cv_model <- cv.glmnet(x, y, alpha = 0) #підбір парам.lambda методом крос-валідації
summary(cv_model)
best_lambda <- cv_model$lambda.min
best_lambda
plot(cv_model) 
best_model <- glmnet(x, y, alpha = 0, lambda = best_lambda)
coef(best_model)
plot(model_la, xvar = "lambda")
y_predicted <- predict(model_la, s = best_lambda, newx = x)#прогнозовані дані
sst <- sum((y - mean(y))^2)
sse <- sum((y_predicted - y)^2)#якість прогнозу
sse
rsq <- 1 - sse/sst   #r^2 
rsq

#test
#перевірка якості прогнозу кожної з моделей на тест.вибірці
test_hat <- predict(model,testSet)
test_hat
sum((testSet-test_hat)^2)   #77824.74
test_hat2 <- predict(model1,testSet)
test_hat2
sum((testSet-test_hat2)^2)  ##67864.43; 
y_test <- testSet$Class #визначаємо відгук
x_test <- data.matrix(testSet[, c('Motor','Screw','Pgain', 'Vgain')])
test_hat3 <- predict(model_la, s = best_lambda, newx = x_test)
sum((testSet-test_hat3)^2)  ##72579.05


