%knn
x_1 = full(X);
y_1 = full(Y);
indices = crossvalind('Kfold', y_1, 10);
x_train = x_1(indices ~= 1, :);
y_train = y_1(indices ~= 1, :);
x_test = x_1(indices == 1, :);
y_test = y_1(indices == 1, :);

model = fitcknn(x_train, y_train, 'NumNeighbors', 18);
y_pre = predict(model, x_test);
accuracy = mean(y_pre == y_test); %0.7133

%cv
knnaccu = cell(200, 1);
for i = 1: 50
    model = fitcknn(x_train, y_train, 'NumNeighbors', i);
    y_pre = predict(model, x_test);
    knnaccu{i} = mean(y_pre == y_test);
end