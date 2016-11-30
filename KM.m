%kmeans
x_1 = full(X);
y_1 = full(Y);
indices = crossvalind('Kfold', y_1, 10);
x_train = x_1(indices ~= 1, :);
y_train = y_1(indices ~= 1, :);
x_test = x_1(indices == 1, :);
y_test = y_1(indices == 1, :);

[idx, centr] = kmeans(x_train, 88);  % 0.6956, 88 
y_est = zeros(450, 1);

for j = 1:450
    index = -1;
    dis = 1000000;
    for k = 1:150
        center = centr(k, :);
        point = x_test(j, :);
        distance = norm(center-point);
        if(distance < dis)
            dis = distance;
            index = k;
        end
    end
    y_est(j, 1) = mode(y_train(idx==index));
end
accuracy = mean(y_est == y_test);

%cv
indices = crossvalind('Kfold', y_1, 10);
y_est = zeros(450, 1);
kmaccu = cell(150, 1);
for i = 1: 86
    display(i)
    cluster_ = i;
    [idx, centr] = kmeans(x_train, cluster_);
    for j = 1:450
        index = -1;
        dis = 1000000;
        for k = 1:cluster_
            center = centr(k, :);
            point = x_test(j, :);
            distance = norm(center-point);
            if(distance < dis)
                dis = distance;
                index = k;
            end
        end
        y_est(j, 1) = mode(y_train(idx==index));
    end
    kmaccu{i} = mean(y_est == y_test);
    display(kmaccu{i});
end
