clear
clc

load ./data/train_set/words_train.mat
X = full(X)';
Y = full(Y)';
Y = [~Y; Y];

%%
hiddenLayerSize = 5000;
net = patternnet(hiddenLayerSize);

net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 10/100;

net.numLayers = 3;
net.layerConnect = [0, 0, 0; 0, 0, 1; 1, 0, 0];
% net.layerConnect =...
%     [0, 0, 0, 0, 0;...
%     0, 0, 0, 0, 1;...
%     1, 0, 0, 0, 0;...
%     0, 0, 1, 0, 0;...
%     0, 0, 0, 1, 0];
net.outputConnect = [0, 1, 0];
net.biasConnect = [1; 1; 1];

net.layers{3}.transferFcn = 'tansig';

net.layers{2}.dimensions = 2;
net.layers{3}.dimensions = 100;
% net.layers{4}.dimensions = 500;
% net.layers{5}.dimensions = 100;

net.trainParam.max_fail = 10000;

% net.trainParam.min_grad = -1;

%%
% performance_opt = inf;

% for i = 1: 50
    [net_t,tr] = train(net,X,Y,'useGPU','yes','showResources','yes');
% 
%     Yhat = net_t(X);    
%     errors = gsubtract(Y,Yhat);
%     performance = (tr.best_tperf + tr.best_vperf) / 2;
%     if performance < performance_opt
%         performance_opt = performance;
%         net_opt = net_t;
%     end
% end

% Yhat = net_opt(X);
% figure, plotconfusion(Y, Yhat);
view(net)