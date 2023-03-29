clc
clear all 
close all
XTrain = xlsread('train.xlsx',1,'A1:C12018')';
YTrain = xlsread('train.xlsx',1,'D1:D12018')';
%testing
XTest = xlsread('train.xlsx',1,'A1:C12018')';
YTest = xlsread('train.xlsx',1,'D1:D12018')';
%define LSTM
inputSize = 3;
numResponses =1;
numHiddenUnits=100;
layers = [sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
opts = trainingOptions('adam','MaxEpochs',1000,'GradientThreshold',0.01 ...
    ,'InitialLearnRate',0.0001);
net = trainNetwork(XTrain,YTrain,layers,opts);
%predict
YPred1 = predict(net,XTest)

figure,
plot(YPred1,'r-*','LineWidth',1),hold on
plot(YTest,'g-*','LineWidth',1); hold all;
xlabel('sample')
ylabel('values')
grid on
legend('Predicted','original');
title('performance')