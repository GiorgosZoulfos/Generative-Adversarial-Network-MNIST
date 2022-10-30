clear; clc; close all;

%% Load the generative model and data

load('data21.mat')

%% 2.1 - Generate handwritten 8

bigPic = zeros(10*28, 10*28);
for  i = 1:100

    Z = randn(10,1);

    W1 = A_1*Z + B_1;
    Z1 = reLu(W1);
    W2 = A_2*Z1 + B_2;
    X = sigm(W2);
    
    bigPic((ceil(i/10)-1)*28+1:(ceil(i/10))*28, ... 
        (mod(i,10)+10*floor(2^-(mod(i,10)))-1)*28+1: ...
        (mod(i,10)+10*floor(2^-(mod(i,10))))*28) = reshape(X,28,28);
end

imshow(bigPic)

%% Functions used above

% Sigmoid function used at the output of Cross Entropy NN
% so that it is in [0, 1] interval
function out = sigm(inp)
    out = 1 ./ (1 + exp(inp));
end

% ReLu function 
function out = reLu(inp)
    inp(inp <= 0) = 0;
    out = inp;
end