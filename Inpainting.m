clear; close all; clc

%% Load the generative model and data

load('data21.mat')
load('data22.mat')

%% Problem 2.2

% Initialize variables

m = 784;
N = 300;
iter = 2000;
learningRate = 10^-4;

images = zeros(m,4);
restoredImages = zeros(m, 4);
error = zeros(iter,4);

% Define transform T
T = [eye(N) zeros(N,m-N)];

% Perform Gradient Descent for each X_n
for i = 1:4

    images(1:N,i) = T*X_n(:,i);
    % Input of Neural Network
    Z = randn(10,1);

    for j = 1:iter

        % Outpout of Neural Network
        W1 = A_1*Z + B_1;
        Z1 = reLu(W1);
        W2 = A_2*Z1 + B_2;
        X = sigm(W2);

        % Compute error for this iteration
        error(j,i) = N*(log(norm(T*X_n(:,i) - T*X)^2)) + norm(Z)^2;

        % Update Z
        U2 = -(2*T'*(T*X_n(:,i)-T*X))/norm(T*X_n(:,i)-T*X)^2;
        V2 = U2.*derSigm(W2);

        U1 = A_2'*V2;
        V1 = U1.*reLuDer(W1);

        U0 = A_1'*V1;

        Z = Z - learningRate*(N*U0 + 2*Z);

    end

    W1 = A_1*Z + B_1;
    Z1 = reLu(W1);
    W2 = A_2*Z1 + B_2;
    X = sigm(W2);

    restoredImages(:,i) = X;

    figure
    subplot(1,3,1), imshow(reshape(X_i(:,i),28,28)), title('Ideal Image '+string(i))
    subplot(1,3,2), imshow(reshape(images(:,i),28,28)),  ... 
        title('Noisy Image '+string(i)+ ', N = '+string(N))
    subplot(1,3,3), imshow(reshape(restoredImages(:,i),28,28)),  ... 
        title('Restored Image '+string(i))

end


figure
plot(error(:,1))
hold
plot(error(:,2))
plot(error(:,3))
plot(error(:,4))
title('Error, m = '+string(learningRate))
legend('Image 1', 'Image 2', 'Image 3', 'Image 4')

%% Functions used above

% Sigmoid function used at the output of Cross Entropy NN
% so that it is in [0, 1] interval
function out = sigm(inp)
    out = 1 ./ (1 + exp(inp));
end

% Derivative of simgoid function
function out = derSigm(inp)
    out = - exp(inp)./(exp(inp) + 1).^2;
end

% ReLu function 
function out = reLu(inp)
    inp(inp <= 0) = 0;
    out = inp;
end

% Derivative of relu function
function out = reLuDer(inp)
    inp(inp > 0) = 1;
    inp(inp <= 0) = 0;
    out = inp;
end