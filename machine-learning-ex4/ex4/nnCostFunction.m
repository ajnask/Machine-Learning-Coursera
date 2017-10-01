function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

I = eye(num_labels); % I is 10x10
Y = zeros(m, num_labels); % Y is 5000x10 because m is 5000
for i=1:m
  Y(i, :)= I(y(i), :);
end
a1 = [ones(m,1) X]; %X is 5000x400, therefore a1 is 5000x401
z2 = a1 *Theta1'; %z2 will be (5000x401) * (25*401)' = 5000x25
a2 = [ones(size(z2),1) sigmoid(z2)]; % a2 will be 5000x26
z3 = a2*Theta2'; % z3 will be (5000x26) * (10x26)' = 5000x10
a3 = sigmoid(z3); %a3 is 5000x10

%% Calculating J using for loop
%for i=1:m
%  for k=1:num_labels
%  J = J + (Y(i,k)*log(a3(i,k))+(1-Y(i,k))*log(1-a3(i,k)));
%  end
%end
%J=-J/m;

J = (-1/m)*sum(sum((Y).*log(a3) + (1-Y).*log(1-a3), 2));

reg = (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2, 1)) + sum(sum(Theta2(:,2:end).^2, 1)));

J = J + reg;

sigma3 = a3 - Y; %sigma3 is 5000x10
%sigma2 = (sigma3*Theta2 .* sigmoidGradient([ones(size(z2, 1), 1) z2]))(:, 2:end);
sigma2 = ((sigma3*Theta2).*a2.*(1-a2))(:,2:end); 
%sigma2 is 5000x26 because (sigma3*Theta2) is 5000x10 * 10x26 = 5000x26

delta1 = sigma2'*a1; %delta1 will be (5000x26)' * 5000x401 = 26x401
delta2 = sigma3'*a2; %delta2 will be (5000x10)' * 5000x26 = 10x26

%Theta1_grad = delta1/m;
%Theta2_grad = delta2/m; 

Theta1_grad = (delta1/m) + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
Theta2_grad = (delta2/m) + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];
  





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
