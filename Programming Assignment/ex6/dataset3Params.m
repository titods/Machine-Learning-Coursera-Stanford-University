function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
sigma_params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
results = zeros(size(C_params, 1) * size(sigma_params, 1), 3)
iter = 0
for i = 1:size(C_params, 1)
    for j = 1:size(sigma_params, 1)
        iter = iter + 1;
        model = svmTrain(X, y, C_params(i), @(x1, x2) gaussianKernel(x1, x2, sigma_params(j)));
        pred = svmPredict(model, Xval);
        error_pred_eval = mean(double(pred ~= yval));
        results(iter, :) = [C_params(i), sigma_params(j), error_pred_eval];
    end
end

results = sortrows(results, 3); % sort by ascending order in column 3
C = results(1,1);
sigma = results(1,2);

% =========================================================================

end
