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

min_error = 1000;
best_C = C;
best_sigma = sigma;
start_C = 0;
start_sigma = 0;
step = 3;

while true
    error_updated = 0;
    for C = logspace(start_C-step, start_C+step, 7)
        for sigma = logspace(start_sigma-step, start_sigma+step, 7)
            model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
            pred = svmPredict(model, Xval);
            error = mean(double(pred ~= yval));
            if error < min_error
                min_error = error;
                best_C = C;
                best_sigma = sigma;
                error_updated = 1;
            end
        end
    end
    if error_updated == 0
        break
    end
    step = step/3;
    start_C = log10(best_C);
    start_sigma = log10(best_sigma);
end

C = best_C;
sigma = best_sigma;

% =========================================================================

end
