
clear all
close all
clc

init

numBins = 256;

% Build codebook
%[data_train, data_test, stop_kMeans, images_training, images_testing] = getData('Caltech',numBins); % K-means codebook
[data_train, data_test] = getData_rf(); % RF codebook

% Set the random forest parameters
param.num = 40;
param.depth = 8;
param.splitNum = 20;
param.split = 'IG'; % Objective function 'iformation gain' Degree of randomness parameter
mode = 'axis';        
for iter = 1:10
    % Train Random Forest
    tic; % Start timer
    tree = growTrees(data_train,param,mode);
    stop_train(iter) = toc; % Stop the timer

    % Evaluate/Test Random Forest
    tic; % Start timer
    for n=1:size(data_test,1) % Iterate through all rows of test data
        leaves = testTrees([data_test(n,:) 0],tree,mode); % Call the testTrees function
        % average the class distributions of leaf nodes of all trees
        p_rf = tree(1).prob(leaves,:);
        p_rf_sum = sum(p_rf)/length(tree);
        [~,predicted_label(n)] = max(p_rf_sum);
    end
    stop_test(iter) = toc; % Stop the timer

    % Calculate accuracy of classifier
    actual_label = data_test(:,end);
    accuracy(iter) = sum(actual_label == predicted_label')/length(actual_label)*100;
end

% Calculate average and standard deviation accuracy and computation time
std_accuracy = std(accuracy')
avg_accuracy = mean(accuracy)
avg_stop_train = mean(stop_train)
avg_stop_test = mean(stop_test)
avg_time = avg_stop_train + avg_stop_test
