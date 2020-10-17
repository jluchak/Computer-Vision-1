%% Testing Different random forest parameters
% i.e. Number of trees, maximum depth of trees, number of split parameters
clear all
close all
clc

init

numBins = 256;
%[data_train, data_test, stop_kMeans, images_training, images_testing] = getData('Caltech',numBins); % K-means codebook
[data_train, data_test] = getData_rf(); % RF codebook

original_data_train = data_train;

% Reorder data randomly
randidx = randperm(size(original_data_train,1));
original_data_train = original_data_train(randidx,:);

% Cross-validation: Create the 5 subsets
K = 5; % number of folds
subset_size = size(original_data_train,1)/K; % 150/5 = 30 images for each subset

% Set the random forest parameters
param_num = [2 5 10 20 30 40 50 60 70 80 90 100]; % The number of trees
param_depth = [2 3 4 5 6 7 8 9 10]; % Maximum depth of the trees
param_splitNum = [2 3 4 5 6 7 8 9 10 15 20 25 30]; % Number of set of split parameters theta i.e. p=3
param.split = 'IG'; % Objective function 'iformation gain' Degree of randomness parameter
mode = 'axis';

% 5-fold cross-validation: K-1 subsets for training and 1 for validation
for iter = 1:10
    for fold = 1:K
        % Validation subset (only one subset)
        start_idx = subset_size*(fold-1) + 1;
        stop_idx = start_idx + subset_size - 1;
        data_test = original_data_train(start_idx:stop_idx,:); % train and test in same data

        % Training subsets (the rest of the K-1 subsets)
        data_train = original_data_train;
        data_train(start_idx:stop_idx,:) = []; % exclude the subset used for testing

        %% Testing number of trees: param_num
        for i = 1:length(param_num)
            param.num = param_num(i);
            param.depth = 8;
            param.splitNum = 20;
            
            % Train Random Forest
            tic; % Start timer
            tree = growTrees(data_train,param,mode);
            stop = toc; % Stop the timer
            fprintf('TIC TOC Train random forest: %g\n', stop);

            % Evaluate/Test Random Forest
            tic; % Start timer
            for n=1:size(data_test,1) % Iterate through all rows of test data
                leaves = testTrees([data_test(n,:) 0],tree,mode); % Call the testTrees function
                % average the class distributions of leaf nodes of all trees
                p_rf = tree(1).prob(leaves,:);
                p_rf_sum = sum(p_rf)/length(tree);
                [~,predicted_label(n)] = max(p_rf_sum);
            end
            stop = toc; % Stop the timer
            fprintf('TIC TOC Test random forest: %g\n', stop);

            % Calculate accuracy of classifier
            actual_label = data_test(:,end);
            in_tree_accuracy(fold,i) = sum(actual_label == predicted_label')/length(actual_label)*100;
        end

        % Testing maximum depth of trees: param_depth
        for i = 1:length(param_depth)
            param.num = 40;
            param.depth = param_depth(i);
            param.splitNum = 20;
        % Train Random Forest
        tic; % Start timer
        tree = growTrees(data_train,param,mode);
        stop = toc; % Stop the timer
        fprintf('TIC TOC Train random forest: %g\n', stop);

        % Evaluate/Test Random Forest
        tic; % Start timer
        for n=1:size(data_test,1) % Iterate through all rows of test data
            leaves = testTrees([data_test(n,:) 0],tree,mode); % Call the testTrees function
            % average the class distributions of leaf nodes of all trees
            p_rf = tree(1).prob(leaves,:);
            p_rf_sum = sum(p_rf)/length(tree);
            [~,predicted_label(n)] = max(p_rf_sum);
        end
        stop = toc; % Stop the timer
        fprintf('TIC TOC Test random forest: %g\n', stop);

        % Calculate accuracy of classifier
        actual_label = data_test(:,end);
        in_depth_accuracy(fold,i) = sum(actual_label == predicted_label')/length(actual_label)*100;

        end

        % Testing number of split parameters: param_split
        for i = 1:length(param_splitNum)
            param.num = 40;
            param.depth = 8;
            param.splitNum = param_splitNum(i);
        % Train Random Forest
        tic; % Start timer
        tree = growTrees(data_train,param,mode);
        stop = toc; % Stop the timer
        fprintf('TIC TOC Train random forest: %g\n', stop);

        % Evaluate/Test Random Forest
        tic; % Start timer
        for n=1:size(data_test,1) % Iterate through all rows of test data
            leaves = testTrees([data_test(n,:) 0],tree,mode); % Call the testTrees function
            % average the class distributions of leaf nodes of all trees
            p_rf = tree(1).prob(leaves,:);
            p_rf_sum = sum(p_rf)/length(tree);
            [~,predicted_label(n)] = max(p_rf_sum);
        end
        stop = toc; % Stop the timer
        fprintf('TIC TOC Test random forest: %g\n', stop);

        % Calculate accuracy of classifier
        actual_label = data_test(:,end);
        in_split_accuracy(fold,i) = sum(actual_label == predicted_label')/length(actual_label)*100;

        end

    end
    
    % Calculate average and std accuracy across folds
    trees_accuracy(iter,:) = mean(in_tree_accuracy);
    trees_standard_dev(iter,:) = std(in_tree_accuracy);

    depth_accuracy(iter,:) = mean(in_depth_accuracy);
    depth_standard_dev(iter,:) = std(in_depth_accuracy);

    splits_accuracy(iter,:) = mean(in_split_accuracy);
    splits_standard_dev(iter,:) = std(in_split_accuracy);
end

% Calculate average and std accuracy across iterations
global_trees_accuracy = mean(trees_accuracy);
global_trees_standard_dev = mean(trees_standard_dev);

global_depth_accuracy = mean(depth_accuracy);
global_depth_standard_dev = mean(depth_standard_dev);

global_splits_accuracy = mean(splits_accuracy);
global_splits_standard_dev = mean(splits_standard_dev);

%%
% Plotting
figure, %Trial 1: 10,5,3 %Trial 2: 40,7,20 % Trial 3: 40,6,5 % Trial 4: 40,7,15 
% Trial 1B: 10,5,3 % Trial 2B: 40, 8, 20 % Trial 3B: 30, 6, 20 Trial 4B:
% 40,8,15 % Trial 5B: 40, 7, 9
subplot(3,1,1)
plot(param_num, global_trees_accuracy,'-x','linewidth',2);
hold on
errorbar(param_num,global_trees_accuracy,global_trees_standard_dev);
xlim([min(param_num) max(param_num)]);
ylim([min(global_trees_accuracy) max(global_trees_accuracy)]);
xlabel('Number of Trees');
ylabel('Accuracy [%]'); 

subplot(3,1,2)
plot(param_depth, global_depth_accuracy,'-x','linewidth',2);
hold on
errorbar(param_depth,global_depth_accuracy,global_depth_standard_dev);
xlim([min(param_depth) max(param_depth)]);
ylim([min(global_depth_accuracy) max(global_depth_accuracy)]);
xlabel('Depth of Tree');
ylabel('Accuracy [%]');

subplot(3,1,3)
plot(param_splitNum, global_splits_accuracy,'-x','linewidth',2);
hold on
errorbar(param_splitNum,global_splits_accuracy,global_splits_standard_dev);
xlabel('Number of Splits');
xlim([min(param_splitNum) max(param_splitNum)]);
ylim([min(global_splits_accuracy) max(global_splits_accuracy)]);
ylabel('Accuracy [%]');
