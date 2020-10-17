% Simple Random Forest Toolbox for Matlab
% written by Mang Shao and Tae-Kyun Kim, June 20, 2014.
% updated by Tae-Kyun Kim, Feb 09, 2017

% This is a guideline script of simple-RF toolbox.
% The codes are made for educational purposes only.
% Some parts are inspired by Karpathy's RF Toolbox

% Under BSD Licence
clear all
close all
clc

question = 1;

if question == 1
    
% Initialisation
init;

mode = 'axis';
test_train = 0; % Are we using the test data by dividing the train data?

% Select dataset
[data_train, data_test] = getData('Toy_Spiral'); % {'Toy_Gaussian', 'Toy_Spiral', 'Toy_Circle', 'Caltech'}

if test_train == 1
    data_test = [];
    
    % Label 1 (color red)
    idx = find(data_train(:,3) == 1);
    idx_test = idx(1:2:length(idx));
    data_test = data_train(idx_test,:);
    data_train(idx_test,:) = [];

    % Label 2 (color green)
    idx = find(data_train(:,3) == 2);
    idx_test = idx(1:2:length(idx));
    data_test = [data_test; data_train(idx_test,:)];
    data_train(idx_test,:) = [];
    
    % Label 3 (color blue)
    idx = find(data_train(:,3) == 3);
    idx_test = idx(1:2:length(idx));
    data_test = [data_test; data_train(idx_test,:)];
    data_train(idx_test,:) = [];
end

%%%%%%%%%%%%%
% check the training and testing data
    % data_train(:,1:2) : [num_data x dim] Training 2D vectors
    % data_train(:,3) : [num_data x 1] Labels of training data, {1,2,3}
   
plot_toydata(data_train);

    % data_test(:,1:2) : [num_data x dim] Testing 2D vectors, 2D points in the
    % uniform dense grid within the range of [-1.5, 1.5]
    % data_test(:,3) : N/A
    
scatter(data_test(:,1),data_test(:,2),'.b');

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %% QUESTION 1: Training Decision Forest %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
% Set the random forest parameters for instance, 
param.num = 10;         % Number of trees
param.depth = 5;        % Maximum depth of trees
param.splitNum = 3;     % Number of split functions to try (p)
param.split = 'IG';     % Currently support 'information gain' only

%%%%%%%%%%%%%%%%%%%%%%
% Train Random Forest

% Grow all trees
for it = 1:10
    [trees,global_ig] = growTrees(data_train,param,mode);
    final_ig(it) = nanmean(nanmean(global_ig));
end
mean(final_ig)
% Visualise class distributions of the first 9 leaf nodes
figure
idx_tree = 1;
idx_node = 1;
for i = 1:9
    try
       prob_leaf = trees(idx_tree).leaf(idx_node).prob; 
    catch
       idx_tree = idx_tree + 1;
       idx_node = 1;
       prob_leaf = trees(idx_tree).leaf(idx_node).prob;
    end
    subplot(3,3,i)
    bar(prob_leaf);
    title(['Leaf node ', num2str(i)])
    idx_node = idx_node + 1;
end

% Same way to plot it
% figure
% visualise_leaf

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% QUESTION 2: Evaluating Decision Forest on the test data %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Evaluate/Test Random Forest

% grab the few data points and evaluate them one by one by the leant RF
if test_train == 1
    test_point = data_test(:,1:2);
else
    test_point = [-.5 -.7; .4 .3; -.7 .4; .5 -.5];
end

all_probs = [];
for n = 1:size(test_point,1)
    leaves = testTrees([test_point(n,:) 0],trees,mode);
    % average the class distributions of leaf nodes of all trees
    p_rf = trees(1).prob(leaves,:);    
    
    if length(trees) == 1
        p_rf_sum = p_rf;
    else
        p_rf_sum = sum(p_rf)/length(trees);
    end
    
    % Plot class distributions of the leaf nodes
%     figure
%     for idx_tree = 1:length(trees)
%         subplot(3,4,idx_tree)
%         bar(p_rf(idx_tree,:));
%         title(['Leaf node ', num2str(idx_tree)])
%     end
%     subplot(3,4,11)
%     bar(p_rf_sum, 'r')
%     title('Average')
  
    all_probs(n,:) = p_rf_sum;
end

% Plot test and training data together
figure
plot_toydata(data_train);
hold on
for n = 1:size(test_point,1)
    plot(test_point(n,1),test_point(n,2),'Color', all_probs(n,:), 'MarkerSize', 10,...
        'Marker', 's', 'MarkerFaceColor', all_probs(n,:), 'MarkerEdgeColor', 'k');
end

%% Accuracy
if test_train == 1
    true_labels = [];
    predicted_labels = [];
    for n = 1:size(test_point,1)
        true_labels(n) = data_test(n,3);
        [~, predicted_labels(n)] = max(all_probs(n,:));
    end
end
accuracy = sum(true_labels == predicted_labels)/size(test_point,1)*100;

%% Test on the dense 2D grid data, and visualise the results

if test_train == 0
    for n = 1:size(data_test,1)
        leaves = testTrees(data_test(n,:),trees,mode);

        % average the class distributions of leaf nodes of all trees
        p_rf = trees(1).prob(leaves,:);    
        p_rf_sum = sum(p_rf)/length(trees);

        all_probs_test(n,:) = p_rf_sum;
    end
    visualise(data_train,all_probs_test,0,0)
end

%% Change the RF parameter values and evaluate

% Set the random forest parameters
num_trees = [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]; % Number of trees
depth_trees = [2, 5, 7, 11]; % Maximum depth of trees
splitNum_trees = [2, 5, 7, 11, 15]; % Number of split functions to try (p)

% Change number trees
all_probs = [];
accuracy = [];
global_accuracy_num = [];
for iter = 1:10 % for each iteration
    for i = 1:length(num_trees) % for each number of trees
        param.num = num_trees(i);
        param.depth = 5; % Default
        param.splitNum = 3; % Default

        % Train Random Forest
        trees = growTrees(data_train,param);
        
        % Test Random Forest
        if test_train == 0
            all_probs_test = [];
            % 2D dense grid
            for n = 1:size(data_test,1)
                leaves = testTrees(data_test(n,:),trees);

                % average the class distributions of leaf nodes of all trees
                p_rf = trees(1).prob(leaves,:);    
                p_rf_sum = sum(p_rf)/length(trees);

                all_probs_test(n,:) = p_rf_sum;
            end
            visualise(data_train,all_probs_test,0,0)
        else
            for n = 1:size(test_point,1)
                leaves = testTrees([test_point(n,:) 0],trees);
                % average the class distributions of leaf nodes of all trees
                p_rf = trees(1).prob(leaves,:);
                if length(trees) == 1
                    p_rf_sum = p_rf;
                else
                    p_rf_sum = sum(p_rf)/length(trees);
                end
                all_probs(n,:) = p_rf_sum;
            end
            
            % Accuracy
            true_labels = [];
            predicted_labels = [];
            for j = 1:size(test_point,1)
                true_labels(j) = data_test(j,3);
                [~, predicted_labels(j)] = max(all_probs(j,:));
            end
            accuracy(i) = sum(true_labels == predicted_labels)/size(test_point,1)*100;
        end
    end
    global_accuracy_num(iter,:) = accuracy;
end

figure
subplot(3,1,1)
plot(num_trees,mean(global_accuracy_num),'-x','linewidth',2)
xlabel('Number of trees')
xlim([min(num_trees) max(num_trees)])
ylim([min(mean(global_accuracy_num)) 100])
ylabel('Accuracy [%]')

% Change depth trees
all_probs = [];
accuracy = [];
global_accuracy_depth = [];
for iter = 1:10 % for each iteration
    for i = 1:length(depth_trees) % for each depth of trees
        param.depth = depth_trees(i);
        param.num = 10; % Default
        param.splitNum = 3; % Default

        % Train Random Forest
        trees = growTrees(data_train,param);
        
        % Test Random Forest
        if test_train == 0
            all_probs_test = [];
            % 2D dense grid
            for n = 1:size(data_test,1)
                leaves = testTrees(data_test(n,:),trees);

                % average the class distributions of leaf nodes of all trees
                p_rf = trees(1).prob(leaves,:);    
                p_rf_sum = sum(p_rf)/length(trees);

                all_probs_test(n,:) = p_rf_sum;
            end
            visualise(data_train,all_probs_test,0,0)
        else
            for n = 1:size(test_point,1)
                leaves = testTrees([test_point(n,:) 0],trees);
                % average the class distributions of leaf nodes of all trees
                p_rf = trees(1).prob(leaves,:);
                if length(trees) == 1
                    p_rf_sum = p_rf;
                else
                    p_rf_sum = sum(p_rf)/length(trees);
                end
                all_probs(n,:) = p_rf_sum;
            end
            
            % Accuracy
            true_labels = [];
            predicted_labels = [];
            for j = 1:size(test_point,1)
                true_labels(j) = data_test(j,3);
                [~, predicted_labels(j)] = max(all_probs(j,:));
            end
            accuracy(i) = sum(true_labels == predicted_labels)/size(test_point,1)*100;
        end
    end
    global_accuracy_depth(iter,:) = accuracy;
end

subplot(3,1,2)
plot(depth_trees,mean(global_accuracy_depth),'-x','linewidth',2)
xlabel('Depth of trees')
xlim([min(depth_trees) max(depth_trees)])
ylim([min(mean(global_accuracy_depth)) 100])
ylabel('Accuracy [%]')

% Change number of split functions
all_probs = [];
accuracy = [];
global_accuracy_split = [];
for iter = 1:10 % for each iteration
    for i = 1:length(splitNum_trees) % for each number of split functions
        param.splitNum = splitNum_trees(i);
        param.num = 10; % Default
        param.depth = 5; % Default

        % Train Random Forest
        trees = growTrees(data_train,param);
        
        % Test Random Forest
        if test_train == 0
            all_probs_test = [];
            % 2D dense grid
            for n = 1:size(data_test,1)
                leaves = testTrees(data_test(n,:),trees);

                % average the class distributions of leaf nodes of all trees
                p_rf = trees(1).prob(leaves,:);    
                p_rf_sum = sum(p_rf)/length(trees);

                all_probs_test(n,:) = p_rf_sum;
            end
            visualise(data_train,all_probs_test,0,0)
        else
            for n = 1:size(test_point,1)
                leaves = testTrees([test_point(n,:) 0],trees);
                
                % average the class distributions of leaf nodes of all trees
                p_rf = trees(1).prob(leaves,:);
                if length(trees) == 1
                    p_rf_sum = p_rf;
                else
                    p_rf_sum = sum(p_rf)/length(trees);
                end
                all_probs(n,:) = p_rf_sum;
            end
            
            % Accuracy
            true_labels = [];
            predicted_labels = [];
            for j = 1:size(test_point,1)
                true_labels(j) = data_test(j,3);
                [~, predicted_labels(j)] = max(all_probs(j,:));
            end
            accuracy(i) = sum(true_labels == predicted_labels)/size(test_point,1)*100;
        end
    end
    global_accuracy_split(iter,:) = accuracy;
end

subplot(3,1,3)
plot(splitNum_trees,mean(global_accuracy_split),'-x','linewidth',2)
xlabel('Number of split functions')
xlim([min(splitNum_trees) max(splitNum_trees)])
ylim([min(mean(global_accuracy_split)) 100])
ylabel('Accuracy [%]')

end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% QUESTION 3: Caltech101 dataset for image categorisation %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if question == 3

init;
numBins = [2 4 8 16 32 64 128 256 512 1024];
mode = 'two';
for i=1:length(numBins)
    %% Question 3-1: K-means codebook
    
    % Select dataset
    % we do bag-of-words technique to convert images to vectors (histogram of codewords)
    % Set 'showImg' in getData.m to 0 to stop displaying training and testing images and their feature vectors
    [data_train, data_test, stop_kMeans(i), images_training, images_testing] = getData('Caltech',numBins(i));
    
    %% Question 3-2: RF classifier
    for it = 1:5 %10 % Iterate the random forest classifier over 10 trials
        % Set the random forest parameters
        param.num = 40; % The number of trees
        param.depth = 8; % Maximum depth of the trees
        param.splitNum = 20; % Number of set of split parameters theta i.e. p = 3
        param.split = 'IG'; % Objective function 'iformation gain' Degree of randomness parameter

        % Train Random Forest
        tic; % Start timer
        tree = growTrees(data_train,param,mode);
        stop_train(i) = toc; % Stop the timer

        % Evaluate/Test Random Forest
        tic; % Start timer
        for n=1:size(data_test,1) % Iterate through all rows of test data
            leaves = testTrees([data_test(n,:) 0],tree,mode); % Call the testTrees function
            size(leaves)
            % average the class distributions of leaf nodes of all trees
            p_rf = tree(1).prob(leaves,:);
            p_rf_sum = sum(p_rf)/length(tree);
            [~,predicted_label(n)] = max(p_rf_sum);
        end
        stop_test(i) = toc; % Stop the timer

        % Calculate accuracy of classifier
        actual_label = data_test(:,end);
        accuracy(it) = sum(actual_label == predicted_label')/length(actual_label)*100;
    end

    avg_accuracy(i) = mean(accuracy'); % Calculate average accuracy 
end

% Print and plot the accuracies
for i=1:length(numBins-1)
    fprintf('When numBins = %0.0f , average accuracy is %0.2f and time = %0.3f\n',numBins(i),avg_accuracy(i),...
        stop_kMeans(i)+stop_train(i)+stop_test(i));
end
figure
plot(numBins,avg_accuracy,'-ob');

%% Calculate the confusion matrix
num_classes = 10;
confusion_matrix = zeros(num_classes,num_classes); % initialize
for test_class = 1:num_classes
    idx_test = find(actual_label == test_class);
    if isempty(idx_test) == 0 % there is at least one value of this class
        for pred_class = 1:num_classes
            idx_pred = find(predicted_label(idx_test) == pred_class);
            if isempty(idx_pred) == 0 % there is at least one value of this class
                confusion_matrix(test_class,pred_class) = length(idx_pred)/length(predicted_label(idx_test));
            end
        end
    end
end

% Plot confusion matrix with colormap
figure
ax1 = axes('Position',[0 0 1 1],'Visible','off');
ax2 = axes('Position',[.2 .2 .7 .7]);
imagesc(confusion_matrix*100)
c = colorbar;
c.Label.String = 'Classification frequency [%]';
%title('Confusion matrix for 4-way classification')

% Create strings of the percentage in confusion matrix
value_perc = num2str(confusion_matrix(:)*100,'%0.2f');
value_perc = [value_perc, repmat('%',numel(confusion_matrix),1)];
value_perc = strtrim(cellstr(value_perc)); % remove any space padding

% Find the value in the middle of square to place text with accuracy
[x,y] = meshgrid(1:size(confusion_matrix,1));
strings = text(x(:),y(:),value_perc(:),'HorizontalAlignment','center');
mid_pos = mean(get(gca,'CLim')); % find middle value of the color range

% Change text colors
text_colors = repmat(confusion_matrix(:)*100 < mid_pos,1,3);
set(strings,{'Color'},num2cell(text_colors,2));

% Define labels and replace spaces with "newline" for better visualisation
labels = {'tick','trilobite','umbrella','watch','water lilly','wheelchair',...
          'wild cat','window chair','wrench','yin yang'};

% Change axis
set(gca,'XTick',1:size(confusion_matrix,1),'XTickLabel',labels,...
    'YTick',1:size(confusion_matrix,1),'YTickLabel',labels);    
%ytickangle(90); % rotate 90 degrees the Y axis labels

% Add the labels of actual and predicted class
axes(ax1) % sets ax1 to current axes
text(0.09,0.55,{'ACTUAL';'CLASS'})
text(0.45,0.12,'PREDICTED CLASS')

%% Plot example success/failures

folderName = 'Caltech_101/101_ObjectCategories';
classList = dir(folderName);
classList = {classList(3:end).name}; % 10 classes

numImages = 4; % to plot
success = predicted_label' == actual_label;

% Find index of images correctly classified and the ones missclassified
idx_success = find(success == 1);
idx_failure = find(success == 0);

% Randomly select some images to plot
rand_images_success = randsample(idx_success,numImages);
rand_images_failure = randsample(idx_failure,numImages);

% Create global list images used for testing
all_images_testing = [];
for row = 1:size(images_testing,1)
    all_images_testing = horzcat(all_images_testing, images_testing(row,:));
end

% Plot success and failures
fig_success = figure;
fig_failure = figure;
for m = 1:length(rand_images_success)
    % Success
    label = predicted_label(rand_images_success(m));
    num_image_folder = all_images_testing(rand_images_success(m));
    
    % Plot the correctly classified image
    subFolderName = fullfile(folderName,classList{label});
    imgList = dir(fullfile(subFolderName,'*.jpg'));
    I = imread(fullfile(subFolderName,imgList(num_image_folder).name));
    figure(fig_success)
    subplot(2,2,m)
    imshow(I);
    title(labels(label))
    
    % Failure
    pred_label = predicted_label(rand_images_failure(m));
    true_label = actual_label(rand_images_failure(m));
    num_image_folder = all_images_testing(rand_images_failure(m));
    
    % Plot the misclassified image
    subFolderName = fullfile(folderName,classList{true_label});
    imgList = dir(fullfile(subFolderName,'*.jpg'));
    I = imread(fullfile(subFolderName,imgList(num_image_folder).name));
    figure(fig_failure)
    subplot(2,2,m)
    imshow(I);
    title(labels(pred_label))
end
figure(fig_success)
suptitle('Correctly classified images')
    
figure(fig_failure)
suptitle('Misclassified images')

end

%% Question 3-3: RF codebook

init;

% Select dataset
% we do bag-of-words technique to convert images to vectors (histogram of codewords)
% Set 'showImg' in getData.m to 0 to stop displaying training and testing images and their feature vectors
[data_train, data_test] = getData_rf();
mode = 'axis';
for it = 1:10 % Iterate the random forest classifier over 10 trials
    % Set the random forest parameters
    param.num = 40; % The number of trees
    param.depth = 8; % Maximum depth of the trees
    param.splitNum = 20; % Number of set of split parameters theta i.e. p = 3
    param.split = 'IG'; % Objective function 'iformation gain' Degree of randomness parameter

    % Train Random Forest
    tic; % Start timer
    tree = growTrees(data_train,param,mode);
    stop_train(it) = toc; % Stop the timer

    % Evaluate/Test Random Forest
    tic; % Start timer
    for n=1:size(data_test,1) % Iterate through all rows of test data
        leaves = testTrees(data_test(n,:),tree,mode); % Call the testTrees function
        % average the class distributions of leaf nodes of all trees
        p_rf = tree(1).prob(leaves,:);
        p_rf_sum = sum(p_rf)/length(tree);
        [~,predicted_label(n)] = max(p_rf_sum);
    end
    stop_test(it) = toc; % Stop the timer

    % Calculate accuracy of classifier
    actual_label = data_test(:,end);
    accuracy(it) = sum(actual_label == predicted_label')/length(actual_label)*100;
end
std(accuracy')
avg_accuracy = mean(accuracy'); % Calculate average accuracy 
avg_stop_train = mean(stop_train);
avg_stop_test = mean(stop_test);
avg_time = avg_stop_train + avg_stop_test; % Calculate average computation time 
