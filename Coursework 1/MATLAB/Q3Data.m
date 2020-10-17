function [data_train, data_query ] = Q3Data( MODE )    
% Generate training and testing data
showImg = 0; % Show training & testing images and their image feature vector (histogram representation)

PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
PHOW_Step = 8; % The lower the denser. Select from {2,4,8,16}

switch MODE
        
    case 'Caltech' % Caltech dataset
        close all;
        imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)
        folderName = './Caltech_101/101_ObjectCategories';
        classList = dir(folderName);
        classList = {classList(3:end).name} % 10 classes
        
        disp('Loading training images...')
        % Load Images -> Description (Dense SIFT)
        cnt = 1;
        if showImg
            figure('Units','normalized','Position',[.05 .1 .4 .9]);
            suptitle('Training image samples');
        end
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx{c} = randperm(length(imgList));
            imgIdx_tr = imgIdx{c}(1:imgSel(1));
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
            
            for i = 1:length(imgIdx_tr)
                I = imread(fullfile(subFolderName,imgList(imgIdx_tr(i)).name));
                
                % Visualise
                if i < 6 & showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3
                    I = rgb2gray(I); % PHOW work on gray scale image
                end
                
                % For details of image description, see http://www.vlfeat.org/matlab/vl_phow.html
                [~, desc_tr{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
            end
        end
        
        disp('Building visual codebook...')
        % Build visual vocabulary (codebook) for 'Bag-of-Words method'
        desc_sel = single(vl_colsubset(cat(2,desc_tr{:}), 10e4)); % Randomly select 100k SIFT descriptors for clustering
        
        %% Data information
        % desc_sel: 128 x 100000 single - A set of 100,000 patch vector
        % that have 128 elements in each vector
        %
        % desc_tr: 10 x 15 cells -  Each cell is a training image that is 
        % converted into a number of patch vectors with 128 elements.
        % 10 different classes with 15 images in each class.
        %
        % desc_te: 10 x 15 cells - Each cell is a testing image that is 
        % converted into a number of patch vectors with 128 elements.
        % 10 different classes with 15 images in each class.
        
        %% K-means clustering for each patch in the entire data set
       
        numBins = 256; % Define the number of clusters to use in kmeans

        % Call the kmeans function
        %   index: 100,000 x 1 - Returns the cluster index for all 
        %          100,000 patch vectors
        %
        %   center: numBins x 128 returns the centroid location of each
        %           cluster in desc_sel.
        tic; % Initiate the timer
        [~, center] = kmeans(desc_sel,numBins); 

        stop = toc; % Stop the timer
        fprintf('TIC TOC K-means: %g\n', stop); % Print the time it takes to run kmeans

        %% Vector Quantization training data
        disp('Encoding Images...')
        
        tic; % Initiate timer
        numb_images = 0; % Initiate number of images counter
        data_train = zeros(numel(desc_tr),numBins+1); % Initiate data_train 
        
        % For every single image in the training data set, 
        % determine the closest cluster from the kmeans algorithm relative 
        % to each patch vector, and create a Bag Of Words (BOW).
        for r = 1:1:size(desc_tr,1) % Iterate through all rows
            for c = 1:1:size(desc_tr,2) % Iterate through all columns
                image = desc_tr{r,c}; % Define the current loops image
                hist_vector = zeros(1,length(image)); % Reset
                
                for i = 1:1:length(image) % Iterate through all patch vectors
                    db_I = double(image(:,i)); % Convert to double
                    db_center = double(center'); % Convert to double and transpose
                    
                    % Calculate the distance between the current patch 
                    % vector and all of the clusters
                    distances = vecnorm(db_I - db_center);
                    
                    % Determine the index of the cluster corresponding to 
                    % the minimum distance and print it into a vector
                    [~,cluster_num] = min(distances);
                    hist_vector(i) = cluster_num;
                end
                % Use the histogram function to create a bag of words for every image  
                numb_images = numb_images + 1; % Update number of iamges counter
                [BOW,~] = hist(hist_vector,numBins); % Generate bag of words vector
                
                % Visual Vocabulary - Training Data
                data_train(numb_images,1:end-1) = BOW; % Store BOW words
                data_train(numb_images,end) = r; % Label
            end
        end
        stop = toc; % Stop the timer
        fprintf('TIC TOC bag of words training: %g\n', stop);
        
        % Plot a histogram for training data bag of words
        histogram(hist_vector,numBins)
        xlabel('Codewords')
        ylabel('frequency')
        title('Visual bag of words of a training image')
                
        % Clear unused varibles to save memory
        clearvars desc_tr desc_sel
        
end

switch MODE
    case 'Caltech'
        if showImg
        figure('Units','normalized','Position',[.05 .1 .4 .9]);
        suptitle('Testing image samples');
        end
        disp('Processing testing images...');
        cnt = 1;
        % Load Images -> Description (Dense SIFT)
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
            
            for i = 1:length(imgIdx_te)
                I = imread(fullfile(subFolderName,imgList(imgIdx_te(i)).name));
                
                % Visualise
                if i < 6 & showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3
                    I = rgb2gray(I);
                end
                [~, desc_te{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step);
            
            end
        end
        suptitle('Testing image samples');
                if showImg
            figure('Units','normalized','Position',[.5 .1 .4 .9]);
        suptitle('Testing image representations: 256-D histograms');
        end

%% Vector Quantization testing data
        disp('Encoding Images...')
        
        tic; % Initiate timer
        numb_images = 0; % Initiate number of images counter
        data_test = zeros(numel(desc_te),numBins); % Initiate data_test 
        
        % For every single image in the testing data set, 
        % determine the closest cluster from the kmeans algorithm relative 
        % to each patch vector, and create a Bag Of Words (BOW).
        for r = 1:1:size(desc_te,1) % Iterate through all rows
            for c = 1:1:size(desc_te,2) % Iterate through all columns
                image = desc_te{r,c}; % Define the current loops image
                hist_vector = zeros(1,length(image)); % Reset
                
                for i = 1:1:length(image) % Iterate through all patch vectors
                    db_I = double(image(:,i)); % Convert to double
                    db_center = double(center'); % Convert to double and transpose
                    
                    % Calculate the distance between the current patch 
                    % vector and all of the clusters
                    distances = vecnorm(db_I - db_center);
                    
                    % Determine the index of the cluster corresponding to 
                    % the minimum distance and print it into a vector
                    [~,cluster_num] = min(distances);
                    hist_vector(i) = cluster_num;
                end
                % Use the histogram function to create a bag of words for every image  
                numb_images = numb_images + 1; % Update number of images counter
                [BOW,~] = hist(hist_vector,numBins); % Generate bag of words vector
                
                % Visual Vocabulary - Testing Data
                data_test(numb_images,:) = BOW; % Store BOW words
            end
        end
        stop = toc; % Stop the timer
        fprintf('TIC TOC bag of words testing: %g\n', stop);
        
        % Plot a histogram for testing data bag of words
        histogram(hist_vector,numBins)
        xlabel('Codewords')
        ylabel('frequency')
        title('Visual bag of words of a testing image')
                
        % Clear unused varibles to save memory
        clearvars desc_te
        
    otherwise % Dense point for 2D toy data
        xrange = [-1.5 1.5];
        yrange = [-1.5 1.5];
        inc = 0.02;
        [x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
        data_query = [x(:) y(:) zeros(length(x)^2,1)];
end
end




