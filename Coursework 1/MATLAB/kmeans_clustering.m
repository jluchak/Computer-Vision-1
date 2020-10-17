
function clusters = kmeans_clustering(trainIn, K, iterations)
% It applies the K-means algorithm, which partitions the data into K clusters.
% Inputs:
%      trainIn: A two-column matrix of Longitude and Latitude pairs,
%               where the rows correspond to different training data points.
%            K: A value that specifies the number of clusters to be created.
%   iterations: A value which is the maximum number of iterations allowed
%               if the algorithm does not converge before.
% Outputs:
%     clusters: A cell array with dimensions K x 1. Every component of this
%               cell array will be a vector with the position (Longitude
%               and Latitude) of all the points considered part of that cluster.

% Place centroids (c) at random locations (within allowed position range)
lat_min = min(trainIn(:,1));
lat_max = max(trainIn(:,1));
long_min = min(trainIn(:,2));
long_max = max(trainIn(:,2));

% Random locations
centroids(:,1) = lat_min + rand(1,K)*(lat_max - lat_min); % rand latitude
centroids(:,2) = long_min + rand(1,K)*(long_max - long_min); % rand longitude

% Start k-means clustering iteration
i = 1;
while i < iterations
    clusters = cell(K,1); % initialize cell array
    % Find nearest centroid for each point
    for point = 1:size(trainIn,1)
        for c = 1:K
            % Calculate the Euclidian distance from the point to every centroid
            dist(c) = sqrt(sum((trainIn(point,:)-centroids(c,:)).^2));
        end
        [~, nearest_centroid] = min(dist);
        
        % Check if list points for this centroid has already been created
        if isempty(clusters{nearest_centroid})
            clusters{nearest_centroid} = trainIn(point,:);
        else
            clusters{nearest_centroid}(end+1,:) = trainIn(point,:); % append point
        end
    end
  
    % Find new centroid (mean of all points in cluster)
    for c = 1:K
        % Check if none of the cluster assignment change
        if isequal(centroids(c,:), mean(clusters{c})) == 1
            change_centroids = 0; % there has been no change
        else
            change_centroids = 1;
        end
        % I use a modified formula to avoid empty clusters
        if isempty(clusters{c})
            centroids(c,:) = 1/(size(clusters{c},1)+1)*centroids(c,:);
        else
            centroids(c,:) = 1/(size(clusters{c},1)+1)*(sum(clusters{c},1) + centroids(c,:));
        end
        %centroids(c,:) = mean(clusters{c}); % typical k-means might create empty clusters
    end
    
    % If the clusters stay the same, it has converged
    if change_centroids == 0
        break
    end
  
    i = i + 1; % repeat until convergence
end

% % Plot the clusters in 2D
% figure
% hold on
% %title(['Clusters of data points generated for K = ', num2str(K), ''])
% colorVec = hsv(K); % create a color map saturation
% for i = 1:K
%     for j = 1:size(clusters{i},1)
%         plot(clusters{i}(j,1),clusters{i}(j,2), '.', 'Color', colorVec(i,:))
%     end
% end
% xlabel('Latitude [^\circ]')
% ylabel('Longitude [^\circ]')

end
