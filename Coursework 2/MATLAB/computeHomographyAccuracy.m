
function HA = computeHomographyAccuracy(x,y,H,folderName,imgList)
% Find the accuracy by projecting the points and compare with real ones
for i = 1:size(x,2)
    points_A = [x(1,i); y(1,i)];
    points_B = [x(2,i); y(2,i)];
    
    points_A_hom_coord = H*[points_B;1]; % estimated points A
    points_A_proj = points_A_hom_coord/points_A_hom_coord(3); % obtain projected values
    points_A_proj = points_A_proj(1:2); % only keep x and y
    %xa = (H(1,1)*points_B(1) + H(1,2)*points_B(2) + H(1,3))/(H(3,1)*points_B(1) + H(3,2)*points_B(2) + 1);
    %ya = (H(2,1)*points_B(1) + H(2,2)*points_B(2) + H(2,3))/(H(3,1)*points_B(1) + H(3,2)*points_B(2) + 1);
    
    % Calculate the distance between correct and estimated points
    distance_points(i) = pdist([points_A_proj(1),points_A_proj(2);
                                points_A(1),points_A(2)],'Euclidean');
                                
    % Plot one image of B and the corresponding projected point in A
    if i == 1
        % Point in image B
        I = rgb2gray(imread(fullfile(folderName,imgList(2).name))); % image B
        figure;
        subplot(1,2,1)
        imshow(I);
        hold on
        plot(points_B(1), points_B(2), 'b.', 'MarkerSize', 20)
        %title('Selected point in image B','FontSize',15)
        %title('(a)', 'FontSize', 20)

        % Corresponding projected point and original point in image A
        I = rgb2gray(imread(fullfile(folderName,imgList(1).name))); % image A
        subplot(1,2,2)
        imshow(I);
        hold on
        plot(points_A(1), points_A(2), 'b.', 'MarkerSize', 20)
        plot(points_A_proj(1), points_A_proj(2), 'r.', 'MarkerSize', 20)
        %title('(b)', 'FontSize', 20)
        %title('Corresponding \color{red}projected \color{black}and \color{blue}original \color{black}point in image A','FontSize',15)
        pos = get(gca, 'Position');
        pos(1) = 0.47; % x
        set(gca, 'Position', pos)
    end
end
HA = mean(distance_points); % homography accuracy

end