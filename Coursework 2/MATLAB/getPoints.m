
function [x, y] = getPoints(folderName,imgList,image)
% Obtain set of corresponding points by clicking on the same interest point
% in the different images.
x = [];
y = [];
if image == 'A'
    i = 1; % image index
elseif image == 'B'
    i = 2;
end

switch image
    case 'both'
        for i = 1:2 %size(imgList,1)
            I = rgb2gray(imread(fullfile(folderName,imgList(i).name)));
            figure(i);
            imshow(I);
            [X, Y] = getpts; % select points from image
            x(i,:) = round(X); % pixel value
            y(i,:) = round(Y);
            close(gcf) % close figure
        end
    otherwise
        I = rgb2gray(imread(fullfile(folderName,imgList(i).name)));
        figure;
        imshow(I);
        [X, Y] = getpts; % select points from image
        x = round(X'); % pixel value
        y = round(Y');
        close(gcf) % close figure
end
end