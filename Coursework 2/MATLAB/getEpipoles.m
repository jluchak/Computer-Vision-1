
function epipole = getEpipoles(folderName, imgList, image, F)
% Epipole in image A is the intersection of all epipolar lines in image A
% Obtain:
    % Epipole in image B by solving F*e_b = 0 (since F is from B [x] to A [x'])
    % Epipole in image A by solving transpose(F)*e_a = 0
if image == 'A'
    i = 1; % image index
    [~, ~, V] = svd(F');
elseif image == 'B'
    i = 2;
    [~, ~, V] = svd(F); 
end
last_eigen = V(:,end);
epipole = last_eigen/last_eigen(3); % normalize by dividing by z
epipole = ceil(epipole(1:2)); % 2D point pipole in image B

% Plot image and epipole
I = rgb2gray(imread(fullfile(folderName,imgList(i).name)));
figure;
imshow(I);
hold on
plot(epipole(1), epipole(2), 'b.', 'MarkerSize', 20)
%title(['Epipole in image ', image], 'FontSize', 20)

end