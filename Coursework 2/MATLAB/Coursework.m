
clear all
close all
clc
                        %%%%%%%%%%%%%%%%%%
                        %% Q1: Matching %%
                        %%%%%%%%%%%%%%%%%%
%% 1) Manual
% Select at least 4 points for finding the homography matrix
folderName = 'sequences_images/myimages/FD';
imgList = dir(fullfile(folderName,'*.jpg'));
[x, y] = getPoints(folderName, imgList, 'both'); % select at least 4 points to find H

%% 3) Transformation estimation

%% Q1.3a) Homography matrix %%%
A = buildMatrixA(x,y,'homography'); % find matrix A
[U,S,V] = svd(A); % singular value decomposition
h = V(:,end)/V(end,end); % homography transfo'rmation
H = [h(1) h(2) h(3);
     h(4) h(5) h(6);
     h(7) h(8) h(9)];

%% Q1.3b) Fundamental matrix %%%
[x, y] = getPoints(folderName, imgList, 'both'); % select at least 8 points to find F
A = buildMatrixA(x,y,'fundamental');
[U,S,V] = svd(A); % singular value decomposition
f = V(:,end)/V(end,end);
F = [f(1) f(2) f(3);
     f(4) f(5) f(6);
     f(7) f(8) f(9)];

% Using Matlab functions...
matchedPoints1 = [];
matchedPoints2 = [];
for i = 1:size(x,2)
    % Points of image A and B
    matchedPoints1 = [matchedPoints1; x(1,i), y(1,i)];
    matchedPoints2 = [matchedPoints2; x(2,i), y(2,i)];
end
f8norm = estimateFundamentalMatrix(matchedPoints2,matchedPoints1,'Method','Norm8Point');
fRANSAC = estimateFundamentalMatrix(matchedPoints1,matchedPoints2, 'Method', 'RANSAC', 'NumTrials', 2000, 'DistanceThreshold', 1e-4);
[isIn,epipole] = isEpipoleInImage(fRANSAC,size(I_l));

%% Q1.3c) Homography accuracy %%%
% Take new set of points (>= 1 pair) from the images
[x, y] = getPoints(folderName, imgList, 'both');
HA = computeHomographyAccuracy(x,y,H,folderName,imgList);

%% Q1.3d) Fundamental matrix accuracy %%%

% Calculate coordinates of EPIPOLES (epipole can't be 0)
image = 'A';
epipole = getEpipoles(folderName, imgList, image, F);
% Check that F*[epipole_B;1] = 0 and F'*[epipole_A;1] = 0

% Calculate epipolar lines for the image
point1 = epipole;
point2 = [];
[point2(1), point2(2)] = getPoints(folderName, imgList, image); % select just one point from that image
ep_line_y = getEpipolarLine(point1, point2, folderName, imgList, image);

% Select points in image B and obtain the EPIPOLAR LINES in A
[x, y] = getPoints(folderName, imgList, 'B');
FA = computeFundamentalAccuracy(x,y,F,folderName,imgList);

                        %%%%%%%%%%%%%%%%%%%%%%%%
                        %% Q2: Image Geometry %%
                        %%%%%%%%%%%%%%%%%%%%%%%%
%% 2) Stereo Vision %%

I_l = rgb2gray(imread(fullfile(folderName,imgList(1).name))); % left image: A
I_r = rgb2gray(imread(fullfile(folderName,imgList(2).name))); % right image: B

% Define parameters
f = 26; % focal length of camera (typically 18-55mm)
b = 200; % baseline (distance between left and right camera: 20cm)

%% Q2.2b) Epipoles and epipolar lines for both images %%%
plotMoreEpipolarLinesEpipoles(folderName, imgList, fRANSAC)

%% Q2.2b) Disparity map %%%
% The closer the object, the larger the disparity.
% Disparity: d = x_l - x_r;

% Window of pixels
W = [5 21]; % size window: only odd
originalSize = [];
a = [];
figure
for i = 1:length(W)
    a(i) = subplot(1,2,i);
    disparity_map = disparityMap(I_l, I_r, W(i));
%     norm_disparity_map = (disparity_map-min(range_d))/(max(range_d)-min(range_d));
%     imshow(norm_disparity_map,[0 1]);
    imshow(disparity_map, [0, size(I_l,2)-1]);
    colormap(gca,jet)
    originalSize(i,:) = get(gca, 'Position');
    title(['W = ', num2str(W(i))],'FontSize',20)
end
c = colorbar('FontSize',16);
c.Label.String = 'Disparity (pixels)';
c.Label.FontSize = 18;
set(a(1), 'Position', originalSize(1,:))
set(a(2), 'Position', originalSize(2,:))
pos = get(a(2), 'Position');
pos(1) = 0.48; % x
set(a(2), 'Position', pos)

%% Q2.2c) Q2.2d) Depth maps %%%
% Depth is inversely proportional to disparity

% Samsung Galaxy S7: Sensor size (5.76mm x 4.29mm) and 12MP
% Pixel to mm
%disparity_mm = disparity_map*5.76/(12*10^6);

a = [];

% 1 - Original depth map
z = f*b./disparity_map;
z(z == Inf) = max(z(isfinite(z))); % cap max depth
figure
a(1) = subplot(1,3,1);
imshow(z)
originalSize1 = get(gca, 'Position');
title('Original depth map','FontSize',20);
colormap(gca,gray);

% 2 - Changing focal length
new_f = f+2;
z = new_f*b./disparity_map;
z(z == Inf) = max(z(isfinite(z))); % cap max depth
a(2) = subplot(1,3,2);
imshow(z)
originalSize2 = get(gca, 'Position');
title('Changing focal length','FontSize',20);
colormap(gca,gray);

% 3 - Add random noise to the disparity map
mean_noise = 1;
std_noise = 1;
noise = normrnd(mean_noise, std_noise, size(disparity_map,1), size(disparity_map,2)); 
disparityMapNoise = disparity_map + noise;
%J = imnoise(disparity_map, 'gaussian', 1, 0.5);
% figure
% imshow(disparityMapNoise, [0, size(I_l,2)-1]);
% colormap(gca,jet)
z = f*b./disparityMapNoise;
z(z == Inf) = max(z(isfinite(z))); % cap max depth
a(3) = subplot(1,3,3);
imshow(z);
originalSize3 = get(gca, 'Position');
title('Adding noise','FontSize',20);
colormap(gca,gray);
c = colorbar('FontSize',16);
c.Label.String = 'Depth (mm)';
c.Label.FontSize = 18;
c.Ticks = linspace(0, 1, 12);
c.TickLabels = [' ',num2cell(500:500:5000), ' '];
set(a(1), 'Position', originalSize1)
set(a(2), 'Position', originalSize2)
set(a(3), 'Position', originalSize3)
pos = get(a(2), 'Position');
pos(1) = 0.36; % x
set(a(2), 'Position', pos)
pos = get(a(3), 'Position');
pos(1) = 0.59; % x
set(a(3), 'Position', pos)

%% Q2.2e) Stereo image rectification %%%

% Create orthogonal unit vectors
r1 = [epipole/norm(epipole);0];
r2 = [-epipole(2), epipole(1), 0]'/norm(epipole);
r3 = cross(r1,r2);

% Orthogonal matrix
R_rect = [r1'; r2'; r3'];

R = eye(3); % rotation matrix
R_l = R_rect; % left rotation matrix
R_r = R*R_rect; % right rotation matrix

% Obtain rectified images
tic
I_l_rectified = rectifyImage(I_l,R_l,f); % left image
I_r_rectified = rectifyImage(I_r,R_r,f); % right image
toc

figure
imshowpair(I_l_rectified,I_r_rectified,'montage')

%% MATLAB FUNCTIONS

I1 = imread('scene1.row3.col1.ppm'); 
I2 = imread('scene1.row3.col2.ppm');
I1gray = rgb2gray(I1);
I2gray = rgb2gray(I2);

%% Disparity map and depth with Tsukuba
disparityRange = [0 16];
disparity_tsukuba = disparity(I1gray,I2gray,'BlockSize',21,'DisparityRange',disparityRange);
a = [];
figure
a(1) = subplot(1,2,1);
imshow(disparity_tsukuba,disparityRange);
title('Disparity map','FontSize',20);
colormap(gca,jet)
originalSize1 = get(gca, 'Position');
c = colorbar('FontSize',16);
c.Label.String = 'Disparity (pixels)';
c.Label.FontSize = 18;

z = f*b./disparity_tsukuba;
z(z == Inf) = max(z(isfinite(z))); % cap max depth
a(2) = subplot(1,2,2);
imshow(z,[unique(min(min(z))) unique(max(max(z)))]);
title('Depth map','FontSize',20);
colormap(gca,gray)
originalSize2 = get(gca, 'Position');
c = colorbar('FontSize',16);
c.Label.String = 'Depth (mm)';
c.Label.FontSize = 18;

set(a(1), 'Position', [originalSize1(1)-0.05 originalSize1(2:4)])
set(a(2), 'Position', originalSize2)
pos = get(a(2), 'Position');
pos(1) = 0.53; % x
set(a(2), 'Position', pos)

%% Stereo rectified images with Tsukuba
% Set 1 to visualize and 0 else 
visualize = 1; 

if (visualize == 1)
    figure;
    imshowpair(I1, I2,'montage');
    title('I1 (left); I2 (right)');
    figure(2);
    imshow(stereoAnaglyph(I1,I2));
    title('Composite Image (Red - Left Image, Cyan - Right Image)');
end 

% Collect interest points 

blobs1 = detectSURFFeatures(I1gray, 'MetricThreshold', 2000);
blobs2 = detectSURFFeatures(I2gray, 'MetricThreshold', 2000);

if (visualize == 1)
    figure;
    imshow(I1);
    hold on;
    plot(selectStrongest(blobs1, 30));
    title('Thirty strongest SURF features in I1');

    figure;
    imshow(I2);
    hold on;
    plot(selectStrongest(blobs2, 30));
    title('Thirty strongest SURF features in I2');
end 

% Find point correspondences

[features1, validBlobs1] = extractFeatures(I1gray, blobs1);
[features2, validBlobs2] = extractFeatures(I2gray, blobs2);

% Match featues using SAD
indexPairs = matchFeatures(features1, features2, 'Metric', 'SAD','MatchThreshold', 5);

matchedPoints1 = validBlobs1(indexPairs(:,1),:);
matchedPoints2 = validBlobs2(indexPairs(:,2),:);

if (visualize == 1)
    figure;
    showMatchedFeatures(I1, I2, matchedPoints1, matchedPoints2);
    legend('Putatively matched points in I1', 'Putatively matched points in I2');
end 

% Remove outliers using Epopolar Constraints

[fMatrix, epipolarInliers, status] = estimateFundamentalMatrix(...
  matchedPoints1, matchedPoints2, 'Method', 'RANSAC', ...
  'NumTrials', 10000, 'DistanceThreshold', 0.8, 'Confidence', 99.99);

if status ~= 0 || isEpipoleInImage(fMatrix, size(I1)) ...
  || isEpipoleInImage(fMatrix', size(I2))
  error(['Either not enough matching points were found or '...
         'the epipoles are inside the images. You may need to '...
         'inspect and improve the quality of detected features ',...
         'and/or improve the quality of your images.']);
end

inlierPoints1 = matchedPoints1(epipolarInliers, :);
inlierPoints2 = matchedPoints2(epipolarInliers, :);

if (visualize == 1)
    figure;
    showMatchedFeatures(I1, I2, inlierPoints1, inlierPoints2);
    legend('Inlier points in I1', 'Inlier points in I2');
end 

% Rectify Images

[t1, t2] = estimateUncalibratedRectification(fMatrix, ...
  inlierPoints1.Location, inlierPoints2.Location, size(I2));
tform1 = projective2d(t1);
tform2 = projective2d(t2);

[I1Rect, I2Rect] = rectifyStereoImages(I1, I2, tform1, tform2);
if (visualize == 1)
    figure;
    imshowpair(I1Rect, I2Rect,'montage');
    figure;
    imshow(stereoAnaglyph(I1Rect, I2Rect));
    title('Rectified Stereo Images (Red - Left Image, Cyan - Right Image)');
end
