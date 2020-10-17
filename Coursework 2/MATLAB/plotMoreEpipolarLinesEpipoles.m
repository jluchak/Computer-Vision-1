
function plotMoreEpipolarLinesEpipoles(folderName, imgList, F)

epipole_A = getEpipoles(folderName, imgList, 'A', F);
epipole_B = getEpipoles(folderName, imgList, 'B', F);
I_l = rgb2gray(imread(fullfile(folderName,imgList(1).name)));
I_r = rgb2gray(imread(fullfile(folderName,imgList(2).name)));
close all

figure(1);
% Left image
subplot(1,2,1)
imshow(I_l);
hold on

% Calculate epipolar lines for the image A
point1 = epipole_A;
point2 = [];
[x, y] = getPoints(folderName, imgList, 'A');
for i = 1:length(x)
    point2 = [x(i);y(i)];
    ep_line_y = getEpipolarLine(point1, point2, folderName, imgList, 'A');
    figure(1)
    fplot(ep_line_y,[1,size(I_l,2)],'Color','blue','LineWidth',2)
    plot(point2(1), point2(2), 'r.', 'MarkerSize', 20)
end
plot(epipole_A(1), epipole_A(2), 'm.', 'MarkerSize', 20)
%title('(A) Left image', 'FontSize', 26)
%title('Epipolar lines and epipole in image A', 'FontSize', 20)

% Right image
subplot(1,2,2)
imshow(I_r);
hold on

% Calculate epipolar lines for the image B
point1 = epipole_B;
point2 = [];
[x, y] = getPoints(folderName, imgList, 'B');
for i = 1:length(x)
    point2 = [x(i);y(i)];
    ep_line_y = getEpipolarLine(point1, point2, folderName, imgList, 'B');
    figure(1)
    fplot(ep_line_y,[1,size(I_r,2)],'Color','blue','LineWidth',2)
    plot(point2(1), point2(2), 'r.', 'MarkerSize', 20)
end
plot(epipole_B(1), epipole_B(2), 'm.', 'MarkerSize', 20)
%title('(B) Right image', 'FontSize', 26)
pos = get(gca, 'Position');
pos(1) = 0.47; % x
set(gca, 'Position', pos)

end