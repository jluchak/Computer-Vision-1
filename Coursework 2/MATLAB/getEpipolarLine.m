
function ep_line_y = getEpipolarLine(point1, point2, folderName, imgList, image)
% Epipolar line in A contains interest point A and is obtained with
% interest point A and epipole A, viceversa for image B.

% Find line equation
syms x y
eqn = (y - point1(2) == (point2(2)-point1(2))/(point2(1)-point1(1))*(x - point1(1)));
v_y = solve(eqn, y);
ep_line_y = vpa(v_y, 5); % epipolar line equation for the image

% Plot image and epipolar line
if image == 'A'
    i = 1; % image index
elseif image == 'B'
    i = 2;
end
I = rgb2gray(imread(fullfile(folderName,imgList(i).name)));
figure;
imshow(I);
hold on
fplot(ep_line_y,[1,size(I,2)],'Color','blue','LineWidth',2)
plot(point1(1), point1(2), 'm.', 'MarkerSize', 20)
plot(point2(1), point2(2), 'r.', 'MarkerSize', 20)
%title(['\color{blue}Epipolar line \color{black}in image ', image, ' containing \color{magenta}epipole \color{black}and \color{red}interest point'], 'FontSize', 20)

end
