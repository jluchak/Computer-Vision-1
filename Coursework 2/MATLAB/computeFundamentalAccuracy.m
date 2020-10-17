
function FA = computeFundamentalAccuracy(X,Y,F,folderName,imgList)
% Find the accuracy by finding the epipolar lines in image A (from the points in B)
for i = 1:size(X,2)
    points_B = [X(i); Y(i)];
    epipolar_line_A = F*[points_B(1);points_B(2);1]; % line coefficients: ax + by + c = 0

    % Distance from a point to a line
    xo = points_B(1);
    yo = points_B(2);
    a = epipolar_line_A(1);
    b = epipolar_line_A(2);
    c = epipolar_line_A(3);
    distance_point_line(i) = abs(a*xo + b*yo + c)/sqrt(a^2 + b^2);
    
    % Plot one image of B and the corresponding epipolar lines in A
    if i == 1
        % Equation line
        syms x y
        eqn = (a*x + b*y + c == 0);
        v_y = solve(eqn, y);
        ep_line_y = vpa(v_y, 5); % epipolar line equation for image A
        
        % Point in image B
        I = rgb2gray(imread(fullfile(folderName,imgList(2).name))); % image B
        figure;
        subplot(1,2,1)
        imshow(I);
        hold on
        plot(xo, yo, 'b.', 'MarkerSize', 20)
        %title('(a)', 'FontSize', 20)
        %title('Selected point in image B','FontSize',15)

        % Corresponding epipolar line in image A
        I = rgb2gray(imread(fullfile(folderName,imgList(1).name))); % image A
        subplot(1,2,2)
        imshow(I);
        hold on
        fplot(ep_line_y,[1,size(I,2)],'Color','blue','LineWidth',2)
        %title('(b)', 'FontSize', 20)
        %title('Corresponding epipolar line in image A','FontSize',15)
        pos = get(gca, 'Position');
        pos(1) = 0.47; % x
        set(gca, 'Position', pos)
    end
end
FA = mean(distance_point_line); % fundamental matrix accuracy

end