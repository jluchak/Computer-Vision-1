
function A = buildMatrixA(x,y,type)
% Create matrix A used to solve Ah = 0 and Af = 0
A = [];
for i = 1:size(x,2)
    % Points of image A and B
    points_a = [x(1,i); y(1,i)];
    points_b = [x(2,i); y(2,i)];

    switch type
        case 'homography'
            % Add all set of points in a matrix
            A = [A;
                 0 0 0 -points_b(1) -points_b(2) -1 points_a(2)*points_b(1) points_a(2)*points_b(2) points_a(2);
                 -points_b(1) -points_b(2) -1 0 0 0 points_a(1)*points_b(1) points_a(1)*points_b(2) points_a(1)];
        case 'fundamental'
            A = [A;
                 points_a(1)*points_b(1) points_a(1)*points_b(2) points_a(1) points_a(2)*points_b(1) points_a(2)*points_b(2) points_a(2) points_b(1) points_b(2) 1];
    end
end

end