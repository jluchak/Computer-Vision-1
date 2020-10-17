
function disparity_map = disparityMap(I_l, I_r, W)
disparity_map = zeros(size(I_l)); % initialize
for y = 1:size(I_l,1)
    for x = 1:size(I_l,2)  
        % Window
        xo = x - (W-1)/2;
        xf = x + (W-1)/2;
        yo = y - (W-1)/2;
        yf = y + (W-1)/2;
               
        % Check we are within limits
        range_y = yo:yf;
        idx_y_valid = find(range_y > 0 & range_y <= size(I_l,1));
        range_x = xo:xf;
        idx_x_valid = find(range_x > 0 & range_x <= size(I_l,2));

        % Create window matrix of intensity for left image
        initial_w_l = zeros(W,W);
        initial_w_l(idx_y_valid,idx_x_valid) = I_l(range_y(idx_y_valid),range_x(idx_x_valid));

        % Disparity
        %range_d = -(size(I_l,2)-x):(x-1); % explore all the points in the right image
        range_d = 0:(x-1); % in our case there will only be positive disparities
        
        C = NaN(1,length(range_d)); % SSD cost
        for idx_d = 1:length(range_d)
            d = range_d(idx_d); % disparity
            w_r = zeros(W,W); % right window will contain intensities
            
            % Check we are within limits
            range_xr = xo-d:xf-d;
            idx_xr_valid = find(range_xr > 0 & range_xr <= size(I_l,2));
            global_x_valid = idx_x_valid(ismembc(idx_x_valid,idx_xr_valid));
            
            % Change left window and keep only the valid rows
            w_l = zeros(W,W);
            w_l(idx_y_valid,global_x_valid) = initial_w_l(idx_y_valid,global_x_valid);
            
            % Create window matrix of intensities for right image
            w_r(idx_y_valid,global_x_valid) = I_r(range_y(idx_y_valid),range_xr(global_x_valid));
            
            % Assign SSD cost
            C(idx_d) = sum(sum((w_l - w_r).^2));
            
%             if C(idx_d) < 10e-5
%                 break
%             end
        end
        
        % Plot SSD cost vs disparity
%         figure
%         plot(range_d, C)
%         xlabel('Disparity')
%         ylabel('SSD')

        % Best matching disparity for this point: with highest similarity measure
        [~, index] = min(C);
        disparity_map(y,x) = range_d(index);
    end
end
end
