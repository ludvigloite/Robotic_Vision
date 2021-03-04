clear; clc;

detections = load('../data/detections.txt');

% The script runs up to, but not including, this image.
run_until = 87; % Task 1.3
% run_until = 88; % Task 1.4
% run_until = size(detections, 1); % Task 1.7

% Change this if you want the Quanser visualization for a different image.
% (Can be useful for Task 1.4.)
visualize_number = 0;

quanser = Quanser;

% Initialize the parameter vector
p = [11.6 ; 28.9 ; 0.0]*pi/180; % Optimal for image number 0
% p = [0.0 ; 0.0 ; 0.0]; % For Task 1.5

all_residuals = [];
trajectory = [];
for image_number=0:run_until-1
    weights = detections(image_number + 1, 1:3:end);
    uv = [detections(image_number + 1, 2:3:end) ;
          detections(image_number + 1, 3:3:end) ];

    % Tip:
    % 'uv' is a 2x7 array of detected marker locations.
    % It is the same size in each image, but some of its
    % entries may be invalid if the corresponding markers were
    % not detected. Which entries are valid is encoded in
    % the 'weights' array, which is a 1D array of length 7.

    % Tip:
    % Make your optimization method accept a lambda function
    % to compute the vector of residuals. You can then reuse
    % the method later by passing a different lambda function.
    residualsfun = @(p) quanser.residuals(uv, weights, p(1), p(2), p(3));

    p = gauss_newton(residualsfun, p);

    % Note:
    % The plotting code assumes that p is a 3 x 1 column vector
    % and r is a 2n x 1 column vector (n=7), where the first
    % n elements are the horizontal residual components, and
    % the last n elements the vertical components.

    r = residualsfun(p);
    all_residuals = [all_residuals ; r'];
    trajectory = [trajectory ; p'];
    if image_number == visualize_number
        fprintf('Residuals on image number %d:\n', image_number); disp(r);
        quanser.draw(uv, weights, image_number);
    end
end

generate_quanser_summary(trajectory, all_residuals, detections);
