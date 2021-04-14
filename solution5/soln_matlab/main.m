clc;
clear;

K = load('../data/K.txt');
I1 = im2double(imread('../data/image1.jpg'));
I2 = im2double(imread('../data/image2.jpg'));

% ransac = false; % Part 2, 3
ransac = true; % Part 4
if ransac == true
    matches = load('../data/task4matches.txt');
else
    matches = load('../data/matches.txt');
end

uv1 = [matches(:,1:2)' ; ones(1, size(matches, 1))];
uv2 = [matches(:,3:4)' ; ones(1, size(matches, 1))];
xy1 = K\uv1;
xy2 = K\uv2;

if ransac == true
    figure(3);
    e = epipolar_distance(F_from_E(load('E.txt'), K), uv1, uv2);
    histogram(abs(e), 100, 'Normalization', 'cdf', 'BinLimits', [0, 40], 'DisplayStyle', 'stairs');
    title('Cumulative distribution of |epipolar distances| for good E');

    confidence = 0.99;
    inlier_fraction = 0.50;
    distance_threshold = 4.0;
    num_trials = get_num_ransac_trials(8, confidence, inlier_fraction)
    [~,inliers] = estimate_E_ransac(xy1, xy2, K, distance_threshold, num_trials);
    uv1 = uv1(:,inliers);
    uv2 = uv2(:,inliers);
    xy1 = xy1(:,inliers);
    xy2 = xy2(:,inliers);
end

E = estimate_E(xy1, xy2);
T_all = decompose_E(E);

if ransac == false
    writematrix(E, 'E.txt', 'Delimiter', 'tab')
end

best_num_visible = 0;
for i=1:4
    T = T_all(:,:,i);
    P1 = [1 0 0 0 ; 0 1 0 0 ; 0 0 1 0];
    P2 = T(1:3,:);
    X1 = triangulate_many(xy1, xy2, P1, P2);
    X2 = T*X1;
    num_visible = sum((X1(3,:) > 0) & (X2(3,:) > 0));
    if num_visible > best_num_visible
        best_num_visible = num_visible;
        best_i = i;
        best_X = X1;
    end
end
X = best_X;

rng(4); % Comment out to get a random selection each time. The seed value 4 seems to produce a nice distribution of points, at least on Matlab 2019b.
draw_correspondences(I1, I2, uv1, uv2, F_from_E(E, K));
draw_point_cloud(X, I1, uv1, [-1,+1], [-1,+1], [1,3]);
