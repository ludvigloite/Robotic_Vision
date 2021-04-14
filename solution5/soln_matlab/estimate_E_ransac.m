function [E,inliers] = estimate_E_ransac(xy1, xy2, K, distance_threshold, num_trials)

    % Tip: The following snippet extracts a random subset of 8
    % correspondences (w/o replacement) and estimates E using them.
    %   sample = randperm(size(xy1, 2), 8);
    %   E = estimate_E(xy1(:,sample), xy2(:,sample));

    uv1 = K*xy1;
    uv2 = K*xy2;

    fprintf('Running RANSAC with %g px inlier threshold and %d trials...', distance_threshold, num_trials);
    best_num_inliers = -1;
    for i=1:num_trials
        sample = randperm(size(xy1, 2), 8);
        E_i = estimate_E(xy1(:,sample), xy2(:,sample));
        d_i = epipolar_distance(F_from_E(E_i, K), uv1, uv2);
        inliers_i = abs(d_i) < distance_threshold;
        num_inliers_i = sum(inliers_i);
        if num_inliers_i > best_num_inliers
            best_num_inliers = num_inliers_i;
            inliers = inliers_i;
            E = E_i;
        end
    end
    fprintf('Done!\n');
    fprintf('Found solution with %d/%d inliers.\n', sum(inliers), size(xy1, 2));
end
