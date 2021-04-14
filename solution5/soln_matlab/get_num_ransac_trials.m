function n = get_num_ransac_trials(sample_size, confidence, inlier_fraction)
    n = log(1 - confidence)/log(1 - inlier_fraction^sample_size);
    n = ceil(n);
end
