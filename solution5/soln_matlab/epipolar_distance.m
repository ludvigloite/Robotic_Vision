function r = epipolar_distance(F, uv1, uv2)
    % F should be the fundamental matrix (use F_from_E)
    % uv1, uv2 should be 3 x n homogeneous pixel coordinates

    l2 = F*uv1;
    l1 = F'*uv2;
    e = sum(uv2.*l2);
    % Alternatively:
    % n = size(uv1, 2);
    % e = zeros(1, n);
    % for i=1:n
    %     e(i) = uv2(:,i)'*l2(:,i);
    % end
    norm1 = vecnorm(l1(1:2,:));
    norm2 = vecnorm(l2(1:2,:));
    r = 0.5*e.*(1./norm1 + 1./norm2);
end
