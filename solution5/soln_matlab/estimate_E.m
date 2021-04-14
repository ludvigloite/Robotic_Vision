function E = estimate_E(xy1, xy2)
    n = size(xy1, 2);
    A = zeros([n, 9]);
    for i=1:n
        x1 = xy1(1,i);
        y1 = xy1(2,i);
        x2 = xy2(1,i);
        y2 = xy2(2,i);
        A(i,:) = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1];
    end
    [~,~,V] = svd(A);
    E = reshape(V(:,9), [3,3])';
end
