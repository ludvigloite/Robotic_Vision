% This is just a suggestion for how you might
% structure your implementation. Feel free to
% make changes e.g. taking in other arguments.

function p = gauss_newton(residualsfun, p0)
    % See the comment in part1.m regarding the 'residualsfun' argument.

    step_size = 0.25;
    num_iterations = 100;
    finite_difference_epsilon = 1e-5;
    
    p = p0;
    for iteration=1:num_iterations
        % 1: Compute the Jacobian matrix J, using e.g.
        %    finite differences with the given epsilon.

        % 2: Form the normal equation terms JTJ and JTr.

        % 3: Solve for delta and update p as
        % p = p + step_size*delta;
    end
end
