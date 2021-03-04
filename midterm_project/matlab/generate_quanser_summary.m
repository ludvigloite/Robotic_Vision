function generate_quanser_summary(trajectory, all_residuals, detections)
    %
    % Print reprojection error statistics
    %
    reprojection_errors = [];
    for i=1:size(trajectory,1)
        weights = detections(i, 1:3:end);
        r = reshape(all_residuals(i,:), [], 2)';
        e = vecnorm(r);
        e = e(weights==1); % Keep only valid reprojection errors
        reprojection_errors = [reprojection_errors e];
    end
    fprintf('Reprojection error over whole image sequence:\n');
    fprintf('- Maximum: %.04f pixels\n', max(reprojection_errors));
    fprintf('- Average: %.04f pixels\n', mean(reprojection_errors));
    fprintf('- Median: %.04f pixels\n', median(reprojection_errors));

    %
    % Figure: Reprojection error distribution
    %
    fig = figure(2);
    clf(fig);
    histogram(reprojection_errors, 'NumBins', 80);
    ylabel('Frequency');
    xlabel('Reprojection error (pixels)');
    title('Reprojection error distribution');

    %
    % Figure: Comparison between logged encoder values and vision estimates
    %
    logs      = load('../data/logs.txt');
    enc_time  = logs(:,1);
    enc_yaw   = logs(:,2);
    enc_pitch = logs(:,3);
    enc_roll  = logs(:,4);

    % Note: The logs have been time-synchronized with the image sequence,
    % but there will be an offset between the motor angles and the vision
    % estimates. That offset is automatically subtracted here.
    vis_yaw = trajectory(:,1) + enc_yaw(1) - trajectory(1,1);
    vis_pitch = trajectory(:,2) + enc_pitch(2) - trajectory(1,2);
    vis_roll = trajectory(:,3) + enc_roll(3) - trajectory(1,3);

    vis_fps  = 16;
    enc_frame = enc_time*vis_fps;
    vis_frame = 0:(size(trajectory,1)-1);

    fig = figure(3);
    clf(fig);

    subplot(311);
    plot(enc_frame, enc_yaw); hold on;
    plot(vis_frame, vis_yaw);
    legend('Encoder log', 'Vision estimate');
    xlim([0, vis_frame(end)]);
    ylim([-1, 1]);
    ylabel('Yaw (radians)');
    title('Helicopter trajectory');

    subplot(312);
    plot(enc_frame, enc_pitch); hold on;
    plot(vis_frame, vis_pitch);
    xlim([0, vis_frame(end)]);
    ylim([-0.25, 0.6]);
    ylabel('Pitch (radians)');

    subplot(313);
    plot(enc_frame, enc_roll); hold on;
    plot(vis_frame, vis_roll);
    xlim([0, vis_frame(end)]);
    ylim([-0.6, 0.6]);
    ylabel('Roll (radians)');
    xlabel('Image number');
end
