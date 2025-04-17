function plotFitnessForGAP12()
    % Path to gap12.txt
    fileName = './gap dataset files/gap12.txt';
    fileId = fopen(fileName, 'r');

    if fileId == -1
        error('Error opening file %s.', fileName);
    end

    totalCases = fscanf(fileId, '%d', 1);

    for caseIndex = 1:totalCases
        dims = fscanf(fileId, '%d', 2);
        m = dims(1);
        n = dims(2);

        costMatrix = zeros(m, n);
        for i = 1:m
            costMatrix(i, :) = fscanf(fileId, '%d', [1, n]);
        end

        resourceMatrix = zeros(m, n);
        for i = 1:m
            resourceMatrix(i, :) = fscanf(fileId, '%d', [1, n]);
        end

        capacity = fscanf(fileId, '%d', [m, 1]);

        % Run GA on this case
        [~, fitnessHistory] = solve_gap_ga_binary(m, n, costMatrix, resourceMatrix, capacity);

        % Plot for this case
        figure;
        plot(1:length(fitnessHistory), fitnessHistory, 'b-o', 'LineWidth', 2);

        title(sprintf('Fitness Over Generations (gap12 - Case %d)', caseIndex));
        xlabel('Generation');
        ylabel('Fitness (Total Resource Utilization)');
        grid on;
    end

    fclose(fileId);
end
