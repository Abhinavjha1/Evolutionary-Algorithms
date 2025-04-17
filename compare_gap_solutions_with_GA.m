function compare_gap_solutions_with_GA()
    % Initialize arrays for storing results
    exact_values = [];
    approx_values = [];
    ga_values = [];
    labels = {};

    % Output file path for CSV
    outputFile = 'gap_comparison_results.csv';
    if exist(outputFile, 'file') == 2
        delete(outputFile); % Delete the file if it already exists
    end
    % Write headers to CSV file
    headers = {'Case', 'Optimal_Cost (ILP)', 'Greedy_Cost', 'GA_Fitness'};
    writecell(headers, outputFile);

    for g = 1:12
        filename = sprintf('./gap dataset files/gap%d.txt', g);
        fid = fopen(filename, 'r');
        if fid < 0
            error('File cannot be opened: %s', filename);
        end

        num_instances = fscanf(fid, '%d', 1);
        fprintf('\n--- Reading: %s ---\n', filename(1:end-4));

        for p = 1:num_instances
            m = fscanf(fid, '%d', 1);
            n = fscanf(fid, '%d', 1);

            cost = fscanf(fid, '%d', [n, m])';
            resource = fscanf(fid, '%d', [n, m])';
            capacity = fscanf(fid, '%d', [m, 1]);

            % ILP (Optimal)
            x_opt = solve_gap_ilp(m, n, cost, resource, capacity);
            obj_opt = sum(sum(cost .* x_opt));

            % Greedy (Approximation)
            x_approx = greedy_gap_solver(m, n, cost, resource, capacity);
            obj_approx = sum(sum(cost .* x_approx));

            % GA (Genetic Algorithm)
            x_ga = solve_gap_ga_binary(m, n, cost, resource, capacity);
            obj_ga = sum(sum(cost .* x_ga));

            % Store results in the arrays
            exact_values(end + 1) = obj_opt;
            approx_values(end + 1) = obj_approx;
            ga_values(end + 1) = obj_ga;
            labels{end + 1} = sprintf('G%d-%d', g, p);

            % Write results for the current instance into the CSV
            new_row = {labels{end}, obj_opt, obj_approx, obj_ga};
            writecell(new_row, outputFile, 'WriteMode', 'append');

            fprintf('Instance %s | ILP: %d | Greedy: %d | GA: %d\n', labels{end}, obj_opt, obj_approx, obj_ga);
        end

        fclose(fid);
    end

    % Comparative Plot
    figure;
    hold on;
    plot(exact_values, '-o', 'LineWidth', 2, 'DisplayName', 'Optimal (ILP)');
    plot(approx_values, '-x', 'LineWidth', 2, 'DisplayName', 'Greedy Approximation');
    plot(ga_values, '-s', 'LineWidth', 2, 'DisplayName', 'Genetic Algorithm (GA)');
    hold off;

    title('Comparison of GAP Solutions: ILP vs. Greedy vs. GA');
    xlabel('Instance');
    ylabel('Objective Value');
    legend('Location', 'Best');
    xticks(1:length(labels));
    xticklabels(labels);
    xtickangle(45);
    grid on;
end

function x_matrix = solve_gap_ilp(m, n, c, r, b)
    num_vars = m * n;
    f = -reshape(c, [num_vars, 1]);
    intcon = 1:num_vars;
    lb = zeros(num_vars, 1);
    ub = ones(num_vars, 1);

    Aeq = zeros(n, num_vars);
    beq = ones(n, 1);
    for j = 1:n
        Aeq(j, (j - 1) * m + (1:m)) = 1;
    end

    A = zeros(m, num_vars);
    for i = 1:m
        for j = 1:n
            A(i, (j - 1) * m + i) = r(i, j);
        end
    end

    options = optimoptions('intlinprog', 'Display', 'off');
    [x, ~, flag] = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub, options);

    if flag <= 0
        warning('No feasible ILP solution.');
        x_matrix = zeros(m, n);
    else
        x_matrix = reshape(round(x), [m, n]);
    end
end

function x_matrix = greedy_gap_solver(m, n, c, r, b)
    x_matrix = zeros(m, n);
    remaining = b;

    for j = 1:n
        best = inf;
        chosen = -1;
        for i = 1:m
            if r(i, j) <= remaining(i) && c(i, j) < best
                best = c(i, j);
                chosen = i;
            end
        end
        if chosen > 0
            x_matrix(chosen, j) = 1;
            remaining(chosen) = remaining(chosen) - r(chosen, j);
        end
    end
end

