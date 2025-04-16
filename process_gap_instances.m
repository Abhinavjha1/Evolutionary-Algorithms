function process_gap_instances()
    results = [];
    labels = {};

    for file_idx = 1:12
        filepath = sprintf('./gap dataset files/gap%d.txt', file_idx);
        fid = fopen(filepath, 'r');
        if fid < 0
            error('Unable to open file: %s', filepath);
        end

        instance_count = fscanf(fid, '%d', 1);
        fprintf('\nProcessing File: %s\n', filepath(1:end-4));

        for inst = 1:instance_count
            m = fscanf(fid, '%d', 1);
            n = fscanf(fid, '%d', 1);

            cost = fscanf(fid, '%d', [n, m])';
            resource = fscanf(fid, '%d', [n, m])';
            capacity = fscanf(fid, '%d', [m, 1]);

            solution = solve_assignment_ilp(m, n, cost, resource, capacity);
            total_cost = sum(sum(cost .* solution));

            tag = sprintf('case%d-%d', m * 100 + n, inst);
            fprintf('%s  %d\n', tag, round(total_cost));

            results(end + 1) = round(total_cost);
            labels{end + 1} = sprintf('gap%d-%d', file_idx, inst);
        end

        fclose(fid);
    end

    % Visualize the collected data
    figure;
    plot(results, '-s', 'LineWidth', 2);
    title('Objective Values Across GAP Instances');
    xlabel('Instance Index');
    ylabel('Objective Function Value');
    xticks(1:length(labels));
    xticklabels(labels);
    xtickangle(45);
    grid on;
end

function solution_matrix = solve_assignment_ilp(m, n, cost_matrix, resource_matrix, limits)
    total_vars = m * n;
    objective = -reshape(cost_matrix, [total_vars, 1]);
    binary_vars = 1:total_vars;
    lower_bounds = zeros(total_vars, 1);
    upper_bounds = ones(total_vars, 1);

    % Each task is assigned to exactly one agent
    eq_matrix = zeros(n, total_vars);
    eq_rhs = ones(n, 1);
    for task = 1:n
        eq_matrix(task, (task - 1) * m + (1:m)) = 1;
    end

    % Resource constraints
    ineq_matrix = zeros(m, total_vars);
    for agent = 1:m
        for task = 1:n
            ineq_matrix(agent, (task - 1) * m + agent) = resource_matrix(agent, task);
        end
    end

    opts = optimoptions('intlinprog', 'Display', 'off');
    [sol, ~, flag] = intlinprog(objective, binary_vars, ineq_matrix, limits, eq_matrix, eq_rhs, lower_bounds, upper_bounds, opts);

    if flag <= 0
        warning('Infeasible solution encountered.');
        solution_matrix = zeros(m, n);
    else
        solution_matrix = reshape(round(sol), [m, n]);
    end
end
