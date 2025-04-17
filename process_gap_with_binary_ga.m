function process_gap_with_binary_ga()
    totalFiles = 12;
    outputFile = 'gap_results_bcga_resource_max.csv';
    allResults = cell(totalFiles, 1);

    for fileIdx = 1:totalFiles
        fileName = sprintf('./gap dataset files/gap%d.txt', fileIdx);
        fid = fopen(fileName, 'r');
        if fid == -1
            error('Cannot open file: %s', fileName);
        end

        numInstances = fscanf(fid, '%d', 1);
        results = zeros(numInstances, 1);

        for instIdx = 1:numInstances
            m = fscanf(fid, '%d', 1);  % agents
            n = fscanf(fid, '%d', 1);  % tasks

            cost = fscanf(fid, '%d', [n, m])';
            resource = fscanf(fid, '%d', [n, m])';
            capacity = fscanf(fid, '%d', [m, 1]);

            fprintf('Processing gap%d - Instance %d\n', fileIdx, instIdx);
            assignment = solve_gap_ga_binary(m, n, cost, resource, capacity);
            totalResourceUsed = sum(sum(resource .* assignment));
            results(instIdx) = totalResourceUsed;
        end
        allResults{fileIdx} = results;
        fclose(fid);
    end

    % Save results to CSV
    write_results_to_csv(allResults, outputFile);
end

function write_results_to_csv(allResults, outputFile)
    maxCases = max(cellfun(@length, allResults));
    headers = [{'Case'}, arrayfun(@(x) sprintf('gap%d', x), 1:length(allResults), 'UniformOutput', false)];
    resultTable = cell(maxCases + 1, length(headers));
    resultTable(1,:) = headers;

    for i = 1:maxCases
        resultTable{i+1, 1} = sprintf('Case %d', i);
        for j = 1:length(allResults)
            if i <= length(allResults{j})
                resultTable{i+1, j+1} = allResults{j}(i);
            else
                resultTable{i+1, j+1} = '';
            end
        end
    end

    writecell(resultTable, outputFile);
    fprintf('Results saved to %s\n', outputFile);
end
