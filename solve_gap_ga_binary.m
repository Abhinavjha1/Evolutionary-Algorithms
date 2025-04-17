function [bestMatrix , bestFitness] = solve_gap_ga_binary(m, n, cost, resource, capacity)
    popSize = 100;
    maxGen = 200;
    crossoverRate = 0.8;
    mutationRate = 0.01;

    population = initialize_population(m, n, popSize);
    fitness = evaluate_population(population, m, n, resource, capacity);

    bestFitness = zeros(maxGen, 1);


    for gen = 1:maxGen
        selected = tournament_selection(population, fitness, 2);
        offspring = crossover_population(selected, crossoverRate, m, n);
        mutated = mutate_population(offspring, mutationRate);
        repaired = repair_population(mutated, m, n);

        fitnessOff = evaluate_population(repaired, m, n, resource, capacity);
        [population, fitness] = elitist_replacement(population, fitness, repaired, fitnessOff);

        bestFitness(gen) = max(fitness);

    end

    [~, bestIdx] = max(fitness);
    bestMatrix = reshape(population(bestIdx,:), m, n);
end

function pop = initialize_population(m, n, popSize)
    pop = zeros(popSize, m * n);
    for i = 1:popSize
        for j = 1:n
            idx = (j - 1) * m + randi(m);
            pop(i, idx) = 1;
        end
    end
end

function fitness = evaluate_population(pop, m, n, resource, capacity)
    fitness = zeros(size(pop,1),1);
    for i = 1:size(pop,1)
        x = reshape(pop(i,:), m, n);
        userSum = sum(x, 1);
        serverLoad = sum(resource .* x, 2);

        if all(userSum == 1) && all(serverLoad <= capacity)
            fitness(i) = sum(sum(resource .* x));  % Maximize resource usage
        else
            fitness(i) = sum(sum(resource .* x))-1e3;  % Penalty
        end
    end
end

function selected = tournament_selection(pop, fitness, k)
    popSize = size(pop, 1);
    selected = zeros(size(pop));
    for i = 1:popSize
        candidates = randi(popSize, [1, k]);
        [~, best] = max(fitness(candidates));
        selected(i,:) = pop(candidates(best),:);
    end
end

function offspring = crossover_population(parents, pC, m, n)
    offspring = zeros(size(parents));
    for i = 1:2:size(parents,1)-1
        if rand() < pC
            crossPoint = randi(m*n - 1);
            offspring(i,:) = [parents(i,1:crossPoint), parents(i+1,crossPoint+1:end)];
            offspring(i+1,:) = [parents(i+1,1:crossPoint), parents(i,crossPoint+1:end)];
        else
            offspring(i,:) = parents(i,:);
            offspring(i+1,:) = parents(i+1,:);
        end
    end
end

function mutated = mutate_population(pop, pM)
    mutated = pop;
    for i = 1:numel(pop)
        if rand() < pM
            mutated(i) = 1 - pop(i);
        end
    end
end

function repaired = repair_population(pop, m, n)
    repaired = pop;
    for i = 1:size(pop,1)
        for j = 1:n
            segment = (j-1)*m+1:j*m;
            gene = repaired(i,segment);
            if sum(gene) ~= 1
                gene(:) = 0;
                gene(randi(m)) = 1;
                repaired(i,segment) = gene;
            end
        end
    end
end

function [newPop, newFit] = elitist_replacement(oldPop, oldFit, newPop, newFit)
    [bestFit, idx] = max(oldFit);
    [~, worstIdx] = min(newFit);
    newPop(worstIdx,:) = oldPop(idx,:);
    newFit(worstIdx) = bestFit;
end
