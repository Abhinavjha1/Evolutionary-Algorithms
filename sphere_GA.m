clc; clear;

%% Parameters
nVar = 4;                    % Number of variables
varMin = -10; varMax = 10;   % Variable bounds
nBits = 16;                  % Bits per variable
popSize = 50;               % Population size
nGen = 100;                 % Number of generations
pCrossover = 0.8;           % Crossover probability
pMutation = 0.01;           % Mutation probability

chromLength = nBits * nVar; % Total chromosome length

%% Fitness function: Sphere function
fitnessFunc = @(x) sum(x.^2);

%% Manual binary-to-decimal decoder
decode = @(chrom) binaryToReal(chrom, nBits, nVar, varMin, varMax);

%% Initialize Population
population = randi([0, 1], popSize, chromLength);
bestFitHistory = zeros(nGen, 1);

for gen = 1:nGen
    % Decode population
    decodedPop = zeros(popSize, nVar);
    fitness = zeros(popSize, 1);
    for i = 1:popSize
        decodedPop(i, :) = decode(population(i, :));
        fitness(i) = fitnessFunc(decodedPop(i, :));
    end

    % Selection (roulette wheel)
    invFit = 1 ./ (1 + fitness); % Minimize
    prob = invFit / sum(invFit);
    cumProb = cumsum(prob);
    
    newPop = zeros(size(population));
    for i = 1:2:popSize
        % Parent selection
        p1 = population(find(rand <= cumProb, 1, 'first'), :);
        p2 = population(find(rand <= cumProb, 1, 'first'), :);
        
        % Crossover
        if rand < pCrossover
            point = randi([1, chromLength-1]);
            child1 = [p1(1:point), p2(point+1:end)];
            child2 = [p2(1:point), p1(point+1:end)];
        else
            child1 = p1;
            child2 = p2;
        end
        
        % Mutation
        for j = 1:chromLength
            if rand < pMutation
                child1(j) = 1 - child1(j);
            end
            if rand < pMutation
                child2(j) = 1 - child2(j);
            end
        end
        
        newPop(i, :) = child1;
        newPop(i+1, :) = child2;
    end
    
    % Replace old population
    population = newPop;
    
    % Save best fitness of generation
    bestFitHistory(gen) = min(fitness);
end

%% Final result
[~, bestIdx] = min(fitness);
bestChrom = population(bestIdx, :);
bestX = decode(bestChrom);
bestVal = fitnessFunc(bestX);

fprintf('Best Solution Found:\n');
disp(bestX);
fprintf('Minimum Value of Sphere Function: %.6f\n', bestVal);

%% Plot
figure;
plot(bestFitHistory, 'LineWidth', 2);
xlabel('Generation');
ylabel('Best Fitness');
title('Binary-Coded GA: Sphere Function Minimization');
grid on;

%% --- Binary to Real Function ---
function realVals = binaryToReal(chrom, nBits, nVar, minVal, maxVal)
    realVals = zeros(1, nVar);
    for k = 1:nVar
        idx_start = (k-1)*nBits + 1;
        idx_end = k*nBits;
        bits = chrom(idx_start:idx_end);
        val = 0;
        for b = 1:nBits
            val = val + bits(b) * 2^(nBits - b);
        end
        realVals(k) = minVal + (maxVal - minVal) * val / (2^nBits - 1);
    end
end
