% Load Dataset
raw_data = readtable('E:RaianeshTakaomli\End\bupa.data', 'FileType', 'text');

% Check the number of columns in the dataset
num_columns = size(raw_data, 2);

% Remove rows with missing data
raw_data = rmmissing(raw_data);

% Extract features and labels
data = table2array(raw_data(:, 1:6)); % First 6 columns as features
labels = table2array(raw_data(:, 7)); % 7th column as labels

% Normalize data using Z-score (standardization)
data = zscore(data);

% Parameters for DE (based on the article)
num_clusters = 2; % Number of clusters
num_iterations = 300; % Number of iterations (as per the article)
population_size = 10; % Population size (as per the article)
crossover_prob = 0.5; % Crossover probability (as per the article)
scaling_factor = 0.5; % Mutation scaling factor (as per the article)

% Initialize Population for DE
[num_samples, num_features] = size(data);
population = cell(population_size, 1);
for i = 1:population_size
    % Randomly select cluster centers from the dataset
    random_indices = randperm(num_samples, num_clusters);
    population{i} = data(random_indices, :);
end

% Differential Evolution Algorithm
best_solution = [];
best_fitness = -Inf;

for iter = 1:num_iterations
    new_population = cell(population_size, 1);
    for i = 1:population_size
        % Mutation
        indices = randperm(population_size, 3);
        while any(indices == i)
            indices = randperm(population_size, 3);
        end
        mutant = population{indices(1)} + scaling_factor * (population{indices(2)} - population{indices(3)});
        
        % Ensure mutant dimensions are valid
        if size(mutant, 1) ~= num_clusters || size(mutant, 2) ~= num_features
            mutant = population{i}; % Reset mutant if dimensions are incorrect
        end

        % Crossover
        trial = population{i};
        for j = 1:num_clusters
            if rand < crossover_prob
                trial(j, :) = mutant(j, :);
            end
        end

        % Ensure trial dimensions are valid
        if size(trial, 1) ~= num_clusters || size(trial, 2) ~= num_features
            trial = population{i}; % Reset trial if dimensions are incorrect
        end

        % Selection
        trial_fitness = compute_purity(trial, data, labels, num_clusters);
        current_fitness = compute_purity(population{i}, data, labels, num_clusters);
        if trial_fitness > current_fitness
            new_population{i} = trial;
        else
            new_population{i} = population{i};
        end

        % Update Best Solution
        if trial_fitness > best_fitness
            best_fitness = trial_fitness;
            best_solution = trial;
        end
    end
    population = new_population;
end

% Ensure best_solution has correct dimensions
if size(best_solution, 1) ~= num_clusters || size(best_solution, 2) ~= num_features
    error('Best solution has incorrect dimensions.');
end

% Assign Clusters for DE
distances_de = pdist2(data, best_solution);
[~, assigned_clusters_de] = min(distances_de, [], 2);

% Compute Purity for DE
purity_de = compute_purity(best_solution, data, labels, num_clusters);

% K-Means Clustering with Replicates
[assigned_clusters_kmeans, centroids_kmeans] = kmeans(data, num_clusters, ...
    'MaxIter', num_iterations, 'Replicates', 10);  % Run K-means with multiple replicates

% Compute Purity for K-means
purity_kmeans = compute_purity(centroids_kmeans, data, labels, num_clusters);

% Display Results
disp('Result for DE:');
disp(['Purity: ', num2str(purity_de)]);

disp('Result for K-means:');
disp(['Purity: ', num2str(purity_kmeans)]);

% Plot Purity for DE and K-means
figure;
bar([purity_de, purity_kmeans], 'FaceColor', 'flat'); % Bar plot with custom colors
xticks([1, 2]);
xticklabels({'DE', 'K-means'});
ylabel('Purity');
title('Purity Comparison: DE vs K-means');
colormap([0 0 1; 1 0 0]); % Blue for DE, Red for K-means

% Function to compute purity
function purity = compute_purity(centroids, data, labels, num_clusters)
    distances = pdist2(data, centroids);
    [~, assigned_clusters] = min(distances, [], 2);
    purity = 0;
    for k = 1:num_clusters
        cluster_labels = labels(assigned_clusters == k);
        class_counts = histcounts(cluster_labels, [0.5, 1.5, 2.5]);
        purity = purity + max(class_counts);
    end
    purity = purity / length(labels);
end