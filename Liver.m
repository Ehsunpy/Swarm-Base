function liver_clustering()
    % Load dataset
    raw_data = readtable('E:RaianeshTakaomli\End\bupa.data', 'FileType', 'text');
    
    % Remove missing values
    raw_data = rmmissing(raw_data);
    
    % Extract features and labels
    data = table2array(raw_data(:, 1:6)); % Features
    labels = table2array(raw_data(:, 7)); % Labels
    
    % Normalize data using Z-score
    data = zscore(data);
    
    % DE parameters (as per the paper)
    option.nclusters = 2; % Number of clusters
    option.npop = 10; % Population size
    option.nvar = size(data, 2); % Number of features
    option.maxit = 300; % Maximum iterations
    option.beta = 0.5; % Mutation factor
    option.pr = 0.5; % Crossover probability
    
    % Number of trials (as per the paper)
    num_trials = 10;
    
    % Arrays to store purity results
    purity_de = zeros(num_trials, 1);
    purity_kmeans = zeros(num_trials, 1);
    
    for trial = 1:num_trials
        % DE clustering
        pop = initialize_population(option, data);
        best_solution = [];
        best_fitness = -Inf;
        
        for iter = 1:option.maxit
            new_pop = cell(option.npop, 1);
            for i = 1:option.npop
                % Mutation: DE/rand/1 strategy
                indices = randperm(option.npop, 3);
                while any(indices == i)
                    indices = randperm(option.npop, 3);
                end
                a = indices(1);
                b = indices(2);
                c = indices(3);
                mutant = pop{a} + option.beta * (pop{b} - pop{c});
                
                % Ensure mutant dimensions are valid
                if size(mutant, 1) ~= option.nclusters || size(mutant, 2) ~= option.nvar
                    mutant = pop{i}; % Reset mutant if dimensions are incorrect
                end

                % Crossover: Binomial crossover
                j_rand = randi(option.nvar);
                trial_vec = pop{i};
                for j = 1:option.nvar
                    if (j == j_rand) || (rand < option.pr)
                        trial_vec(:, j) = mutant(:, j);
                    end
                end

                % Ensure trial dimensions are valid
                if size(trial_vec, 1) ~= option.nclusters || size(trial_vec, 2) ~= option.nvar
                    trial_vec = pop{i}; % Reset trial if dimensions are incorrect
                end

                % Selection: Greedy selection
                fitness_trial = compute_purity(trial_vec, data, labels, option.nclusters);
                fitness_current = compute_purity(pop{i}, data, labels, option.nclusters);
                if fitness_trial > fitness_current
                    new_pop{i} = trial_vec;
                else
                    new_pop{i} = pop{i};
                end

                % Update best solution
                if fitness_trial > best_fitness
                    best_fitness = fitness_trial;
                    best_solution = trial_vec;
                end
            end
            pop = new_pop;
        end
        
        % Assign clusters for DE
        distances_de = pdist2(data, best_solution);
        [~, assigned_clusters_de] = min(distances_de, [], 2);
        
        % Compute purity for DE
        purity_de(trial) = compute_purity(best_solution, data, labels, option.nclusters);
        
        % K-means clustering (using default settings)
        [assigned_clusters_kmeans, centroids_kmeans] = kmeans(data, option.nclusters);
        
        % Compute purity for K-means
        purity_kmeans(trial) = compute_purity(centroids_kmeans, data, labels, option.nclusters);
    end
    
    % Compute average purities
    avg_purity_de = mean(purity_de);
    avg_purity_kmeans = mean(purity_kmeans);
    
    % Display results
    fprintf('Average Purity DE: %.4f\n', avg_purity_de);
    fprintf('Average Purity K-means: %.4f\n', avg_purity_kmeans);
    
    % Plot results
    figure;
    bar([avg_purity_de, avg_purity_kmeans], 'FaceColor', 'flat');
    xticks([1, 2]);
    xticklabels({'DE', 'K-means'});
    ylabel('Purity');
    title('Comparison of Purity: DE vs K-means');
    colormap([0 0 1; 1 0 0]); % Blue for DE, Red for K-means
end

function pop = initialize_population(option, data)
    [num_samples, ~] = size(data);
    pop = cell(option.npop, 1);
    for i = 1:option.npop
        % Randomly select 'nclusters' data points as initial centroids
        random_indices = randperm(num_samples, option.nclusters);
        pop{i} = data(random_indices, :);
    end
end

function purity = compute_purity(centroids, data, labels, num_clusters)
    % Calculate distances from data points to centroids
    distances = pdist2(data, centroids);
    
    % Assign each data point to the nearest centroid
    [~, assigned_clusters] = min(distances, [], 2);
    
    % Calculate purity
    purity = 0;
    for k = 1:num_clusters
        % Get labels of points in cluster k
        cluster_labels = labels(assigned_clusters == k);
        
        if isempty(cluster_labels)
            continue;
        end
        
        % Count occurrences of each label in the cluster
        class_counts = histcounts(cluster_labels, [0.5, 1.5, 2.5]);
        
        % Add the count of the most frequent label
        purity = purity + max(class_counts);
    end
    
    % Normalize purity by the total number of data points
    purity = purity / length(labels);
end   