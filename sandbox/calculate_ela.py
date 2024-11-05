import cocoex
from pflacco.classical_ela_features import *
from pflacco.sampling import create_initial_sample

features = []
# Get all 24 single-objective noiseless BBOB function in dimension 2 and 3 for the first five instances.
suite = cocoex.Suite("bbob", "instances:1-5", "function_indices:1-24 dimensions:2,3")
print(suite)
for problem in suite:
    dim = problem.dimension
    fid = problem.id_function
    iid = problem.id_instance

    # Create sample
    X = create_initial_sample(dim, lower_bound = -5, upper_bound = 5)
    y = X.apply(lambda x: problem(x), axis = 1)

    # Calculate ELA features
    ela_meta = calculate_ela_meta(X, y)
    ela_distr = calculate_ela_distribution(X, y)
    nbc = calculate_nbc(X, y)
    disp = calculate_dispersion(X, y)
    ic = calculate_information_content(X, y, seed = 100)

    # Store results in pandas dataframe
    data = pd.DataFrame({**ic, **ela_meta, **ela_distr, **nbc, **disp, **{'fid': fid}, **{'dim': dim}, **{'iid': iid}}, index = [0])
    features.append(data)
    break

features = pd.concat(features).reset_index(drop = True)
print(features)