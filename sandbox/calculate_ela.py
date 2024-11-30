import cocoex
from pflacco.classical_ela_features import *
from pflacco.sampling import create_initial_sample

features = []
dimension = 20
n_samples = 100 * dimension

# Function classes taken from here: https://numbbo.github.io/coco/testsuites/bbob
# 1 Separable Functions
# 2 Functions with low or moderate conditioning
# 3 Functions with high conditioning and unimodal
# 4 Multi-modal functions with adequate global structure
# 5 Multi-modal functions with weak global structure

function_classes = [1, 1, 1, 1, 1,
                    2, 2, 2, 2,
                    3, 3, 3, 3, 3,
                    4, 4, 4, 4, 4,
                    5, 5, 5, 5, 5]

# Get all 24 single-objective noiseless BBOB function in dimension 20 the first instance.
suite = cocoex.Suite("bbob", "instances:1-50", f"function_indices:1-24 dimensions:{dimension}")
for problem in suite:
    dim = problem.dimension
    fid = problem.id_function
    iid = problem.id_instance
    function_class = function_classes[fid-1]

    # Create sample
    X = create_initial_sample(dim, n = n_samples, lower_bound = -5, upper_bound = 5)
    y = X.apply(lambda x: problem(x), axis = 1)

    # Calculate ELA features
    ela_meta = calculate_ela_meta(X, y)
    nbc = calculate_nbc(X, y)
    disp = calculate_dispersion(X, y)
    ic = calculate_information_content(X, y, seed = 100)

    int = ela_meta['ela_meta.lin_simple.intercept']
    lr2 = ela_meta['ela_meta.lin_simple.adj_r2']
    max = ela_meta['ela_meta.lin_simple.coef.max']
    eps_ratio = ic['ic.eps_ratio']
    disp = disp['disp.ratio_mean_02']
    nbc = nbc['nbc.nb_fitness.cor']

    # Store results in pandas dataframe
    data = pd.DataFrame({'function_class': function_class,'function_id': fid, 'int': int, 'lr2': lr2, 'max': max, 'eps_ratio': eps_ratio, 'disp': disp, 'nbc': nbc}, index = [0])
    features.append(data)

features = pd.concat(features).reset_index(drop = True)
features.to_csv('ela_features_with_classes.csv')