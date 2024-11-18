import cocoex
from pflacco.classical_ela_features import *
from pflacco.sampling import create_initial_sample

features = []
dimension = 20
n_samples = 100 * dimension

function_classes = {
    1: "separable - separable",
    2: "separable - separable",
    11: "separable - separable",
    3: "separable - moderate",
    4: "separable - moderate",
    12: "separable - moderate",
    13: "separable - moderate",
    5: "separable - ill-conditioned",
    6: "separable - ill-conditioned",
    14: "separable - ill-conditioned",
    15: "separable - ill-conditioned",
    7: "separable - multi-modal",
    8: "separable - multi-modal",
    16: "separable - multi-modal",
    17: "separable - multi-modal",
    9: "separable - weakly-structured",
    10: "separable - weakly-structured",
    18: "separable - weakly-structured",
    19: "separable - weakly-structured",
    20: "moderate - moderate",
    21: "moderate - moderate",
    28: "moderate - moderate",
    22: "moderate - ill-conditioned",
    23: "moderate - ill-conditioned",
    29: "moderate - ill-conditioned",
    30: "moderate - ill-conditioned",
    24: "moderate - multi-modal",
    25: "moderate - multi-modal",
    31: "moderate - multi-modal",
    32: "moderate - multi-modal",
    26: "moderate - weakly-structured",
    27: "moderate - weakly-structured",
    33: "moderate - weakly-structured",
    34: "moderate - weakly-structured",
    35: "ill-conditioned - ill-conditioned",
    36: "ill-conditioned - ill-conditioned",
    41: "ill-conditioned - ill-conditioned",
    37: "ill-conditioned - multi-modal",
    38: "ill-conditioned - multi-modal",
    42: "ill-conditioned - multi-modal",
    43: "ill-conditioned - multi-modal",
    39: "ill-conditioned - weakly-structured",
    40: "ill-conditioned - weakly-structured",
    44: "ill-conditioned - weakly-structured",
    45: "ill-conditioned - weakly-structured",
    46: "multi-modal - multi-modal",
    47: "multi-modal - multi-modal",
    50: "multi-modal - multi-modal",
    48: "multi-modal - weakly structured",
    49: "multi-modal - weakly structured",
    51: "multi-modal - weakly structured",
    52: "multi-modal - weakly structured",
    53: "weakly structured - weakly structured",
    54: "weakly structured - weakly structured",
    55: "weakly structured - weakly structured"
}

# Get all 24 single-objective noiseless BBOB function in dimension 20 the first instance.
suite = cocoex.Suite("bbob", "instances:1-100", f"function_indices:1-55 dimensions:{dimension}")
for problem in suite:
    dim = problem.dimension
    fid = problem.id_function
    iid = problem.id_instance
    function_class = function_classes[fid]

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
    data = pd.DataFrame({'function_class': function_class,'int': int, 'lr2': lr2, 'max': max, 'eps_ratio': eps_ratio, 'disp': disp, 'nbc': nbc}, index = [0])
    features.append(data)

features = pd.concat(features).reset_index(drop = True)
features.to_csv('ela_features_with_classe_2.csv')