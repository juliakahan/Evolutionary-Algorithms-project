import pandas as pd
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
import cocoex
from coco_wrapper import CocoProblemWrapper
from scipy.optimize import minimize as scipy_minimize
import numpy as np
from bipop_cmaes import run_bipop_cmaes

dimension = 20
suite = cocoex.Suite("bbob", "instances:1-50", f"function_indices:1-24 dimensions:{dimension}")

algorithms = {
    "PSO": PSO(),
    "CMAES": CMAES(),
    "DE": DE(),
    "L-BFGS-B": "scipy",
    "BIPOP-CMAES": "deap"
}

budget = dimension * 500

results = []

for problem in suite:
    print(f"Optimizing problem {problem.id} (fid={problem.id_function}, iid={problem.id_instance})")
    
    pymoo_problem = CocoProblemWrapper(problem)
    
    for algo_name, algorithm in algorithms.items():

        if algo_name == "L-BFGS-B":
            # Define the function for L-BFGS-B
            def obj_func(x):
                out = {}
                pymoo_problem._evaluate(x.reshape(1, -1), out)
                return out["F"][0]  # Return the scalar objective value
            
            # Retrieve bounds
            lower_bounds, upper_bounds = pymoo_problem.bounds()
            bounds = list(zip(lower_bounds, upper_bounds))  # Convert to list of tuples
            
            # Run L-BFGS-B
            res = scipy_minimize(
                fun=obj_func,
                x0=np.random.rand(dimension) * (np.array(upper_bounds) - np.array(lower_bounds)) + np.array(lower_bounds),  # Random initial point
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxfun': budget}
            )
            
            results.append({
                "Problem ID": problem.id,
                "Function ID": problem.id_function,
                "Instance ID": problem.id_instance,
                "Algorithm": algo_name,
                "Best Solution (X)": res.x.tolist() if res.success else None,
                "Best Function Value (F)": res.fun if res.success else None
            })

        elif algo_name == "BIPOP-CMAES":
            best_solution, best_value = run_bipop_cmaes(pymoo_problem, dimension, budget, verbose=False)
            results.append({
                "Problem ID": problem.id,
                "Function ID": problem.id_function,
                "Instance ID": problem.id_instance,
                "Algorithm": algo_name,
                "Best Solution (X)": best_solution,
                "Best Function Value (F)": best_value
            })


        else:
            res = minimize(pymoo_problem,
                        algorithm,
                        seed=1,
                        verbose=False,
                        termination=('n_eval', budget))

            results.append({
                "Problem ID": problem.id,
                "Function ID": problem.id_function,
                "Instance ID": problem.id_instance,
                "Algorithm": algo_name,
                "Best Solution (X)": res.X.tolist() if res.X is not None else None,
                "Best Function Value (F)": res.F[0] if res.F is not None else None
            })

df_results = pd.DataFrame(results)
df_results.to_csv("optimization_results2.csv", index=False)

print("Results saved to 'optimization_results2.csv'")
