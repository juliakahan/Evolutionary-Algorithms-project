import pandas as pd
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
import cocoex
from coco_wrapper import CocoProblemWrapper

dimension = 20
suite = cocoex.Suite("bbob", "instances:1", f"function_indices:1-24 dimensions:{dimension}")

algorithms = {
    "PSO": PSO(),
    "CMAES": CMAES(),
    "DE": DE()
}

budget = dimension * 200

results = []

for problem in suite:
    print(f"Optimizing problem {problem.id} (fid={problem.id_function}, iid={problem.id_instance})")
    
    pymoo_problem = CocoProblemWrapper(problem)
    
    for algo_name, algorithm in algorithms.items():
        
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
df_results.to_csv("optimization_results.csv", index=False)

print("Results saved to 'optimization_results.csv'")
