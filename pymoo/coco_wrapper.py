from pymoo.core.problem import Problem

class CocoProblemWrapper(Problem):
    def __init__(self, coco_problem):
        super().__init__(n_var=coco_problem.dimension, 
                         n_obj=1, 
                         n_constr=0, 
                         xl=coco_problem.lower_bounds, 
                         xu=coco_problem.upper_bounds)
        self.coco_problem = coco_problem

    def _evaluate(self, x, out, *args, **kwargs):
        # COCO problems expect a single point; handle batch evaluation
        f = [self.coco_problem(xi) for xi in x]
        out["F"] = f
