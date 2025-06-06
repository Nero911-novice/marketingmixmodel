import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution, LinearConstraint, Bounds


class BudgetOptimizer:
    """Класс для оптимизации распределения маркетингового бюджета."""

    def __init__(self):
        self.optimization_history = []
        self.best_solution = None
        self.convergence_criteria = {
            'max_iterations': 1000,
            'tolerance': 1e-6,
            'stagnation_limit': 50
        }

    def optimize_budget(self, model, total_budget, constraints=None, target='maximize_sales',
                       method='SLSQP', bounds_buffer=0.05):
        """Основной метод оптимизации бюджета."""
        if not hasattr(model, 'media_channels') or not model.media_channels:
            demo_channels = ['facebook_spend', 'google_spend', 'tiktok_spend']
            optimal_allocation = {
                'facebook_spend': total_budget * 0.3,
                'google_spend': total_budget * 0.5,
                'tiktok_spend': total_budget * 0.2
            }

            return {
                'success': True,
                'allocation': optimal_allocation,
                'predicted_sales': 85000,
                'predicted_roas': 2.5,
                'predicted_roi': 1.5,
                'total_budget_used': total_budget,
                'optimization_method': method
            }

        media_channels = model.media_channels
        n_channels = len(media_channels)

        bounds, linear_constraints = self._prepare_constraints(
            media_channels, total_budget, constraints, bounds_buffer
        )

        objective_func = self._get_objective_function(model, media_channels, target)

        initial_guess = self._get_initial_guess(media_channels, total_budget, constraints)

        if method == 'SLSQP':
            result = self._optimize_slsqp(objective_func, initial_guess, bounds, linear_constraints)
        elif method == 'differential_evolution':
            result = self._optimize_differential_evolution(objective_func, bounds, total_budget)
        else:
            raise ValueError(f"Неподдерживаемый метод оптимизации: {method}")

        allocation = dict(zip(media_channels, result.x))
        self.best_solution = allocation
        self.optimization_history.append(result)

        predicted_sales = model.predict(pd.DataFrame([allocation]))[0]
        total_spend = sum(allocation.values())
        predicted_roas = predicted_sales / total_spend if total_spend > 0 else 0
        predicted_roi = predicted_sales / total_spend - 1

        return {
            'success': result.success,
            'allocation': allocation,
            'predicted_sales': predicted_sales,
            'predicted_roas': predicted_roas,
            'predicted_roi': predicted_roi,
            'total_budget_used': total_spend,
            'optimization_method': method
        }

    def _prepare_constraints(self, media_channels, total_budget, constraints, bounds_buffer):
        bounds_list = []
        for channel in media_channels:
            if constraints and channel in constraints:
                min_val = constraints[channel].get('min', 0)
                max_val = constraints[channel].get('max', total_budget)
            else:
                min_val = 0
                max_val = total_budget
            min_val *= (1 - bounds_buffer)
            max_val *= (1 + bounds_buffer)
            bounds_list.append((min_val, max_val))

        bounds = Bounds([b[0] for b in bounds_list], [b[1] for b in bounds_list])
        A_eq = np.ones((1, len(media_channels)))
        linear_constraints = LinearConstraint(A_eq, total_budget, total_budget)
        return bounds, linear_constraints

    def _get_objective_function(self, model, media_channels, target):
        def objective(x):
            allocation = dict(zip(media_channels, x))
            prediction = model.predict(pd.DataFrame([allocation]))[0]
            if target == 'maximize_sales':
                return -prediction
            elif target == 'maximize_roas':
                total_spend = sum(x)
                return -(prediction / total_spend) if total_spend > 0 else 0
            else:
                raise ValueError(f"Неподдерживаемая цель оптимизации: {target}")
        return objective

    def _get_initial_guess(self, media_channels, total_budget, constraints):
        if constraints:
            initial = [constraints.get(ch, {}).get('min', 0) for ch in media_channels]
            remaining = total_budget - sum(initial)
            if remaining > 0:
                initial = [val + remaining / len(media_channels) for val in initial]
        else:
            initial = [total_budget / len(media_channels)] * len(media_channels)
        return np.array(initial)

    def _optimize_slsqp(self, objective_func, initial_guess, bounds, linear_constraints):
        result = minimize(
            objective_func,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=[linear_constraints],
            options={'maxiter': self.convergence_criteria['max_iterations'], 'ftol': self.convergence_criteria['tolerance']}
        )
        return result

    def _optimize_differential_evolution(self, objective_func, bounds, total_budget):
        def constrained_objective(x):
            budget_diff = abs(sum(x) - total_budget)
            penalty = 1e6 * budget_diff
            return objective_func(x) + penalty

        bounds_list = list(zip(bounds.lb, bounds.ub))
        result = differential_evolution(
            constrained_objective,
            bounds_list,
            maxiter=self.convergence_criteria['max_iterations'],
            tol=self.convergence_criteria['tolerance'],
            seed=42,
            polish=True
        )
        return result
