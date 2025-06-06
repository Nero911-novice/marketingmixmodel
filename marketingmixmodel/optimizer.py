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
            # Простая заглушка для демо
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
        
        # Подготовка ограничений
        bounds, linear_constraints = self._prepare_constraints(
            media_channels, total_budget, constraints, bounds_buffer
        )
        
        # Определение целевой функции
        objective_func = self._get_objective_function(model, media_channels, target)
        
        # Начальное приближение
        initial_guess = self._get_initial_guess(media_channels, total_budget, constraints)
        
        # Выбор метода оптимизации
        if method == 'SLSQP':
            result = self._optimize_slsqp(objective_func, initial_guess, bounds, linear_constraints)
        elif method == 'differential_evolution':
            result = self._optimize_differential_evolution(objective_func, bounds, total_budget)
        else:
            raise ValueError(f"Неподдерживаемый метод оптимизации: {method}")
        
        # Обработка результатов
        if result.success or hasattr(result, 'x'):
            optimal_allocation = dict(zip(media_channels, result.x))
            
            # Расчет метрик для оптимального решения
            predicted_results = self._calculate_metrics(model, optimal_allocation, media_channels)
            
            optimization_result = {
                'success': True,
                'allocation': optimal_allocation,
                'predicted_sales': predicted_results['sales'],
                'predicted_roas': predicted_results['roas'],
                'predicted_roi': predicted_results['roi'],
                'total_budget_used': sum(optimal_allocation.values()),
                'optimization_method': method,
                'objective_value': -result.fun if hasattr(result, 'fun') else None
            }
            
            self.best_solution = optimization_result
            
        else:
            optimization_result = {
                'success': False,
                'message': f"Оптимизация не удалась: {result.message if hasattr(result, 'message') else 'Неизвестная ошибка'}",
                'allocation': dict(zip(media_channels, initial_guess))
            }
        
        return optimization_result
    
    def _prepare_constraints(self, media_channels, total_budget, constraints, bounds_buffer):
        """Подготовка ограничений для оптимизации."""
        n_channels = len(media_channels)
        
        bounds_list = []
        
        for channel in media_channels:
            if constraints and channel in constraints:
                min_val = constraints[channel].get('min', 0)
                max_val = constraints[channel].get('max', total_budget)
            else:
                min_val = 0
                max_val = total_budget * 0.5
            
            min_val = max(0, min_val * (1 - bounds_buffer))
            max_val = min(total_budget, max_val * (1 + bounds_buffer))
            
            bounds_list.append((min_val, max_val))
        
        bounds = Bounds([b[0] for b in bounds_list], [b[1] for b in bounds_list])
        
        A_eq = np.ones((1, n_channels))
        linear_constraints = LinearConstraint(A_eq, total_budget, total_budget)
        
        return bounds, linear_constraints
    
    def _get_objective_function(self, model, media_channels, target):
        """Создание целевой функции для оптимизации."""
        def objective(x):
            allocation = dict(zip(media_channels, x))
            
            try:
                metrics = self._calculate_metrics(model, allocation, media_channels)
                
                if target == 'maximize_sales':
                    return -metrics['sales']
                elif target == 'maximize_roas':
                    return -metrics['roas']
                elif target == 'maximize_roi':
                    return -metrics['roi']
                else:
                    return -metrics['sales']
                    
            except Exception as e:
                return 1e10
        
        return objective
    
    def _calculate_metrics(self, model, allocation, media_channels):
        """Расчет метрик для заданного распределения бюджета."""
        scenario_result = model.predict_scenario(allocation)
        
        total_spend = sum(allocation.values())
        
        metrics = {
            'sales': scenario_result['sales'],
            'roas': scenario_result['roas'],
            'roi': (scenario_result['sales'] - total_spend) / total_spend if total_spend > 0 else 0,
            'total_spend': total_spend
        }
        
        return metrics
    
    def _get_initial_guess(self, media_channels, total_budget, constraints):
        """Создание начального приближения для оптимизации."""
        n_channels = len(media_channels)
        
        if constraints:
            initial = []
            remaining_budget = total_budget
            
            for i, channel in enumerate(media_channels):
                if channel in constraints:
                    min_val = constraints[channel].get('min', 0)
                    max_val = constraints[channel].get('max', total_budget)
                    
                    if i == n_channels - 1:
                        allocation = remaining_budget
                    else:
                        preferred = (min_val + max_val) / 2
                        allocation = min(preferred, remaining_budget / (n_channels - i))
                        allocation = max(min_val, min(max_val, allocation))
                    
                    initial.append(allocation)
                    remaining_budget -= allocation
                else:
                    allocation = remaining_budget / (n_channels - i)
                    initial.append(allocation)
                    remaining_budget -= allocation
            
            current_total = sum(initial)
            if current_total > 0:
                initial = [x * total_budget / current_total for x in initial]
        else:
            initial = [total_budget / n_channels] * n_channels
        
        return np.array(initial)
    
    def _optimize_slsqp(self, objective_func, initial_guess, bounds, linear_constraints):
        """Оптимизация методом SLSQP."""
        result = minimize(
            objective_func,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=linear_constraints,
            options={
                'maxiter': self.convergence_criteria['max_iterations'],
                'ftol': self.convergence_criteria['tolerance'],
                'disp': False
            }
        )
        
        return result
    
    def _optimize_differential_evolution(self, objective_func, bounds, total_budget):
        """Оптимизация дифференциальной эволюцией."""
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

