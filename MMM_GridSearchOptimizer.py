from sklearn.model_selection import TimeSeriesSplit
from itertools import product

# ==========================================
# GRID SEARCH OPTIMIZER - НОВЫЙ КЛАСС
# ==========================================

class MMM_GridSearchOptimizer:
    """
    Класс для автоматического подбора оптимальных параметров Adstock и Saturation
    для всех медиа-каналов в Marketing Mix Model через Grid Search.
    """
    
    def __init__(self, cv_folds=3, scoring='r2', verbose=True):
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.verbose = verbose
        
        self.search_results = []
        self.best_params = {}
        self.best_score = -np.inf
        
        # Предустановки параметров по типам каналов
        self.channel_presets = {
            'paid_search': {'decay_range': (0.2, 0.6), 'alpha_range': (0.8, 1.5)},
            'social_media': {'decay_range': (0.1, 0.4), 'alpha_range': (0.5, 1.2)},
            'display': {'decay_range': (0.3, 0.7), 'alpha_range': (0.6, 1.3)},
            'video': {'decay_range': (0.4, 0.8), 'alpha_range': (0.7, 1.4)},
            'offline': {'decay_range': (0.5, 0.9), 'alpha_range': (0.4, 1.0)},
            'default': {'decay_range': (0.2, 0.7), 'alpha_range': (0.5, 1.5)}
        }
    
    def _detect_channel_type(self, channel_name):
        """Определение типа канала по названию."""
        channel_lower = channel_name.lower()
        
        if any(keyword in channel_lower for keyword in ['google', 'search', 'sem']):
            return 'paid_search'
        elif any(keyword in channel_lower for keyword in ['facebook', 'instagram', 'tiktok', 'social']):
            return 'social_media'
        elif any(keyword in channel_lower for keyword in ['display', 'banner', 'programmatic']):
            return 'display'
        elif any(keyword in channel_lower for keyword in ['youtube', 'video', 'tv']):
            return 'video'
        elif any(keyword in channel_lower for keyword in ['offline', 'radio', 'print', 'ooh']):
            return 'offline'
        else:
            return 'default'
    
    def _generate_parameter_grid(self, media_channels, X_media, decay_steps=5, alpha_steps=5, gamma_steps=3):
        """Генерация сетки параметров для поиска."""
        param_grid = {}
        
        for channel in media_channels:
            channel_type = self._detect_channel_type(channel)
            presets = self.channel_presets[channel_type]
            
            # Генерация decay параметров
            decay_min, decay_max = presets['decay_range']
            decay_values = np.linspace(decay_min, decay_max, decay_steps)
            
            # Генерация alpha параметров
            alpha_min, alpha_max = presets['alpha_range']
            alpha_values = np.linspace(alpha_min, alpha_max, alpha_steps)
            
            # Генерация gamma параметров (относительно медианы канала)
            channel_data = X_media[channel]
            median_spend = channel_data[channel_data > 0].median() if len(channel_data[channel_data > 0]) > 0 else 1.0
            
            gamma_values = [
                median_spend * 0.3,  # Низкая точка насыщения
                median_spend * 0.7,  # Средняя точка насыщения  
                median_spend * 1.2   # Высокая точка насыщения
            ]
            
            param_grid[channel] = {
                'decay': decay_values,
                'alpha': alpha_values,
                'gamma': gamma_values
            }
        
        return param_grid
    
    def _create_param_combinations(self, param_grid):
        """Создание всех возможных комбинаций параметров."""
        channels = list(param_grid.keys())
        
        channel_combinations = []
        for channel in channels:
            channel_params = param_grid[channel]
            combinations = list(product(
                channel_params['decay'],
                channel_params['alpha'], 
                channel_params['gamma']
            ))
            channel_combinations.append(combinations)
        
        all_combinations = list(product(*channel_combinations))
        
        param_combinations = []
        for combination in all_combinations:
            params = {}
            for i, channel in enumerate(channels):
                decay, alpha, gamma = combination[i]
                params[channel] = {
                    'decay': decay,
                    'alpha': alpha,
                    'gamma': gamma
                }
            param_combinations.append(params)
        
        return param_combinations
    
    def _evaluate_params(self, params, model_class, X, y, media_channels):
        """Оценка качества параметров через кросс-валидацию."""
        try:
            adstock_params = {ch: {'decay': params[ch]['decay']} for ch in media_channels}
            saturation_params = {ch: {'alpha': params[ch]['alpha'], 'gamma': params[ch]['gamma']} 
                                for ch in media_channels}
            
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = model_class(
                    adstock_params=adstock_params,
                    saturation_params=saturation_params,
                    regularization='Ridge',
                    alpha=0.01
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                if self.scoring == 'r2':
                    score = r2_score(y_val, y_pred)
                elif self.scoring == 'mape':
                    score = -mean_absolute_percentage_error(y_val, y_pred)
                elif self.scoring == 'mae':
                    score = -np.mean(np.abs(y_val - y_pred))
                else:
                    score = r2_score(y_val, y_pred)
                
                scores.append(score)
            
            return np.mean(scores), np.std(scores)
            
        except Exception:
            return -np.inf, np.inf
    
    def grid_search(self, model_class, X, y, media_channels, 
                   decay_steps=5, alpha_steps=5, gamma_steps=3, max_combinations=1000):
        """Основной метод Grid Search."""
        
        if self.verbose:
            print("🔍 Запуск Grid Search для оптимизации параметров MMM...")
        
        X_media = X[media_channels]
        param_grid = self._generate_parameter_grid(
            media_channels, X_media, decay_steps, alpha_steps, gamma_steps
        )
        
        param_combinations = self._create_param_combinations(param_grid)
        
        if len(param_combinations) > max_combinations:
            np.random.seed(42)
            selected_indices = np.random.choice(len(param_combinations), max_combinations, replace=False)
            param_combinations = [param_combinations[i] for i in selected_indices]
        
        best_score = -np.inf
        best_params = None
        
        for i, params in enumerate(param_combinations):
            mean_score, std_score = self._evaluate_params(
                params, model_class, X, y, media_channels
            )
            
            result = {
                'params': params.copy(),
                'mean_score': mean_score,
                'std_score': std_score,
                'iteration': i
            }
            self.search_results.append(result)
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()
        
        self.best_score = best_score
        self.best_params = best_params
        
        return self.best_params, self.best_score
    
    def plot_search_progress(self):
        """Создание графика прогресса поиска."""
        if not self.search_results:
            return None
        
        iterations = [r['iteration'] for r in self.search_results]
        scores = [r['mean_score'] for r in self.search_results]
        
        best_scores = []
        current_best = -np.inf
        for score in scores:
            if score > current_best:
                current_best = score
            best_scores.append(current_best)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=iterations, y=scores, mode='markers',
            name='Все результаты', marker=dict(color='lightblue', size=4), opacity=0.6
        ))
        
        fig.add_trace(go.Scatter(
            x=iterations, y=best_scores, mode='lines',
            name='Лучший результат', line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f"Прогресс Grid Search (метрика: {self.scoring})",
            xaxis_title="Итерация", yaxis_title=f"Значение {self.scoring}",
            height=400, template="plotly_white"
        )
        
        return fig

# ==========================================
# МОДИФИКАЦИЯ КЛАССА MarketingMixModel
# ==========================================

def add_grid_search_method():
    """Добавление метода Grid Search в класс MarketingMixModel."""
    
    def auto_optimize_parameters(self, X, y, media_channels, 
                                decay_steps=4, alpha_steps=4, gamma_steps=3,
                                cv_folds=3, scoring='r2', max_combinations=500):
        """Автоматическая оптимизация параметров."""
        
        optimizer = MMM_GridSearchOptimizer(
            cv_folds=cv_folds, scoring=scoring, verbose=True
        )
        
        best_params, best_score = optimizer.grid_search(
            model_class=self.__class__, X=X, y=y, media_channels=media_channels,
            decay_steps=decay_steps, alpha_steps=alpha_steps, gamma_steps=gamma_steps,
            max_combinations=max_combinations
        )
        
        if best_params:
            self.adstock_params = {ch: {'decay': best_params[ch]['decay']} 
                                 for ch in media_channels}
            self.saturation_params = {ch: {'alpha': best_params[ch]['alpha'], 
                                         'gamma': best_params[ch]['gamma']} 
                                    for ch in media_channels}
        
        return best_params, best_score, optimizer
    
    return auto_optimize_parameters

# Добавляем метод к классу
MarketingMixModel.auto_optimize_parameters = add_grid_search_method()

# ==========================================
# ОБНОВЛЕНИЕ STREAMLIT APP
# ==========================================

# В классе MMM_App добавьте этот метод:

def _interpret_parameters(self, decay, alpha):
    """Интерпретация параметров для пользователя."""
    if decay < 0.3:
        decay_interp = "Быстрое затухание"
    elif decay < 0.6:
        decay_interp = "Среднее затухание"
    else:
        decay_interp = "Медленное затухание"
    
    if alpha < 0.8:
        alpha_interp = "Быстрое насыщение"
    elif alpha < 1.2:
        alpha_interp = "Умеренное насыщение"
    else:
        alpha_interp = "Медленное насыщение"
    
    return f"{decay_interp}, {alpha_interp}"

# ==========================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ==========================================

"""
Теперь вы можете использовать Grid Search следующим образом:

# В Streamlit приложении пользователь выбирает режим поиска и запускает:

if st.button("🚀 Запустить автоматический подбор"):
    # Создание модели
    model = MarketingMixModel()
    
    # Автоматическая оптимизация
    best_params, best_score, optimizer = model.auto_optimize_parameters(
        X=X_train, 
        y=y_train,
        media_channels=['facebook_spend', 'google_spend', 'tiktok_spend'],
        decay_steps=3,  # Быстрый поиск
        alpha_steps=3,
        gamma_steps=3,
        cv_folds=3,
        scoring='r2',
        max_combinations=200
    )
    
    # Применение найденных параметров и обучение
    model.fit(X_train, y_train)
    
    # Сохранение результатов
    st.session_state.grid_search_results = {
        'best_params': best_params,
        'best_score': best_score,
        'optimizer': optimizer
    }

# Параметры автоматически применяются к модели!
"""
