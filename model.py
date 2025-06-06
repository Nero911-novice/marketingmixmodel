import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import warnings

warnings.filterwarnings('ignore')


class MarketingMixModel:
    """Основной класс для Marketing Mix Modeling."""

    def __init__(self, adstock_params=None, saturation_params=None,
                 regularization='Ridge', alpha=1.0, normalize_features=True):
        self.adstock_params = adstock_params or {}
        self.saturation_params = saturation_params or {}
        self.regularization = regularization
        self.alpha = alpha
        self.normalize_features = normalize_features

        self.scaler = StandardScaler() if normalize_features else None
        self.regressor = self._get_regressor()

        # Метаданные модели
        self.is_fitted = False
        self.feature_names = None
        self.media_channels = None
        self.target_name = None

    def _get_regressor(self):
        """Получить регрессор на основе типа регуляризации."""
        if self.regularization == 'Ridge':
            return Ridge(alpha=self.alpha, fit_intercept=True)
        elif self.regularization == 'Lasso':
            return Lasso(alpha=self.alpha, fit_intercept=True, max_iter=2000)
        elif self.regularization == 'ElasticNet':
            return ElasticNet(alpha=self.alpha, fit_intercept=True, max_iter=2000)
        else:
            raise ValueError(f"Неподдерживаемый тип регуляризации: {self.regularization}")

    def _apply_adstock(self, media_data, decay_rate=0.5):
        """Применить простую Adstock трансформацию."""
        adstocked = np.zeros_like(media_data, dtype=float)
        for i in range(len(media_data)):
            if i == 0:
                adstocked[i] = media_data[i]
            else:
                adstocked[i] = media_data[i] + decay_rate * adstocked[i-1]
        return adstocked

    def _apply_saturation(self, media_data, alpha=1.0, gamma=None):
        """Применить Hill Saturation трансформацию."""
        if gamma is None:
            gamma = np.median(media_data[media_data > 0]) if np.any(media_data > 0) else 1.0

        media_data = np.maximum(media_data, 1e-10)
        gamma = max(gamma, 1e-10)

        numerator = np.power(media_data, alpha)
        denominator = np.power(media_data, alpha) + np.power(gamma, alpha)
        denominator = np.maximum(denominator, 1e-10)

        return numerator / denominator

    def _apply_transformations(self, X_media, fit=False):
        """Применить трансформации adstock и saturation к медиа-переменным."""
        X_transformed = X_media.copy()

        for channel in X_media.columns:
            # Adstock
            if channel in self.adstock_params:
                decay_rate = self.adstock_params[channel].get('decay', 0.5)
            else:
                decay_rate = 0.5

            X_transformed[channel] = self._apply_adstock(X_media[channel].values, decay_rate)

            # Saturation
            if channel in self.saturation_params:
                alpha = self.saturation_params[channel].get('alpha', 1.0)
                gamma = self.saturation_params[channel].get('gamma', None)
            else:
                alpha = 1.0
                gamma = None

            X_transformed[channel] = self._apply_saturation(X_transformed[channel].values, alpha, gamma)

        return X_transformed

    def fit(self, X, y):
        """Обучить модель."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X должен быть pandas DataFrame")

        X_media = X[[col for col in X.columns if any(keyword in col.lower() for keyword in ['spend', 'cost', 'budget'])]]
        X_non_media = X[[col for col in X.columns if col not in X_media.columns]]

        X_media_transformed = self._apply_transformations(X_media, fit=True)
        X_final = pd.concat([X_media_transformed, X_non_media], axis=1)

        if self.normalize_features:
            X_final = pd.DataFrame(self.scaler.fit_transform(X_final), columns=X_final.columns)

        self.regressor.fit(X_final, y)
        self.is_fitted = True
        self.feature_names = X_final.columns.tolist()
        self.media_channels = X_media.columns.tolist()
        self.X_train = X_final
        self.y_train = y
        self.target_name = y.name if hasattr(y, 'name') else 'target'

    def predict(self, X):
        """Сделать прогноз."""
        if not self.is_fitted:
            raise ValueError("Модель не обучена")

        X_media = X[self.media_channels] if self.media_channels else pd.DataFrame()
        X_non_media = X[[col for col in X.columns if col not in self.media_channels]]

        X_media_transformed = self._apply_transformations(X_media, fit=False)
        X_final = pd.concat([X_media_transformed, X_non_media], axis=1)

        if self.normalize_features:
            X_final = pd.DataFrame(self.scaler.transform(X_final), columns=X_final.columns)

        return self.regressor.predict(X_final)

    def get_model_metrics(self, X_test, y_test):
        """Получить метрики модели."""
        y_pred = self.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

        return {
            'r2': r2,
            'mape': mape,
            'mae': mae,
            'rmse': rmse
        }

    def get_model_quality_assessment(self, X_test, y_test):
        """Оценить качество модели и вернуть статус."""
        metrics = self.get_model_metrics(X_test, y_test)
        r2 = metrics['r2']
        mape = metrics['mape'] * 100

        if r2 > 0.8 and mape < 10:
            status = "🟢 Модель работает отлично!"
        elif r2 > 0.6 and mape < 15:
            status = "🟡 Модель работает хорошо"
        elif r2 > 0.4 and mape < 20:
            status = "🟠 Модель работает удовлетворительно"
        else:
            status = "🔴 Модель работает плохо"

        quality_score = (r2 * 100 + (100 - mape)) / 2

        return {
            'r2': r2,
            'mape': mape,
            'quality_score': quality_score,
            'status': status,
        }

    def get_contributions(self):
        """Получить вклад каждого канала в продажи."""
        if not self.is_fitted:
            raise ValueError("Модель не обучена")

        contributions = {}
        intercept = self.regressor.intercept_
        contributions['Base'] = intercept

        for i, channel in enumerate(self.media_channels):
            coef = self.regressor.coef_[i]
            spend = self.X_train[channel].sum()
            contributions[channel] = coef * spend

        return contributions

    def generate_roas_table(self):
        """Создание таблицы ROAS по каждому каналу."""
        if not self.is_fitted:
            raise ValueError("Модель не обучена")

        contributions = self.get_contributions()
        total_contribution = sum(contributions.values()) - contributions.get('Base', 0)
        total_sales = self.y_train.sum()

        roas_data = []
        for channel in self.media_channels:
            spend = self.X_train[channel].sum()
            contribution = contributions[channel]
            roas_val = contribution / spend if spend > 0 else 0

            roas_data.append({
                'Channel': channel.replace('_spend', '').replace('_', ' ').title(),
                'ROAS': roas_val,
                'Total_Spend': round(spend, 0),
                'Total_Contribution': round(contribution, 0)
            })

        return pd.DataFrame(roas_data)

    def demo_channel_contributions(self):
        """Сгенерировать демо-данные по вкладу каналов."""
        demo_data = []
        for channel in ['facebook_spend', 'google_spend', 'tiktok_spend']:
            spend = 1000000 * np.random.rand()
            roas_val = np.random.uniform(1.5, 3.0)
            contribution = spend * roas_val

            demo_data.append({
                'Channel': channel.replace('_spend', '').replace('_', ' ').title(),
                'ROAS': roas_val,
                'Total_Spend': round(spend, 0),
                'Total_Contribution': round(contribution, 0)
            })

        return pd.DataFrame(demo_data)

    def predict_scenario(self, scenario_budget, seasonality_factor=1.0, competition_factor=1.0):
        """Предсказать результаты для заданного сценария бюджета."""
        if not self.is_fitted:
            raise ValueError("Модель не обучена")

        scenario_data = pd.DataFrame()
        for feature in self.feature_names:
            if feature in scenario_budget:
                scenario_data[feature] = [scenario_budget[feature]]
            else:
                scenario_data[feature] = [self.X_train[feature].mean()]

        predicted_sales = self.predict(scenario_data)[0]
        predicted_sales *= seasonality_factor * competition_factor

        total_spend = sum(scenario_budget.values())
        predicted_roas = predicted_sales / total_spend if total_spend > 0 else 0

        return {
            'sales': predicted_sales,
            'roas': predicted_roas,
            'total_spend': total_spend
        }
