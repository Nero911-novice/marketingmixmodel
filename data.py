import pandas as pd
import numpy as np


class DataProcessor:
    """Класс для обработки и подготовки данных для Marketing Mix Model."""

    def __init__(self):
        self.data_quality_checks = {}

    def generate_demo_data(self, n_periods=104, start_date='2023-01-01', frequency='W'):
        """Генерация демонстрационных данных для MMM."""
        date_range = pd.date_range(start=start_date, periods=n_periods, freq=frequency)

        np.random.seed(42)

        seasonal_annual = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_periods) / 52)
        seasonal_monthly = 1 + 0.15 * np.sin(2 * np.pi * np.arange(n_periods) / 4.33)
        trend = 1 + 0.002 * np.arange(n_periods)
        noise = np.random.normal(0, 0.1, n_periods)
        holiday_effect = np.random.choice([0, 0, 0, 0.3], n_periods, p=[0.85, 0.05, 0.05, 0.05])

        facebook_base = 45000 + 15000 * seasonal_annual + 8000 * np.random.normal(0, 1, n_periods)
        facebook_spend = np.maximum(facebook_base, 5000)

        google_base = 67000 + 20000 * seasonal_monthly + 12000 * np.random.normal(0, 1, n_periods)
        google_spend = np.maximum(google_base, 8000)

        tiktok_base = 15000 + 25000 * (np.arange(n_periods) / n_periods) + 10000 * np.random.normal(0, 1.5, n_periods)
        tiktok_spend = np.maximum(tiktok_base, 0)

        youtube_base = 32000 + 12000 * seasonal_annual + 8000 * np.random.normal(0, 1, n_periods)
        youtube_spend = np.maximum(youtube_base, 2000)

        offline_base = 25000 + 8000 * seasonal_annual + 5000 * np.random.normal(0, 0.8, n_periods)
        offline_spend = np.maximum(offline_base, 1000)

        facebook_impressions = facebook_spend * 50 + np.random.normal(0, facebook_spend * 5, n_periods)
        google_clicks = google_spend * 0.035 + np.random.normal(0, google_spend * 0.007, n_periods)

        base_sales = 50000 * trend * seasonal_monthly
        media_effect = (
            facebook_spend * 0.005 + google_spend * 0.007 + tiktok_spend * 0.006 +
            youtube_spend * 0.004 + offline_spend * 0.003
        )
        external_effect = promo_effect = 10000 * holiday_effect
        total_orders = base_sales + media_effect + external_effect + noise * 1000

        data = pd.DataFrame({
            'date': date_range,
            'orders': total_orders.astype(int),
            'facebook_spend': facebook_spend.astype(int),
            'google_spend': google_spend.astype(int),
            'tiktok_spend': tiktok_spend.astype(int),
            'youtube_spend': youtube_spend.astype(int),
            'offline_spend': offline_spend.astype(int),
            'facebook_impressions': facebook_impressions.astype(int),
            'google_clicks': google_clicks.astype(int),
            'promo_activity': holiday_effect.astype(int)
        })

        return data

    def validate_data(self, data, target_column):
        """Простая валидация данных."""
        validation_results = {}

        required_columns = [target_column, 'date']
        media_columns = [col for col in data.columns if col.endswith('_spend')]
        date_format_ok = True
        missing_counts = data.isnull().sum()
        duplicate_dates = data['date'].duplicated().sum()

        missing_required = [col for col in required_columns if col not in data.columns]
        validation_results['required_columns'] = {
            'description': 'Обязательные столбцы присутствуют',
            'status': len(missing_required) == 0,
            'details': missing_required
        }

        if 'date' in data.columns:
            try:
                pd.to_datetime(data['date'])
                date_format_ok = True
            except Exception:
                date_format_ok = False

        validation_results['date_format'] = {
            'description': 'Формат даты корректный',
            'status': date_format_ok
        }

        validation_results['missing_values'] = {
            'description': 'Отсутствуют пропущенные значения',
            'status': missing_counts.sum() == 0,
            'details': missing_counts[missing_counts > 0].to_dict()
        }

        validation_results['duplicate_dates'] = {
            'description': 'Отсутствуют дубликаты дат',
            'status': duplicate_dates == 0
        }

        passed_checks = sum(1 for check in validation_results.values() if check['status'])
        quality_score = passed_checks / len(validation_results) * 100

        validation_results['quality_score'] = {
            'description': 'Общая оценка качества данных',
            'status': quality_score >= 80,
            'details': quality_score
        }

        return validation_results

    def prepare_features(self, data, target_column):
        """Формирование матрицы признаков и целевой переменной."""
        df = data.copy()

        media_columns = [col for col in df.columns if col.endswith('_spend')]
        other_columns = [col for col in df.columns if col not in media_columns + [target_column, 'date']]
        all_features = media_columns + other_columns

        for col in all_features:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')

        if df[target_column].isnull().any():
            df[target_column] = df[target_column].fillna(df[target_column].median())

        X = df[all_features].copy()
        y = df[target_column].copy()

        return X, y

    def split_data(self, data, train_ratio=0.8, date_column='date'):
        """Разделение данных на обучающую и тестовую выборки по времени."""
        df = data.copy()
        df = df.sort_values(date_column)

        split_index = int(len(df) * train_ratio)

        train_data = df.iloc[:split_index].copy()
        test_data = df.iloc[split_index:].copy()

        return train_data, test_data
