import pandas as pd
import numpy as np


class DataProcessor:
    """Класс для обработки и подготовки данных для Marketing Mix Model."""
    
    def __init__(self):
        self.data_quality_checks = {}
    
    def generate_demo_data(self, n_periods=104, start_date='2023-01-01', frequency='W'):
        """Генерация демонстрационных данных для MMM."""
        # Создание временного индекса
        date_range = pd.date_range(start=start_date, periods=n_periods, freq=frequency)
        
        # Установка seed для воспроизводимости
        np.random.seed(42)
        
        # Создание базовых паттернов
        seasonal_annual = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_periods) / 52)
        seasonal_monthly = 1 + 0.15 * np.sin(2 * np.pi * np.arange(n_periods) / 4.33)
        trend = 1 + 0.002 * np.arange(n_periods)
        noise = np.random.normal(0, 0.1, n_periods)
        holiday_effect = np.random.choice([0, 0, 0, 0.3], n_periods, p=[0.85, 0.05, 0.05, 0.05])
        
        # Генерация медиа-каналов
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
        
        # Генерация медиа-показателей
        facebook_impressions = facebook_spend * 50 + np.random.normal(0, facebook_spend * 5, n_periods)
        google_clicks = google_spend * 0.035 + np.random.normal(0, google_spend * 0.007, n_periods)
        
        # Внешние факторы
        promo_activity = np.random.choice([0, 1], n_periods, p=[0.75, 0.25])
        competitor_activity = 0.8 + 0.4 * np.random.beta(2, 2, n_periods)
        
        # Генерация целевой переменной (заказы)
        def apply_adstock_saturation(media, decay=0.5, alpha=1.0, gamma_factor=0.3):
            # Adstock
            adstocked = np.zeros_like(media)
            for i in range(len(media)):
                if i == 0:
                    adstocked[i] = media[i]
                else:
                    adstocked[i] = media[i] + decay * adstocked[i-1]
            
            # Saturation
            gamma = np.mean(adstocked) * gamma_factor
            saturated = np.power(adstocked, alpha) / (np.power(adstocked, alpha) + np.power(gamma, alpha))
            return saturated
        
        # Базовая линия
        base_orders = 8000 * trend * seasonal_annual
        
        # Эффекты медиа
        facebook_effect = apply_adstock_saturation(facebook_spend, 0.6, 0.8, 0.4) * 0.15
        google_effect = apply_adstock_saturation(google_spend, 0.4, 1.2, 0.3) * 0.12
        tiktok_effect = apply_adstock_saturation(tiktok_spend, 0.3, 1.5, 0.5) * 0.08
        youtube_effect = apply_adstock_saturation(youtube_spend, 0.7, 0.9, 0.35) * 0.10
        offline_effect = apply_adstock_saturation(offline_spend, 0.8, 0.6, 0.6) * 0.06
        
        # Эффекты внешних факторов
        promo_effect = promo_activity * 1500
        competitor_effect = (1 - competitor_activity) * 1000
        holiday_orders = holiday_effect * 2000
        
        # Итоговые заказы
        total_orders = (base_orders + 
                       facebook_effect + google_effect + tiktok_effect + 
                       youtube_effect + offline_effect +
                       promo_effect + competitor_effect + 
                       holiday_orders + noise * 500)
        
        total_orders = np.maximum(total_orders, 1000)
        
        # Создание DataFrame
        demo_data = pd.DataFrame({
            'date': date_range,
            'orders': total_orders.astype(int),
            
            # Медиа-расходы
            'facebook_spend': facebook_spend.astype(int),
            'google_spend': google_spend.astype(int),
            'tiktok_spend': tiktok_spend.astype(int),
            'youtube_spend': youtube_spend.astype(int),
            'offline_spend': offline_spend.astype(int),
            
            # Медиа-показатели
            'facebook_impressions': facebook_impressions.astype(int),
            'google_clicks': google_clicks.astype(int),
            
            # Внешние факторы
            'promo_activity': promo_activity,
            'competitor_activity': competitor_activity.round(2),
            'holiday_effect': holiday_effect,
            
            # Дополнительные переменные
            'seasonal_index': seasonal_annual.round(2),
            'trend_index': trend.round(2)
        })
        
        return demo_data
    
    def validate_data(self, data):
        """Валидация данных для MMM."""
        validation_results = {}
        
        # 1. Проверка обязательных столбцов
        required_columns = ['date']
        missing_required = [col for col in required_columns if col not in data.columns]
        
        validation_results['required_columns'] = {
            'status': len(missing_required) == 0,
            'message': f"Отсутствуют столбцы: {missing_required}" if missing_required else "Все обязательные столбцы присутствуют"
        }
        
        # 2. Проверка формата даты
        try:
            pd.to_datetime(data['date'])
            date_format_ok = True
            date_message = "Формат даты корректный"
        except:
            date_format_ok = False
            date_message = "Некорректный формат даты"
        
        validation_results['date_format'] = {
            'status': date_format_ok,
            'message': date_message
        }
        
        # 3. Проверка пропущенных значений
        missing_counts = data.isnull().sum()
        
        validation_results['missing_values'] = {
            'status': missing_counts.sum() == 0,
            'message': f"Пропусков: {missing_counts.sum()}" if missing_counts.sum() > 0 else "Пропуски отсутствуют"
        }
        
        # 4. Проверка дубликатов дат
        duplicate_dates = data['date'].duplicated().sum()
        validation_results['duplicate_dates'] = {
            'status': duplicate_dates == 0,
            'message': f"Дубликатов дат: {duplicate_dates}" if duplicate_dates > 0 else "Дубликаты отсутствуют"
        }
        
        # 5. Общая оценка
        passed_checks = sum(1 for check in validation_results.values() if check['status'])
        total_checks = len(validation_results)
        quality_score = passed_checks / total_checks * 100
        
        validation_results['overall_quality'] = {
            'status': quality_score >= 80,
            'message': f"Качество данных: {quality_score:.1f}%",
            'score': quality_score
        }
        
        return validation_results
    
    def prepare_model_data(self, data, target_column, media_columns, external_columns=None, control_columns=None):
        """Подготовка данных для обучения MMM модели."""
        df = data.copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        # Формирование списка всех признаков
        all_features = media_columns.copy()
        
        if external_columns:
            all_features.extend(external_columns)
        
        if control_columns:
            all_features.extend(control_columns)
        
        # Проверка наличия столбцов
        missing_columns = [col for col in all_features + [target_column] if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют столбцы: {missing_columns}")
        
        # Обработка пропущенных значений
        for col in all_features:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
        
        if df[target_column].isnull().any():
            df[target_column] = df[target_column].fillna(df[target_column].median())
        
        # Формирование матрицы признаков
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

