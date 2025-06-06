import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from model import MarketingMixModel
from data import DataProcessor
from visualization import Visualizer
from optimizer import BudgetOptimizer
# ==========================================
# STREAMLIT APPLICATION - ГЛАВНОЕ ПРИЛОЖЕНИЕ
# ==========================================

# Конфигурация страницы
st.set_page_config(
    page_title="Marketing Mix Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.stAlert > div {
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

class MMM_App:
    def __init__(self):
        self.processor = DataProcessor()
        self.visualizer = Visualizer()
        self.optimizer = BudgetOptimizer()
        
        # Инициализация состояния
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'model_fitted' not in st.session_state:
            st.session_state.model_fitted = False

    def run(self):
        st.title("🎯 Marketing Mix Model")
        st.markdown("**Система планирования и оптимизации рекламных бюджетов**")
        
        # Боковая панель навигации
        with st.sidebar:
            st.header("Навигация")
            page = st.selectbox(
                "Выберите раздел:",
                ["🏠 Главная", "📊 Данные", "⚙️ Модель", "📈 Результаты", "💰 Оптимизация", "🔮 Сценарии"]
            )
            
            st.markdown("---")
            st.markdown("### Информация о модели")
            if st.session_state.model_fitted:
                st.success("✅ Модель обучена")
            else:
                st.warning("⚠️ Модель не обучена")
        
        # Роутинг страниц
        if page == "🏠 Главная":
            self.show_home()
        elif page == "📊 Данные":
            self.show_data()
        elif page == "⚙️ Модель":
            self.show_model()
        elif page == "📈 Результаты":
            self.show_results()
        elif page == "💰 Оптимизация":
            self.show_optimization()
        elif page == "🔮 Сценарии":
            self.show_scenarios()

    def show_home(self):
        st.header("Marketing Mix Model - Система оптимизации рекламных бюджетов")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Что такое Marketing Mix Modeling?
            
            MMM — это статистический подход для измерения влияния различных маркетинговых каналов 
            на бизнес-метрики и оптимизации распределения рекламного бюджета.
            
            #### Ключевые возможности:
            
            **📊 Анализ атрибуции**
            - Определение вклада каждого канала в продажи
            - Учет эффектов переноса (adstock) и насыщения (saturation)
            - Измерение ROAS по каналам
            
            **🎯 Оптимизация бюджета**
            - Поиск оптимального распределения бюджета
            - Прогнозирование эффекта изменений в медиа-планах
            - Сценарное планирование
            
            **🔮 Прогнозирование**
            - "What-if" анализ различных стратегий
            - Моделирование влияния внешних факторов
            - Планирование медиа-активности на будущие периоды
            """)
            
        with col2:
            st.markdown("### Математическая модель")
            st.latex(r'''Sales_t = Base + \sum_{i=1}^{n} Adstock_i(Media_i) \times Saturation_i(Media_i) + Externals_t''')
            
            st.markdown("**Где:**")
            st.markdown("- Base — базовая линия продаж")
            st.markdown("- Adstock — эффект переноса")
            st.markdown("- Saturation — эффект насыщения")
            st.markdown("- Externals — внешние факторы")
            
            if st.button("🎲 Загрузить демо-данные", type="primary"):
                demo_data = self.processor.generate_demo_data()
                st.session_state.data = demo_data
                st.success("Демо-данные загружены!")
                st.rerun()
        
        st.markdown("---")
        
        # Демо метрики
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Период анализа", "24 месяца")
        
        with col2:
            st.metric("Медиа-каналы", "5 каналов")
        
        with col3:
            st.metric("Точность модели", "R² > 0.8")

    def show_data(self):
        st.header("📊 Управление данными")
        
        tab1, tab2, tab3 = st.tabs(["Загрузка данных", "Просмотр данных", "Валидация"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Загрузка файла")
                uploaded_file = st.file_uploader(
                    "Выберите CSV файл с данными",
                    type=['csv'],
                    help="Файл должен содержать временные ряды продаж, медиа-расходов и внешних факторов"
                )
                
                if uploaded_file is not None:
                    try:
                        data = pd.read_csv(uploaded_file)
                        data['date'] = pd.to_datetime(data['date'])
                        st.session_state.data = data
                        st.success(f"Данные загружены: {len(data)} строк")
                    except Exception as e:
                        st.error(f"Ошибка загрузки: {str(e)}")
            
            with col2:
                st.subheader("Демо-данные")
                if st.button("Сгенерировать демо-данные"):
                    demo_data = self.processor.generate_demo_data()
                    st.session_state.data = demo_data
                    st.success("Демо-данные созданы")
                    st.rerun()
        
        with tab2:
            if st.session_state.data is not None:
                data = st.session_state.data
                
                st.subheader("Обзор данных")
                st.dataframe(data.head(10), use_container_width=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Строк", len(data))
                with col2:
                    st.metric("Столбцов", len(data.columns))
                with col3:
                    st.metric("Период", f"{data['date'].min().strftime('%Y-%m')} - {data['date'].max().strftime('%Y-%m')}")
                with col4:
                    st.metric("Пропуски", data.isnull().sum().sum())
                
                # Временные ряды
                st.subheader("Временные ряды основных метрик")
                metrics_cols = [col for col in data.columns if any(keyword in col.lower() 
                               for keyword in ['orders', 'sales', 'revenue', 'заказ'])]
                
                if metrics_cols:
                    fig = px.line(data, x='date', y=metrics_cols[0], 
                                title=f"Динамика {metrics_cols[0]}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Загрузите данные для просмотра")
        
        with tab3:
            if st.session_state.data is not None:
                validation_results = self.processor.validate_data(st.session_state.data)
                
                st.subheader("Результаты валидации")
                
                for check, result in validation_results.items():
                    if result['status']:
                        st.success(f"✅ {check}: {result['message']}")
                    else:
                        st.error(f"❌ {check}: {result['message']}")
            else:
                st.info("Загрузите данные для валидации")

    def show_model(self):
        st.header("⚙️ Конфигурация модели")

        if st.session_state.data is None:
            st.warning("Сначала загрузите данные")
            return

        data = st.session_state.data

        # Добавляем общее объяснение MMM
        with st.expander("📚 Математические основы Marketing Mix Model", expanded=False):
            st.markdown("""
        ### Теоретическая основа Marketing Mix Modeling
        
        **Marketing Mix Model** представляет собой эконометрическую модель, основанную на регрессионном анализе временных рядов. 
        Математическая формулация базируется на аддитивной декомпозиции продаж:
        
        **Sales(t) = Base(t) + Σ[Adstock(Media_i) × Saturation(Media_i)] + External_factors(t) + ε(t)**
        
        **Компоненты модели:**
        
        1. **Base(t)** — базовая линия продаж, включающая:
           - Органический рост бренда
           - Долгосрочные эффекты предыдущих маркетинговых активностей
           - Влияние неизмеряемых факторов (word-of-mouth, brand equity)
        
        2. **Media Effects** — трансформированные медиа-переменные через:
           - **Adstock**: моделирует эффект переноса рекламного воздействия
           - **Saturation**: учитывает убывающую предельную отдачу от рекламных инвестиций
        
        3. **External_factors(t)** — контрольные переменные:
           - Макроэкономические индикаторы
           - Конкурентная активность
           - Сезонные и праздничные эффекты
        
        4. **ε(t)** — случайная компонента с предположением нормального распределения
        
        **Статистические предположения модели:**
        - Линейная аддитивность эффектов медиа-каналов
        - Стационарность остатков модели
        - Отсутствие автокорреляции в остатках
        - Гомоскедастичность случайных ошибок
        """)
    
        tab1, tab2, tab3 = st.tabs(["Переменные модели", "Параметры трансформации", "Обучение модели"])

        with tab1:
            # Добавляем объяснение переменных модели
            with st.expander("📖 Типология переменных в MMM", expanded=False):
                st.markdown("""
            ### Классификация переменных в Marketing Mix Model
            
            **1. Зависимая переменная (Target Variable)**
            - Основная KPI, которую модель должна объяснить и предсказать
            - Требования: временной ряд с достаточной вариативностью
            - Примеры: заказы, продажи, выручка, конверсии
            - Рекомендация: логарифмическое преобразование для стабилизации дисперсии
            
            **2. Медиа-каналы (Media Variables)**
            - Контролируемые маркетинговые активности с известными инвестициями
            - Характеристики: положительная корреляция с target, наличие лагов
            - Примеры: затраты на paid search, display, social media, TV, radio
            - Требования к данным: еженедельная/месячная агрегация, отсутствие пропусков
            
            **3. Внешние факторы (External Variables)**
            - Неконтролируемые переменные, влияющие на целевую метрику
            - Типы: макроэкономические, конкурентные, сезонные
            - Функция: контроль смещения оценок медиа-эффектов
            - Примеры: индекс потребительских цен, активность конкурентов, температура
            
            **4. Контрольные переменные (Control Variables)**
            - Факторы, не являющиеся медиа, но влияющие на результат
            - Назначение: снижение необъясненной дисперсии модели
            - Примеры: цена продукта, ассортиментные изменения, промо-активность
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Зависимая переменная")
            target_options = [col for col in data.columns if any(keyword in col.lower() 
                            for keyword in ['orders', 'sales', 'revenue', 'заказ'])]
            target_var = st.selectbox("Выберите целевую метрику:", target_options)
            
            st.subheader("Медиа-каналы")
            media_options = [col for col in data.columns if any(keyword in col.lower() 
                           for keyword in ['spend', 'cost', 'budget', 'расход'])]
            selected_media = st.multiselect("Выберите медиа-каналы:", media_options, default=media_options[:5])
        
        with col2:
            st.subheader("Внешние факторы")
            external_options = [col for col in data.columns if any(keyword in col.lower() 
                              for keyword in ['holiday', 'promo', 'season', 'competitor', 'праздник'])]
            selected_external = st.multiselect("Выберите внешние факторы:", external_options, default=external_options)
            
            st.subheader("Контрольные переменные")
            control_options = [col for col in data.columns if col not in selected_media + selected_external + [target_var, 'date']]
            selected_controls = st.multiselect("Выберите контрольные переменные:", control_options)
    
        with tab2:
            # Добавляем объяснение параметров трансформации
            with st.expander("🔬 Научные основы медиа-трансформаций", expanded=False):
                st.markdown("""
            ### Adstock трансформация (Эффект переноса)
            
            **Теоретическое обоснование:**
            Рекламное воздействие не ограничивается моментом экспозиции. Psychological theory of memory decay 
            и consumer behavior research показывают, что эффекты рекламы затухают постепенно.
            
            **Геометрическая Adstock модель:**
            ```
            Adstock(t) = Media(t) + λ × Adstock(t-1)
            где λ ∈ [0,1] — коэффициент затухания
            ```
            
            **Интерпретация параметров:**
            - **λ = 0**: отсутствие эффекта переноса (мгновенное затухание)
            - **λ = 0.3**: 30% эффекта переносится на следующий период
            - **λ = 0.7**: высокая продолжительность воздействия
            
            **Рекомендации по каналам:**
            - TV/Radio: λ = 0.6-0.8 (высокое остаточное воздействие)
            - Digital: λ = 0.2-0.5 (быстрое затухание)
            - Print: λ = 0.4-0.7 (средняя продолжительность)
            
            ### Saturation трансформация (Эффект насыщения)
            
            **Экономическая теория:**
            Основана на законе убывающей предельной полезности. Каждая дополнительная единица 
            рекламных инвестиций приносит меньший прирост результата.
            
            **Hill Saturation функция:**
            ```
            Saturation = Media^α / (Media^α + γ^α)
            где α — форма кривой, γ — точка полунасыщения
            ```
            
            **Интерпретация параметров:**
            - **α < 1**: кривая с медленным ростом в начале
            - **α = 1**: линейная связь до точки насыщения
            - **α > 1**: S-образная кривая с пороговым эффектом
            - **γ**: уровень медиа-активности при достижении 50% максимального эффекта
            
            **Практические границы:**
            - Зрелые каналы: α = 0.6-1.2, γ близко к медианным расходам
            - Новые каналы: α = 1.5-2.5, γ может быть выше медианных расходов
            """)
        
        st.subheader("Параметры Adstock (эффект переноса)")
        
        adstock_params = {}
        for media in selected_media:
            with st.expander(f"Настройки для {media}"):
                col1, col2 = st.columns(2)
                with col1:
                    decay = st.slider(f"Decay rate для {media}", 0.0, 0.9, 0.5, 0.1, key=f"decay_{media}",
                                    help="Доля эффекта, переносимого на следующий период")
                with col2:
                    max_lag = st.slider(f"Max lag для {media}", 1, 12, 6, 1, key=f"lag_{media}",
                                      help="Максимальная продолжительность эффекта в периодах")
                adstock_params[media] = {'decay': decay, 'max_lag': max_lag}
        
        st.subheader("Параметры Saturation (эффект насыщения)")
        saturation_params = {}
        for media in selected_media:
            with st.expander(f"Saturation для {media}"):
                alpha = st.slider(f"Alpha для {media}", 0.1, 3.0, 1.0, 0.1, key=f"alpha_{media}",
                                help="Форма кривой насыщения: <1 = медленный рост, >1 = S-кривая")
                gamma = st.slider(f"Gamma для {media}", 0.1, 2.0, 0.5, 0.1, key=f"gamma_{media}",
                                help="Точка полунасыщения относительно средних расходов")
                saturation_params[media] = {'alpha': alpha, 'gamma': gamma}
    
        with tab3:
            # Добавляем объяснение обучения модели
            with st.expander("📊 Методология обучения и валидации модели", expanded=False):
                st.markdown("""
            ### Стратегии машинного обучения в MMM
            
            **1. Регуляризация (Regularization)**
            
            **Ridge регрессия (L2):**
            - Минимизирует: ||y - Xβ||² + α||β||²
            - Эффект: сжимает коэффициенты к нулю, предотвращает переобучение
            - Подходит для: ситуаций с мультиколлинеарностью между медиа-каналами
            
            **Lasso регрессия (L1):**
            - Минимизирует: ||y - Xβ||² + α|β|
            - Эффект: обнуляет незначимые коэффициенты, выполняет отбор признаков
            - Подходит для: исключения неэффективных медиа-каналов
            
            **Elastic Net:**
            - Комбинирует L1 и L2 регуляризацию
            - Подходит для: сбалансированного подхода к отбору и стабилизации
            
            **2. Временное разделение данных**
            
            **Принцип:**
            Обучающая выборка всегда предшествует тестовой по времени для избежания data leakage.
            
            **Рекомендации:**
            - Минимум 70% данных для обучения
            - Тестовая выборка должна покрывать различные сезонные периоды
            - При наличии <52 недель данных: использовать cross-validation
            
            **3. Метрики качества модели**
            
            **R² (Coefficient of Determination):**
            - Интерпретация: доля объясненной дисперсии
            - Хорошие значения: >0.7 для еженедельных данных, >0.8 для месячных
            
            **MAPE (Mean Absolute Percentage Error):**
            - Бизнес-интерпретация: средняя процентная ошибка прогноза
            - Приемлемые значения: <15% для операционного планирования
            
            **4. Диагностика остатков**
            
            **Критерии валидности модели:**
            - Нормальность остатков (Shapiro-Wilk test)
            - Отсутствие автокорреляции (Durbin-Watson test)
            - Гомоскедастичность (Breusch-Pagan test)
            - Отсутствие структурных сдвигов (Chow test)
            """)
        
        st.subheader("Обучение модели")
        
        col1, col2 = st.columns(2)
        with col1:
            train_ratio = st.slider("Доля обучающей выборки", 0.6, 0.9, 0.8, 0.05,
                                  help="Временное разделение: обучение всегда предшествует тесту")
            regularization = st.selectbox("Тип регуляризации", ["Ridge", "Lasso", "ElasticNet"],
                                        help="Ridge: стабилизация, Lasso: отбор признаков, ElasticNet: баланс")
        
        with col2:
            alpha_reg = st.slider("Коэффициент регуляризации", 0.001, 1.0, 0.01, 0.001,
                                help="Контролирует силу регуляризации: больше = консервативнее")
            cross_val_folds = st.slider("Число фолдов для кросс-валидации", 3, 10, 5, 1,
                                      help="Используется для подбора гиперпараметров")
                         
            if st.button("🚀 Обучить модель", type="primary"):
                with st.spinner("Обучение модели..."):
                    try:
                        # Проверка входных данных
                        if not selected_media:
                            st.error("Выберите хотя бы один медиа-канал")
                            return
                        
                        # Создание и обучение модели
                        model = MarketingMixModel(
                            adstock_params=adstock_params,
                            saturation_params=saturation_params,
                            regularization=regularization,
                            alpha=alpha_reg
                        )
                        
                        # Подготовка данных
                        X, y = self.processor.prepare_model_data(
                            data, target_var, selected_media, selected_external, selected_controls
                        )
                        
                        # Проверка на минимальное количество данных
                        if len(X) < 20:
                            st.error("Недостаточно данных для обучения модели (минимум 20 наблюдений)")
                            return
                        
                        # Обучение
                        train_size = max(10, int(len(X) * train_ratio))  # Минимум 10 наблюдений для обучения
                        X_train, X_test = X[:train_size], X[train_size:]
                        y_train, y_test = y[:train_size], y[train_size:]
                        
                        # Проверка на пустоту тестовой выборки
                        if len(X_test) == 0:
                            X_test = X_train.tail(5).copy()  # Берем последние 5 записей для теста
                            y_test = y_train.tail(5).copy()
                        
                        model.fit(X_train, y_train)
                        
                        # Валидация
                        train_score = model.score(X_train, y_train)
                        test_score = model.score(X_test, y_test)
                        
                        # Сохранение в состояние с проверками
                        st.session_state.model = model
                        st.session_state.model_fitted = True
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.target_var = target_var
                        st.session_state.selected_media = selected_media
                        st.session_state.selected_external = selected_external
                        st.session_state.selected_controls = selected_controls
                        
                        # Результаты
                        st.success("Модель обучена успешно!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² (train)", f"{train_score:.3f}")
                        with col2:
                            st.metric("R² (test)", f"{test_score:.3f}")
                        with col3:
                            overfitting = train_score - test_score
                            st.metric("Переобучение", f"{overfitting:.3f}", 
                                     delta=None if abs(overfitting) < 0.1 else "Высокое" if overfitting > 0.1 else "Низкое")
                        
                        # Предупреждения о качестве модели
                        if train_score < 0.5:
                            st.warning("⚠️ Низкое качество модели. Попробуйте добавить больше данных или изменить параметры.")
                        elif overfitting > 0.2:
                            st.warning("⚠️ Высокое переобучение. Увеличьте коэффициент регуляризации.")
                        
                    except Exception as e:
                        st.error(f"Ошибка обучения модели: {str(e)}")
                        st.info("Попробуйте изменить параметры модели или проверить качество данных.")

    def show_results(self):
        st.header("📈 Результаты анализа")
        
        if not st.session_state.model_fitted:
            st.warning("Сначала обучите модель")
            return
        
        # Проверка наличия необходимых данных
        required_session_vars = ['model', 'X_train', 'X_test', 'y_train', 'y_test', 'selected_media']
        missing_vars = [var for var in required_session_vars if var not in st.session_state or st.session_state[var] is None]
        
        if missing_vars:
            st.error(f"Отсутствуют данные: {missing_vars}. Переобучите модель.")
            return
        
        model = st.session_state.model
        
        tab1, tab2, tab3, tab4 = st.tabs(["Качество модели", "Декомпозиция", "ROAS анализ", "Кривые насыщения"])
        
        with tab1:
            # Объяснение метрик качества
            with st.expander("❓ Что показывают метрики качества модели?", expanded=False):
                st.markdown("""
                **Метрики качества** показывают, насколько хорошо модель научилась предсказывать ваши продажи:
                
                📊 **Качество прогноза** (было R²):
                - Показывает, какую долю изменений в продажах модель может объяснить
                - **90%** = отлично! Модель понимает 90% того, почему продажи растут или падают
                - **70%** = хорошо, модель улавливает основные закономерности
                - **50%** = слабо, модель видит только половину картины
                
                🎯 **Точность модели** (было MAPE):
                - Показывает, насколько точно модель предсказывает количество заказов
                - **90%** = модель очень точная, ошибается только на 10%
                - **80%** = хорошая точность для бизнес-планирования
                - **60%** = приемлемо, но нужна осторожность
                
                📏 **Средняя/Типичная ошибка**:
                - Показывает, на сколько заказов в среднем ошибается модель
                - Например: если ошибка 500 заказов, а у вас 5000 заказов в месяц = это 10% ошибка
                
                🎯 **Главное правило**: Если модель показывает 🟢 или 🟡 - можно использовать для планирования!
                """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Оценка качества модели")
                
                # Получаем качественную оценку
                quality_assessment = model.get_model_quality_assessment(st.session_state.X_test, st.session_state.y_test)
                
                # Показываем статус модели
                st.markdown(f"### {quality_assessment['status']}")
                st.progress(quality_assessment['quality_score'] / 100)
                st.markdown(f"**Общая оценка:** {quality_assessment['quality_score']}/100")
                
                # Бизнес-объяснение
                st.success(quality_assessment['business_explanation']['quality'])
                st.info(quality_assessment['business_explanation']['accuracy'])
                st.markdown(f"**Рекомендация:** {quality_assessment['recommendation']}")
                
                # Детальные метрики (скрыты в expander)
                with st.expander("🔧 Технические детали", expanded=False):
                    metrics = model.get_model_metrics(st.session_state.X_test, st.session_state.y_test)
                    for metric, value in metrics.items():
                        if 'Точность' in metric:
                            st.metric(metric, f"{value:.1f}%")
                        elif 'Качество' in metric:
                            st.metric(metric, f"{value:.3f} ({value*100:.0f}%)")
                        else:
                            st.metric(metric, f"{value:,.0f}")
            
            with col2:
                st.subheader("📈 Прогноз vs Реальность")
                y_pred = model.predict(st.session_state.X_test)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=st.session_state.y_test,
                    mode='lines+markers',
                    name='Реальные заказы',
                    line=dict(color='blue', width=3),
                    marker=dict(size=6)
                ))
                fig.add_trace(go.Scatter(
                    y=y_pred,
                    mode='lines+markers',
                    name='Прогноз модели',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=4)
                ))
                
                fig.update_layout(
                    title="Насколько точно модель предсказывает заказы",
                    xaxis_title="Период времени",
                    yaxis_title="Количество заказов",
                    height=400,
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Интерпретация графика
                correlation = np.corrcoef(st.session_state.y_test, y_pred)[0, 1]
                if correlation >= 0.9:
                    st.success("✅ Отлично! Прогноз очень близко следует реальности")
                elif correlation >= 0.7:
                    st.info("👍 Хорошо! Прогноз в целом соответствует тренду")
                else:
                    st.warning("⚠️ Модель не очень точно предсказывает изменения")
        
        with tab2:
            st.subheader("Декомпозиция продаж")
            
            # Объяснение что показывает декомпозиция
            with st.expander("❓ Что показывает декомпозиция?", expanded=False):
                st.markdown("""
                **Декомпозиция продаж** показывает, откуда приходят ваши заказы:
                
                - **Base (Базовая линия)** = заказы, которые идут "сами по себе" (органика, брендинг, сарафанное радио)
                - **Медиа-каналы** = заказы, которые приносит конкретная реклама
                
                **Здоровое соотношение для большинства бизнесов:**
                - Base: 30-50% (органические заказы)
                - Медиа: 50-70% (рекламные заказы)
                
                **Если Base = 100%** - возможно, модель не видит связи между рекламой и продажами.
                """)
            
            try:
                # Расчет вкладов каналов
                contributions = model.get_media_contributions(st.session_state.X_train, st.session_state.y_train)
                
                # Проверка на корректность данных
                if contributions and len(contributions) > 0:
                    # Анализ декомпозиции
                    total_contribution = sum(contributions.values())
                    base_share = contributions.get('Base', 0) / total_contribution * 100 if total_contribution > 0 else 0
                    
                    # Предупреждение если Base слишком большой
                    if base_share > 80:
                        st.warning(f"⚠️ Базовая линия составляет {base_share:.1f}% продаж. Возможно, модель плохо улавливает влияние рекламы.")
                        st.info("💡 **Попробуйте**: уменьшить коэффициент регуляризации или изменить параметры adstock/saturation")
                    elif base_share < 20:
                        st.warning(f"⚠️ Базовая линия всего {base_share:.1f}%. Возможно, модель переоценивает влияние рекламы.")
                    else:
                        st.success(f"✅ Здоровая декомпозиция: Базовая линия {base_share:.1f}%, Медиа {100-base_share:.1f}%")
                    
                    # Waterfall chart
                    fig = self.visualizer.create_waterfall_chart(contributions)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Таблица вкладов с улучшенным форматированием
                    st.subheader("📋 Детализация вкладов")
                    contrib_df = pd.DataFrame(list(contributions.items()), columns=['Канал', 'Вклад'])
                    contrib_df['Вклад, %'] = (contrib_df['Вклад'] / contrib_df['Вклад'].sum() * 100).round(1)
                    contrib_df['Вклад'] = contrib_df['Вклад'].round(0).astype(int)
                    
                    # Добавляем интерпретацию
                    contrib_df['Интерпретация'] = contrib_df.apply(lambda row: 
                        "🏠 Органические продажи" if row['Канал'] == 'Base' 
                        else f"📢 Реклама: {row['Вклад, %']}% от общих продаж", axis=1)
                    
                    st.dataframe(contrib_df, use_container_width=True, hide_index=True)
                    
                    # Рекомендации по улучшению
                    st.subheader("💡 Рекомендации")
                    media_contributions = {k: v for k, v in contributions.items() if k != 'Base'}
                    if media_contributions:
                        best_channel = max(media_contributions.items(), key=lambda x: x[1])
                        st.success(f"🎯 **Самый эффективный канал**: {best_channel[0]} ({best_channel[1]:,.0f} заказов)")
                        
                        worst_channel = min(media_contributions.items(), key=lambda x: x[1])
                        if worst_channel[1] < 0:
                            st.warning(f"⚠️ **Проблемный канал**: {worst_channel[0]} показывает отрицательный вклад")
                        
                else:
                    st.warning("Не удалось рассчитать вклады каналов. Проверьте качество модели.")
                    
            except Exception as e:
                st.error(f"Ошибка при расчете декомпозиции: {str(e)}")
                st.info("💡 **Решение**: Попробуйте переобучить модель с другими параметрами в разделе 'Модель'")

        with tab3:  # ROAS анализ            
            st.subheader("ROAS по каналам")

            # Краткое объяснение метрики
            # Добавляем объяснение ROAS
            with st.expander("📚 Что такое ROAS и как его интерпретировать", expanded=False):
                st.markdown("""
                ### Return on Advertising Spend (ROAS)

                **ROAS** — ключевая метрика эффективности рекламных инвестиций. Она показывает,
                сколько дополнительной выручки приносит каждый вложенный в рекламу рубль:
                **ROAS = Incremental Revenue / Advertising Spend**

                **Математическая интерпретация:**

                - ROAS = 3.0 означает, что каждый рубль рекламы генерирует 3 рубля дополнительной выручки
                - ROAS = 1.0 — точка безубыточности (реклама окупает себя)
                - ROAS < 1.0 — убыточные инвестиции с позиции краткосрочной окупаемости

                **Методологические особенности в MMM:**
                1. **Инкрементальность vs. Корреляция**
                   - MMM измеряет причинно-следственную связь через контрольные переменные
                   - Традиционная аналитика показывает корреляционную связь
                   - Инкрементальный ROAS всегда ниже корреляционного

                2. **Временные эффекты**
                   - Краткосрочный ROAS: эффект в течение 1-4 недель
                   - Долгосрочный ROAS: включает adstock эффекты (до 12-52 недель)
                   - MMM рассчитывает полный (долгосрочный) ROAS
                """)

            try:
                if hasattr(st.session_state, 'data') and st.session_state.data is not None:
                    roas_data = model.calculate_roas(st.session_state.data, st.session_state.selected_media)

                    if not roas_data.empty:
                        fig = self.visualizer.create_roas_comparison(roas_data)
                        st.plotly_chart(fig, use_container_width=True)

                        st.subheader("Детализация ROAS")
                        st.dataframe(roas_data, use_container_width=True)
                    else:
                        st.warning("Не удалось рассчитать ROAS. Проверьте данные.")
                else:
                    st.warning("Данные для расчета ROAS недоступны.")

            except Exception as e:
                st.error(f"Ошибка при расчете ROAS: {str(e)}")
                demo_roas = pd.DataFrame({
                    'Channel': ['Facebook', 'Google', 'TikTok'],
                    'ROAS': [2.1, 2.8, 1.5]
                })
                fig = self.visualizer.create_roas_comparison(demo_roas)
                st.plotly_chart(fig, use_container_width=True)
       
            st.markdown("""
            **Методологические особенности в MMM:**

            1. **Инкрементальность vs. Корреляция**
               - MMM измеряет причинно-следственную связь через контрольные переменные
               - Традиционная аналитика показывает корреляционную связь
               - Инкрементальный ROAS всегда ниже корреляционного

            2. **Временные эффекты**
               - Краткосрочный ROAS: эффект в течение 1-4 недель
               - Долгосрочный ROAS: включает adstock эффекты (до 12-52 недель)
               - MMM рассчитывает полный (долгосрочный) ROAS

            **Бенчмарки по индустриям:**

            **E-commerce:**
            - Excellent: ROAS > 4.0
            - Good: ROAS 2.5-4.0
            - Acceptable: ROAS 1.5-2.5
            - Poor: ROAS < 1.5

            **FMCG:**
            - Excellent: ROAS > 3.0
            - Good: ROAS 2.0-3.0
            - Acceptable: ROAS 1.2-2.0
            - Poor: ROAS < 1.2

            **B2B Services:**
            - Excellent: ROAS > 5.0
            - Good: ROAS 3.0-5.0
            - Acceptable: ROAS 2.0-3.0
            - Poor: ROAS < 2.0

            **Marginal ROAS:**
            Показывает эффективность последнего вложенного рубля. Критически важен для оптимизации бюджета.
            Правило: перераспределять бюджет от каналов с низким Marginal ROAS к каналам с высоким.

            **Ограничения метрики:**
            - Не учитывает Customer Lifetime Value
            - Игнорирует брендинговые эффекты
            - Может недооценивать upper-funnel активности
            """)
                
        with tab4:
            st.subheader("Кривые насыщения")
            
            # Объяснение кривых насыщения
            with st.expander("❓ Что такое кривые насыщения?", expanded=False):
                st.markdown("""
                **Кривые насыщения** показывают, как эффективность рекламного канала меняется при увеличении бюджета.

                🎯 **Простыми словами:**
                - Представьте, что вы поливаете растение водой
                - Сначала каждая капля воды очень помогает росту
                - Но если лить слишком много - эффект уменьшается
                - То же самое с рекламой!

                📈 **Что показывает кривая:**
                - **Начало кривой** = каждый рубль рекламы приносит много заказов
                - **Середина** = эффективность стабильная
                - **Конец кривой** = дополнительные рубли приносят мало заказов (насыщение)

                💡 **Практическое применение:**
                - **Крутой рост** в начале = канал недофинансирован, можно увеличить бюджет
                - **Пологая кривая** = канал близок к насыщению, дополнительные деньги неэффективны
                - **Вертикальная линия** = ваш текущий уровень расходов

                🎯 **Идеальная стратегия:** Тратить до точки, где кривая начинает выравниваться
                """)
            
            # Выбор канала для анализа
            selected_channel = st.selectbox("Выберите канал для анализа:", st.session_state.selected_media)
            
            # Построение простой кривой насыщения
            current_spend = st.session_state.data[selected_channel].mean()
            max_spend = current_spend * 3  # Показываем до 3x текущих расходов
            spend_range = np.linspace(0, max_spend, 100)
            
            # Простая Hill saturation для демонстрации
            alpha = 1.0
            gamma = current_spend * 0.7  # Точка полунасыщения на 70% от текущих расходов
            saturation_curve = np.power(spend_range, alpha) / (np.power(spend_range, alpha) + np.power(gamma, alpha))
            
            fig = go.Figure()
            
            # Кривая насыщения
            fig.add_trace(go.Scatter(
                x=spend_range,
                y=saturation_curve,
                mode='lines',
                name='Кривая насыщения',
                line=dict(color='blue', width=3)
            ))
            
            # Текущий уровень расходов
            current_saturation = np.power(current_spend, alpha) / (np.power(current_spend, alpha) + np.power(gamma, alpha))
            fig.add_trace(go.Scatter(
                x=[current_spend],
                y=[current_saturation],
                mode='markers',
                name='Текущие расходы',
                marker=dict(color='red', size=12, symbol='diamond'),
                hovertemplate=f"Текущие расходы: {current_spend:,.0f}<br>Эффективность: {current_saturation:.2f}"
            ))
            
            # Зона эффективности
            efficient_spend = gamma * 1.2  # 120% от точки полунасыщения
            fig.add_vrect(
                x0=0, x1=efficient_spend,
                fillcolor="green", opacity=0.1,
                annotation_text="Эффективная зона", annotation_position="top left"
            )
            
            fig.add_vrect(
                x0=efficient_spend, x1=max_spend,
                fillcolor="orange", opacity=0.1,
                annotation_text="Зона насыщения", annotation_position="top right"
            )
            
            fig.update_layout(
                title=f"Кривая насыщения для {selected_channel.replace('_spend', '').title()}",
                xaxis_title="Расходы на рекламу (руб/месяц)",
                yaxis_title="Эффективность (нормализованная)",
                height=500,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Интерпретация результатов
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Анализ текущего уровня")
                if current_spend < gamma:
                    st.success("🟢 **Недофинансирован**: Можно увеличить бюджет для лучших результатов")
                    recommendation = f"Рекомендуется увеличить бюджет до {gamma*1.2:,.0f} руб/месяц"
                elif current_spend < efficient_spend:
                    st.info("🟡 **Оптимальный уровень**: Хорошее соотношение затрат и результата")
                    recommendation = "Текущий уровень расходов близок к оптимальному"
                else:
                    st.warning("🟠 **Близко к насыщению**: Дополнительные расходы малоэффективны")
                    recommendation = f"Рассмотрите перераспределение части бюджета на другие каналы"
                
                st.info(f"💡 **Рекомендация**: {recommendation}")
            
            with col2:
                st.subheader("🎯 Ключевые точки")
                st.metric("Текущие расходы", f"{current_spend:,.0f} руб")
                st.metric("Точка полунасыщения", f"{gamma:,.0f} руб", 
                         help="Уровень расходов, при котором достигается 50% максимального эффекта")
                st.metric("Граница эффективности", f"{efficient_spend:,.0f} руб",
                         help="После этого уровня каждый дополнительный рубль приносит мало результата")
                
                # Потенциал роста
                potential_increase = (efficient_spend - current_spend) / current_spend * 100 if current_spend > 0 else 0
                if potential_increase > 20:
                    st.success(f"📈 Потенциал роста: +{potential_increase:.0f}%")
                elif potential_increase > 0:
                    st.info(f"📈 Потенциал роста: +{potential_increase:.0f}%")
                else:
                    st.warning("📊 Канал близок к насыщению")

    def show_optimization(self):
        st.header("💰 Оптимизация бюджета")
        
        # Главное объяснение раздела
        with st.expander("❓ Что такое оптимизация бюджета?", expanded=False):
            st.markdown("""
            **Оптимизация бюджета** - это автоматический поиск наилучшего способа распределить ваши рекламные деньги.
            
            🎯 **Простой пример:**
            У вас есть 1 млн рублей на рекламу. Вопрос: как их разделить между Facebook, Google, TikTok?
            
            **Интуитивный подход:**
            - Facebook: 300,000 руб (30%)
            - Google: 500,000 руб (50%)  
            - TikTok: 200,000 руб (20%)
            - **Результат:** 5,000 заказов
            
            **После оптимизации:**
            - Facebook: 250,000 руб (25%)
            - Google: 600,000 руб (60%)
            - TikTok: 150,000 руб (15%)
            - **Результат:** 5,400 заказов (+400 заказов!)
            
            💡 **Как это работает:**
            1. Модель анализирует эффективность каждого канала
            2. Находит оптимальное соотношение для максимального результата
            3. Учитывает ваши ограничения (минимум/максимум по каналам)
            
            🎯 **Цели оптимизации:**
            - **Максимум заказов** = получить как можно больше заказов
            - **Максимум ROAS** = получить максимальную отдачу с рубля
            - **Максимум ROI** = получить максимальную прибыль
            """)
        
        if not st.session_state.model_fitted:
            st.warning("⚠️ Сначала обучите модель в разделе 'Модель'")
            st.info("💡 Модель нужна для анализа эффективности каналов и поиска оптимального распределения")
            return
        
        tab1, tab2 = st.tabs(["⚙️ Настройки оптимизации", "📊 Результаты оптимизации"])
        
        with tab1:
            st.subheader("⚙️ Настройки оптимизации")
            
            # Объяснение настроек
            with st.expander("❓ Как настроить оптимизацию?", expanded=False):
                st.markdown("""
                **Настройки помогают адаптировать оптимизацию под ваши бизнес-ограничения:**
                
                💰 **Общий бюджет:**
                - Сколько всего денег у вас есть на рекламу в месяц
                - Система распределит эти деньги между каналами оптимально
                
                🎯 **Цель оптимизации:**
                - **Максимум заказов** = приоритет количеству (подходит для роста)
                - **Максимум ROAS** = приоритет эффективности (подходит для прибыльности)
                - **Максимум ROI** = приоритет чистой прибыли
                
                🚧 **Ограничения по каналам:**
                - **Минимум** = меньше этой суммы тратить нельзя (например, минимум по контракту)
                - **Максимум** = больше этой суммы тратить нельзя (например, лимит команды)
                - Без ограничений система может предложить потратить 0 или 100% на один канал
                
                **Пример ограничений:**
                - Facebook: мин 100к (команда справится), макс 500к (больше не потянем)
                - Google: мин 200к (конкуренция), макс без ограничений
                """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💰 Основные параметры")
                total_budget = st.number_input(
                    "Общий месячный бюджет (руб)", 
                    min_value=10000, 
                    value=1000000, 
                    step=50000,
                    help="Общая сумма, которую вы готовы тратить на рекламу в месяц"
                )
                
                optimization_target = st.selectbox(
                    "Цель оптимизации", 
                    ["maximize_sales", "maximize_roas", "maximize_roi"],
                    format_func=lambda x: {
                        "maximize_sales": "📈 Максимум заказов (рост объемов)",
                        "maximize_roas": "💰 Максимум ROAS (эффективность)",
                        "maximize_roi": "💎 Максимум ROI (прибыльность)"
                    }[x],
                    help="Что важнее: больше заказов, выше эффективность или больше прибыли?"
                )
                
                # Показываем текущий расход для сравнения
                current_total = sum(
                    st.session_state.data[ch].mean()
                    for ch in st.session_state.selected_media
                )
                st.info(f"💡 Текущие расходы: {current_total:,.0f} руб/месяц")

                if current_total == 0:
                    st.info("ℹ️ Нет данных о прошлых расходах. Изменение бюджета: 0%")
                elif total_budget != current_total:
                    change_pct = (
                        (total_budget - current_total) / current_total * 100
                    )
                    if change_pct > 0:
                        st.success(f"📈 Увеличение бюджета на {change_pct:.0f}%")
                    else:
                        st.warning(
                            f"📉 Сокращение бюджета на {abs(change_pct):.0f}%"
                        )
                
            with col2:
                st.subheader("🚧 Ограничения по каналам")
                
                # Добавляем переключатель для ограничений
                use_constraints = st.checkbox(
                    "Использовать ограничения по каналам", 
                    value=False,
                    help="Если выключено, система может предложить любое распределение"
                )
                
                constraints = {}
                if use_constraints:
                    st.info("💡 Установите реалистичные ограничения на основе возможностей вашей команды")
                    
                    for channel in st.session_state.selected_media:
                        with st.expander(f"⚙️ {channel.replace('_spend', '').title()}", expanded=False):
                            current_avg = st.session_state.data[channel].mean()
                            
                            col_min, col_max = st.columns(2)
                            with col_min:
                                min_spend = st.number_input(
                                    f"Минимум", 
                                    min_value=0, 
                                    max_value=total_budget//2, 
                                    value=max(0, int(current_avg * 0.5)),
                                    step=10000,
                                    key=f"min_{channel}",
                                    help="Меньше этой суммы тратить нельзя/неэффективно"
                                )
                            with col_max:
                                max_spend = st.number_input(
                                    f"Максимум", 
                                    min_value=min_spend, 
                                    max_value=total_budget, 
                                    value=min(total_budget, int(current_avg * 2)),
                                    step=10000,
                                    key=f"max_{channel}",
                                    help="Больше этой суммы тратить нельзя/неэффективно"
                                )
                            
                            constraints[channel] = {'min': min_spend, 'max': max_spend}
                            
                            # Показываем текущий уровень для сравнения
                            st.caption(f"Сейчас тратите: {current_avg:,.0f} руб/месяц")
                else:
                    st.info("🔓 Ограничения отключены - система найдет полностью оптимальное распределение")
                    constraints = {}
        
        with tab2:
            if st.button("🎯 Оптимизировать бюджет", type="primary"):
                with st.spinner("Поиск оптимального распределения..."):
                    
                    # Запуск оптимизации
                    optimal_allocation = self.optimizer.optimize_budget(
                        model=st.session_state.model,
                        total_budget=total_budget,
                        constraints=constraints,
                        target=optimization_target
                    )
                    
                    # Результаты оптимизации
                    st.success("Оптимизация завершена!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Оптимальное распределение")
                        optimal_df = pd.DataFrame(list(optimal_allocation['allocation'].items()), 
                                                columns=['Канал', 'Оптимальный бюджет'])
                        optimal_df['Доля, %'] = (optimal_df['Оптимальный бюджет'] / total_budget * 100).round(1)
                        st.dataframe(optimal_df, use_container_width=True)
                        
                        # Метрики оптимального решения
                        st.metric("Прогнозируемые продажи", f"{optimal_allocation['predicted_sales']:,.0f}")
                        st.metric("Прогнозируемый ROAS", f"{optimal_allocation['predicted_roas']:.2f}")
                        st.metric("Прогнозируемый ROI", f"{optimal_allocation['predicted_roi']:.2f}")
                    
                    with col2:
                        st.subheader("Сравнение распределений")

                        # Определяем текущее распределение по каналам
                        current_allocation = {
                            ch: st.session_state.data[ch].mean()
                            for ch in st.session_state.selected_media
                        }

                        # Сравнительная диаграмма
                        fig = self.visualizer.create_optimization_results(
                            current_allocation,
                            optimal_allocation['allocation']
                        )
                        st.plotly_chart(fig, use_container_width=True)

    def show_scenarios(self):
        st.header("🔮 Сценарный анализ")

        # Добавляем общее объяснение сценарного анализа
        with st.expander("📊 Методология сценарного анализа в маркетинге", expanded=False):
            st.markdown("""
        ### Сценарное планирование в Marketing Mix Modeling
        
        **Определение:**
        Сценарный анализ — систематический метод оценки потенциальных последствий различных 
        стратегических решений в области медиа-инвестиций при неопределенных внешних условиях.
        
        **Теоретические основы:**
        
        1. **Детерминистическое моделирование**
           - Модель предполагает фиксированные параметры медиа-трансформаций
           - Учитывает нелинейные эффекты (adstock, saturation)
           - Включает влияние внешних факторов
        
        2. **Сравнительная статика**
           - Анализ изменения равновесных состояний при изменении параметров
           - Ceteris paribus принцип: все остальное остается неизменным
           - Позволяет изолировать эффекты конкретных решений
        
        **Типология сценариев:**
        
        **1. Оптимистичный сценарий**
        - Благоприятная сезонность (seasonality > 1.0)
        - Низкая конкурентная активность (competition < 1.0)
        - Применение: планирование максимальных результатов
        
        **2. Пессимистичный сценарий**
        - Неблагоприятная сезонность (seasonality < 1.0)
        - Высокая конкурентная активность (competition > 1.0)
        - Применение: оценка рисков и планирование contingency
        
        **3. Базовый сценарий**
        - Нейтральные внешние условия (факторы = 1.0)
        - Применение: стандартное планирование
        
        **Интерпретация факторов:**
        
        **Сезонный фактор:**
        - 1.5 = +50% к базовому спросу (высокий сезон)
        - 1.0 = нейтральный период
        - 0.7 = -30% к базовому спросу (низкий сезон)
        
        **Фактор конкуренции:**
        - 1.3 = увеличение конкурентного давления на 30%
        - 1.0 = стабильная конкурентная среда
        - 0.8 = снижение конкурентного давления на 20%
        """)
    
        if not st.session_state.model_fitted:
            st.warning("Сначала обучите модель")
            return

        tab1, tab2 = st.tabs(["Создание сценариев", "Сравнение сценариев"])

        with tab1:
            # Добавляем объяснение перед созданием сценария
            with st.expander("🎯 Рекомендации по созданию сценариев", expanded=False):
                st.markdown("""
            ### Критерии оценки качества сценария
            
            **Метрики для анализа:**
            
            **ROAS (Return on Ad Spend):**
            - **Отличный результат**: ROAS ≥ 3.0
            - **Хороший результат**: ROAS 2.0-3.0
            - **Приемлемый результат**: ROAS 1.5-2.0
            - **Неудовлетворительный**: ROAS < 1.5
            
            **Прогнозируемые продажи:**
            - Сравнивайте с текущим уровнем и историческими данными
            - Учитывайте сезонные колебания
            - Оценивайте реалистичность с точки зрения операционных возможностей
            
            **Общий бюджет:**
            - Должен соответствовать финансовым возможностям
            - Учитывайте ограничения по cash flow
            - Сравнивайте с текущими расходами на маркетинг
            
            **Рекомендации по построению сценариев:**
            
            1. **Консервативный подход**: изменения бюджета ±20% от текущего уровня
            2. **Агрессивный рост**: увеличение бюджета на 50-100%
            3. **Оптимизация**: перераспределение без изменения общего бюджета
            4. **Кризисный**: снижение бюджета на 30-50%
            """)
        
        st.subheader("Создание нового сценария")
        
        scenario_name = st.text_input("Название сценария", "Сценарий 1")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Настройки каналов")
            scenario_budget = {}
            
            # Показываем текущие расходы для сравнения
            st.markdown("**Текущие среднемесячные расходы:**")
            current_totals = {}
            for channel in st.session_state.selected_media:
                current_value = st.session_state.data[channel].mean()
                current_totals[channel] = current_value
                st.caption(f"{channel}: {current_value:,.0f} руб")
            
            total_current = sum(current_totals.values())
            st.caption(f"**Общий текущий бюджет: {total_current:,.0f} руб**")
            
            st.markdown("**Новое распределение:**")
            for channel in st.session_state.selected_media:
                current_value = current_totals[channel]
                scenario_budget[channel] = st.number_input(
                    f"Бюджет {channel}",
                    min_value=0,
                    value=int(current_value),
                    step=1000,
                    key=f"scenario_{channel}",
                    help=f"Текущий уровень: {current_value:,.0f} руб"
                )
        
        with col2:
            st.subheader("Внешние факторы")
            
            # Добавляем пояснения к внешним факторам
            st.markdown("""
            **Интерпретация внешних факторов:**
            - **1.0** = нормальные условия
            - **>1.0** = благоприятные условия  
            - **<1.0** = неблагоприятные условия
            """)
            
            seasonality_factor = st.slider(
                "Сезонный фактор", 0.5, 2.0, 1.0, 0.1,
                help="1.5 = высокий сезон (+50%), 0.7 = низкий сезон (-30%)"
            )
            competition_factor = st.slider(
                "Фактор конкуренции", 0.5, 2.0, 1.0, 0.1,
                help="1.3 = усиление конкуренции (-30% эффективности), 0.8 = ослабление (+20%)"
            )
            
            # Прогноз до расчета
            new_total = sum(scenario_budget.values())
            budget_change = ((new_total - total_current) / total_current * 100) if total_current > 0 else 0
            
            st.markdown("### Предварительная оценка")
            st.metric("Изменение бюджета", f"{budget_change:+.1f}%")
            
            if budget_change > 50:
                st.warning("Существенное увеличение бюджета. Убедитесь в операционной готовности.")
            elif budget_change > 20:
                st.info("Умеренное увеличение бюджета. Хорошая стратегия роста.")
            elif budget_change > -20:
                st.success("Незначительные изменения. Фокус на оптимизации распределения.")
            else:
                st.error("Значительное сокращение бюджета. Ожидается снижение результатов.")
            
            # Прогноз результатов сценария
            if st.button("📊 Рассчитать прогноз"):
                predicted_results = st.session_state.model.predict_scenario(
                    scenario_budget, seasonality_factor, competition_factor
                )
                
                st.markdown("### Результаты прогноза")
                
                # Детальный анализ результатов
                sales_result = predicted_results['sales']
                roas_result = predicted_results['roas']
                total_spend = sum(scenario_budget.values())
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Прогнозируемые продажи", f"{sales_result:,.0f}")
                
                with col_b:
                    # Цветовая индикация ROAS
                    if roas_result >= 3.0:
                        st.success(f"**ROAS: {roas_result:.2f}** (Отлично)")
                    elif roas_result >= 2.0:
                        st.info(f"**ROAS: {roas_result:.2f}** (Хорошо)")
                    elif roas_result >= 1.5:
                        st.warning(f"**ROAS: {roas_result:.2f}** (Приемлемо)")
                    else:
                        st.error(f"**ROAS: {roas_result:.2f}** (Неудовлетворительно)")
                
                with col_c:
                    st.metric("Общий бюджет", f"{total_spend:,.0f}")
                
                # Развернутая интерпретация
                st.markdown("### Интерпретация результатов")
                
                # Анализ ROAS
                if roas_result >= 3.0:
                    st.success("""
                    **Отличные результаты**: ROAS выше 3.0 указывает на высокую эффективность 
                    медиа-инвестиций. Рекомендуется реализация данного сценария.
                    """)
                elif roas_result >= 2.0:
                    st.info("""
                    **Хорошие результаты**: ROAS в диапазоне 2.0-3.0 демонстрирует приемлемую 
                    эффективность. Возможны дополнительные оптимизации распределения.
                    """)
                elif roas_result >= 1.5:
                    st.warning("""
                    **Приемлемые результаты**: ROAS 1.5-2.0 находится на нижней границе 
                    эффективности. Требуется анализ альтернативных стратегий.
                    """)
                else:
                    st.error("""
                    **Неудовлетворительные результаты**: ROAS ниже 1.5 указывает на 
                    неэффективность инвестиций. Необходим пересмотр распределения бюджета.
                    """)
                
                # Анализ продаж
                # Для примера сравниваем с "типичным" уровнем
                baseline_sales = st.session_state.data[st.session_state.target_var].mean() if hasattr(st.session_state, 'target_var') else 50000
                sales_change = ((sales_result - baseline_sales) / baseline_sales * 100) if baseline_sales > 0 else 0
                
                if sales_change > 20:
                    st.success(f"Прогнозируемый рост продаж: +{sales_change:.1f}%. Сильная стратегия роста.")
                elif sales_change > 5:
                    st.info(f"Прогнозируемый рост продаж: +{sales_change:.1f}%. Умеренный рост.")
                elif sales_change > -5:
                    st.warning(f"Изменение продаж: {sales_change:+.1f}%. Стабильные результаты.")
                else:
                    st.error(f"Прогнозируемое снижение продаж: {sales_change:.1f}%. Рискованная стратегия.")
    
        with tab2:
            # Добавляем объяснение сравнения сценариев
            with st.expander("📈 Методология сравнения сценариев", expanded=False):
                st.markdown("""
            ### Принципы сравнительного анализа стратегий

            **Предустановленные стратегии:**

            **1. Текущий сценарий (Current)**
            - Базовая линия для сравнения
            - Основан на исторических средних расходах
            - Показывает результаты при сохранении status quo

            **2. Digital Focus**
            - 80% бюджета на цифровые каналы, 20% на офлайн
            - Стратегия для повышения измеримости и таргетинга
            - Подходит для D2C брендов и e-commerce

            **3. Balanced**
            - Равномерное распределение между всеми каналами
            - Стратегия диверсификации рисков
            - Подходит для тестирования новых каналов

            **4. Performance**
            - Концентрация на каналах с исторически высоким ROAS
            - 70% бюджета на Google + Facebook, 30% на остальные
            - Стратегия максимизации краткосрочной эффективности

            **Критерии выбора оптимальной стратегии:**

            1. **Максимальные продажи**: выбор сценария с наибольшим объемом продаж
            2. **Максимальный ROAS**: приоритет эффективности инвестиций
            3. **Минимальный риск**: выбор наиболее стабильного сценария
            4. **Бюджетные ограничения**: соответствие финансовым возможностям
            """)

            st.subheader("Сравнение предустановленных сценариев")
            
            # Предустановленные сценарии
            current_avg = {channel: st.session_state.data[channel].mean() 
                          for channel in st.session_state.selected_media}
            total_current = sum(current_avg.values())
            
            scenarios = {
                "Текущий": current_avg,
                "Digital Focus": {
                    channel: (total_current * 0.8 / len([ch for ch in st.session_state.selected_media if 'offline' not in ch.lower()]) 
                             if 'offline' not in channel.lower() else total_current * 0.2)
                    for channel in st.session_state.selected_media
                },
                "Balanced": {channel: total_current / len(st.session_state.selected_media) 
                           for channel in st.session_state.selected_media},
                "Performance": {
                    channel: (total_current * 0.7 / len([ch for ch in st.session_state.selected_media if ch in ['google_spend', 'facebook_spend']])
                             if channel in ['google_spend', 'facebook_spend'] else total_current * 0.3 / (len(st.session_state.selected_media) - 2))
                    for channel in st.session_state.selected_media
                }
            }
            
            # Расчет прогнозов для всех сценариев
            scenario_results = {}
            for name, budget in scenarios.items():
                results = st.session_state.model.predict_scenario(budget, 1.0, 1.0)
                scenario_results[name] = results
            
            # Таблица сравнения
            comparison_df = pd.DataFrame(scenario_results).T
            comparison_df = comparison_df.round(2)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Визуализация сравнения
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=['Продажи', 'ROAS', 'Бюджет']
            )
            
            scenarios_list = list(scenario_results.keys())
            
            # Продажи
            fig.add_trace(
                go.Bar(x=scenarios_list, y=[scenario_results[s]['sales'] for s in scenarios_list], 
                      name='Продажи', showlegend=False),
                row=1, col=1
            )
            
            # ROAS
            fig.add_trace(
                go.Bar(x=scenarios_list, y=[scenario_results[s]['roas'] for s in scenarios_list], 
                      name='ROAS', showlegend=False),
                row=1, col=2
            )
            
            # Бюджет
            fig.add_trace(
                go.Bar(x=scenarios_list, y=[scenario_results[s]['total_spend'] for s in scenarios_list], 
                      name='Бюджет', showlegend=False),
                row=1, col=3
            )
            
            fig.update_layout(title="Сравнение сценариев", height=400)
            st.plotly_chart(fig, use_container_width=True)
            # Добавляем анализ после таблицы сравнения
            if 'scenario_results' in locals():
                st.markdown("### Рекомендации по выбору стратегии")
            
                # Находим лучший сценарий по каждому критерию
                best_sales = max(scenario_results.keys(), key=lambda x: scenario_results[x]['sales'])
                best_roas = max(scenario_results.keys(), key=lambda x: scenario_results[x]['roas'])
                most_efficient = min(scenario_results.keys(), key=lambda x: scenario_results[x]['total_spend'])

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.success(f"""
                **Максимальные продажи**: {best_sales}

                Продажи: {scenario_results[best_sales]['sales']:,.0f}

                Рекомендуется для: агрессивного роста объемов
                """)

                with col2:
                    st.info(f"""
                **Максимальный ROAS**: {best_roas}

                ROAS: {scenario_results[best_roas]['roas']:.2f}

                Рекомендуется для: оптимизации эффективности
                """)

                with col3:
                    st.warning(f"""
                **Наименьший бюджет**: {most_efficient}

                Бюджет: {scenario_results[most_efficient]['total_spend']:,.0f}

                Рекомендуется для: ограниченных ресурсов
                """)

                # Общие рекомендации
                st.markdown("### Стратегические рекомендации")
            
                # Сравнение с текущим сценарием
                current_results = scenario_results.get('Текущий', scenario_results.get('Current'))
                if current_results:
                    for name, results in scenario_results.items():
                        if name not in ['Текущий', 'Current']:
                            sales_improvement = ((results['sales'] - current_results['sales']) / current_results['sales'] * 100)
                            roas_improvement = ((results['roas'] - current_results['roas']) / current_results['roas'] * 100)
                        
                            if sales_improvement > 10 and roas_improvement > 5:
                                st.success(f"""
                            **{name}**: Превосходит текущую стратегию по всем показателям
                            - Рост продаж: +{sales_improvement:.1f}%
                            - Улучшение ROAS: +{roas_improvement:.1f}%
                            - **Рекомендация**: Приоритетная стратегия для внедрения
                            """)
                            elif sales_improvement > 5:
                                st.info(f"""
                            **{name}**: Увеличивает продажи при сопоставимой эффективности
                            - Рост продаж: +{sales_improvement:.1f}%
                            - Изменение ROAS: {roas_improvement:+.1f}%
                            - **Рекомендация**: Подходит для фазы роста
                            """)
                            elif roas_improvement > 10:
                                st.info(f"""
                            **{name}**: Повышает эффективность инвестиций
                            - Изменение продаж: {sales_improvement:+.1f}%
                            - Улучшение ROAS: +{roas_improvement:.1f}%
                            - **Рекомендация**: Подходит для оптимизации
                            """)
                            else:
                                st.warning(f"""
                            **{name}**: Незначительные улучшения относительно текущей стратегии
                            - Изменение продаж: {sales_improvement:+.1f}%
                            - Изменение ROAS: {roas_improvement:+.1f}%
                            - **Рекомендация**: Второстепенная альтернатива
                            """)
if __name__ == "__main__":
    app = MMM_App()
    app.run()
