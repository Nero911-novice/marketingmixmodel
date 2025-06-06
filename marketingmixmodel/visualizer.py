import numpy as np
import pandas as pd
import plotly.graph_objects as go

class Visualizer:
    """Класс для создания визуализаций результатов Marketing Mix Model."""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8'
        }
        
        self.media_colors = {
            'facebook': '#1877f2',
            'google': '#4285f4',
            'tiktok': '#000000',
            'youtube': '#ff0000',
            'instagram': '#e4405f',
            'offline': '#6c757d',
            'base': '#343a40'
        }
        
    def create_waterfall_chart(self, contributions, title="Декомпозиция продаж по каналам"):
        """Создание waterfall диаграммы для визуализации вкладов каналов."""
        # Проверка входных данных
        if not contributions or len(contributions) == 0:
            # Создаем простой bar chart если нет данных для waterfall
            fig = go.Figure()
            fig.add_annotation(
                text="Нет данных для отображения",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title=title, height=400)
            return fig
        
        # Подготовка данных
        channels = list(contributions.keys())
        values = list(contributions.values())
        
        # Проверка на корректность значений
        values = [float(v) if v is not None and not np.isnan(float(v)) else 0 for v in values]
        
        # Сортировка по убыванию (исключая Base)
        if 'Base' in contributions:
            base_value = contributions['Base']
            other_contributions = {k: v for k, v in contributions.items() if k != 'Base'}
            sorted_others = sorted(other_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            
            channels = ['Base'] + [item[0] for item in sorted_others]
            values = [base_value] + [item[1] for item in sorted_others]
        else:
            sorted_items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            channels = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]
        
        # Создание цветов
        colors = []
        for channel in channels:
            channel_lower = str(channel).lower()
            if any(key in channel_lower for key in self.media_colors.keys()):
                # Найти подходящий цвет
                for key, color in self.media_colors.items():
                    if key in channel_lower:
                        colors.append(color)
                        break
                else:
                    colors.append(self.color_palette['primary'])
            else:
                colors.append(self.color_palette['primary'])
        
        try:
            # Создание waterfall графика
            fig = go.Figure(go.Waterfall(
                name="Вклады",
                orientation="v",
                measure=["absolute"] + ["relative"] * (len(channels) - 1),
                x=channels,
                y=values,
                text=[f"{val:,.0f}" for val in values],
                textposition="outside",
                connector={"line": {"color": "gray"}},
                marker_color=colors
            ))
            
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                showlegend=False,
                xaxis_title="Каналы",
                yaxis_title="Вклад в продажи",
                height=500,
                template="plotly_white"
            )
            
        except Exception as e:
            # Fallback к обычному bar chart если waterfall не работает
            fig = go.Figure(data=[
                go.Bar(x=channels, y=values, marker_color=colors,
                       text=[f"{val:,.0f}" for val in values], textposition='outside')
            ])
            
            fig.update_layout(
                title={
                    'text': title + " (Bar Chart)",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title="Каналы",
                yaxis_title="Вклад в продажи",
                height=500,
                template="plotly_white"
            )
        
        return fig
    
    def create_roas_comparison(self, roas_data, title="ROAS по каналам"):
        """Создание сравнительной диаграммы ROAS."""
        try:
            # Проверка входных данных
            if roas_data is None or roas_data.empty or 'ROAS' not in roas_data.columns or 'Channel' not in roas_data.columns:
                # Создаем пустой график с сообщением
                fig = go.Figure()
                fig.add_annotation(
                    text="Нет данных для отображения ROAS",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                fig.update_layout(title=title, height=400)
                return fig
            
            # Сортировка по ROAS
            roas_sorted = roas_data.sort_values('ROAS', ascending=True)
            
            # Создание цветов
            colors = []
            for channel in roas_sorted['Channel']:
                channel_lower = str(channel).lower()
                if any(key in channel_lower for key in self.media_colors.keys()):
                    # Найти подходящий цвет
                    for key, color in self.media_colors.items():
                        if key in channel_lower:
                            colors.append(color)
                            break
                    else:
                        # Цвет по ROAS если не найден специфический
                        roas_val = roas_sorted[roas_sorted['Channel'] == channel]['ROAS'].iloc[0]
                        if roas_val >= 3:
                            colors.append(self.color_palette['success'])
                        elif roas_val >= 1:
                            colors.append(self.color_palette['warning'])
                        else:
                            colors.append(self.color_palette['danger'])
                else:
                    # Цвет по ROAS
                    roas_val = roas_sorted[roas_sorted['Channel'] == channel]['ROAS'].iloc[0]
                    if roas_val >= 3:
                        colors.append(self.color_palette['success'])
                    elif roas_val >= 1:
                        colors.append(self.color_palette['warning'])
                    else:
                        colors.append(self.color_palette['danger'])
            
            fig = go.Figure(data=[
                go.Bar(
                    x=roas_sorted['ROAS'],
                    y=roas_sorted['Channel'],
                    orientation='h',
                    marker_color=colors,
                    text=[f"{val:.2f}" for val in roas_sorted['ROAS']],
                    textposition='outside'
                )
            ])
            
            fig.add_vline(
                x=1, 
                line_dash="dash", 
                line_color="red",
                annotation_text="Точка безубыточности",
                annotation_position="top right"
            )
            
            fig.update_layout(
                title={'text': title, 'x': 0.5, 'xanchor': 'center'},
                xaxis_title="ROAS",
                yaxis_title="Каналы",
                height=400,
                template="plotly_white",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            # Fallback к простому графику
            fig = go.Figure()
            fig.add_annotation(
                text=f"Ошибка создания графика: {str(e)[:50]}...",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title=title, height=400)
            return fig
    
    def create_budget_allocation_pie(self, budget_data, title="Распределение бюджета"):
        """Создание круговой диаграммы распределения бюджета."""
        channels = list(budget_data.keys())
        values = list(budget_data.values())
        total_budget = sum(values)
        
        colors = []
        for channel in channels:
            channel_lower = channel.lower()
            colors.append(self.media_colors.get(channel_lower, self.color_palette['primary']))
        
        fig = go.Figure(data=[go.Pie(
            labels=channels,
            values=values,
            marker_colors=colors,
            textinfo='label+percent',
            textposition='outside',
            hovertemplate="<b>%{label}</b><br>Бюджет: %{value:,.0f}<br>Доля: %{percent}<extra></extra>"
        )])
        
        fig.update_layout(
            title={
                'text': f"{title}<br><sub>Общий бюджет: {total_budget:,.0f}</sub>",
                'x': 0.5,
                'xanchor': 'center'
            },
            height=500,
            template="plotly_white"
        )
        
        return fig
    
    def create_optimization_results(self, current_allocation, optimal_allocation, 
                                  title="Результаты оптимизации бюджета"):
        """Создание сравнения текущего и оптимального распределения."""
        channels = list(current_allocation.keys())
        current_values = [current_allocation[ch] for ch in channels]
        optimal_values = [optimal_allocation.get(ch, 0) for ch in channels]
        
        fig = go.Figure()
        
        # Текущее распределение
        fig.add_trace(go.Bar(
            name='Текущее',
            x=channels,
            y=current_values,
            marker_color=self.color_palette['info'],
            opacity=0.7
        ))
        
        # Оптимальное распределение
        fig.add_trace(go.Bar(
            name='Оптимальное',
            x=channels,
            y=optimal_values,
            marker_color=self.color_palette['success']
        ))
        
        # Расчет изменений
        for i, channel in enumerate(channels):
            change = optimal_values[i] - current_values[i]
            change_pct = (change / current_values[i] * 100) if current_values[i] > 0 else 0
            
            fig.add_annotation(
                x=i,
                y=max(optimal_values[i], current_values[i]) + max(optimal_values) * 0.05,
                text=f"{change_pct:+.1f}%",
                showarrow=False,
                font=dict(
                    size=10,
                    color=self.color_palette['success'] if change > 0 else self.color_palette['danger']
                )
            )
        
        fig.update_layout(
            title={'text': title, 'x': 0.5, 'xanchor': 'center'},
            xaxis_title="Каналы",
            yaxis_title="Бюджет",
            barmode='group',
            height=500,
            template="plotly_white"
        )
        
        return fig

