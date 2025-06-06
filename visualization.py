import numpy as np
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
        if not contributions or len(contributions) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="Нет данных для отображения",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title=title, height=400)
            return fig

        channels = list(contributions.keys())
        values = list(contributions.values())

        values = [float(v) if v is not None and not np.isnan(float(v)) else 0 for v in values]

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

        colors = []
        for channel in channels:
            channel_lower = str(channel).lower()
            if any(key in channel_lower for key in self.media_colors.keys()):
                for key, color in self.media_colors.items():
                    if key in channel_lower:
                        colors.append(color)
                        break
            else:
                colors.append(self.color_palette['info'])

        fig = go.Figure(go.Waterfall(
            name="", orientation="v",
            measure=["relative"] * len(channels),
            x=channels, text=[f"{v:,.0f}" for v in values],
            y=values,
            connector={"line": {"color": "rgba(63, 63, 63, 0.7)"}},
            decreasing={"marker": {"color": self.color_palette['danger']}},
            increasing={"marker": {"color": self.color_palette['success']}},
            totals={"marker": {"color": self.color_palette['primary']}}
        ))

        fig.update_layout(title=title, height=400)
        return fig

    def create_budget_allocation_chart(self, budget_data, title="Распределение бюджета"):
        """Создание столбчатой диаграммы распределения бюджета."""
        if not budget_data:
            fig = go.Figure()
            fig.add_annotation(
                text="Нет данных для отображения",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title=title, height=400)
            return fig

        channels = list(budget_data.keys())
        values = list(budget_data.values())

        colors = []
        for channel in channels:
            channel_lower = str(channel).lower()
            if any(key in channel_lower for key in self.media_colors.keys()):
                for key, color in self.media_colors.items():
                    if key in channel_lower:
                        colors.append(color)
                        break
            else:
                colors.append(self.color_palette['info'])

        fig = go.Figure(data=[
            go.Bar(x=channels, y=values, marker_color=colors, text=[f"{v:,.0f}" for v in values])
        ])
        fig.update_layout(title=title, xaxis_title="Каналы", yaxis_title="Бюджет", template="plotly_white")
        return fig

    def compare_budget_allocations(self, current_allocation, optimal_allocation, title="Сравнение бюджетов"):
        """Сравнение текущего и оптимального распределения бюджета."""
        channels = list(current_allocation.keys())
        current_values = list(current_allocation.values())
        optimal_values = [optimal_allocation.get(ch, 0) for ch in channels]

        colors_current = [self.media_colors.get(ch.split('_')[0], self.color_palette['info']) for ch in channels]
        colors_optimal = colors_current

        fig = go.Figure(data=[
            go.Bar(name='Текущий', x=channels, y=current_values, marker_color=colors_current),
            go.Bar(name='Оптимальный', x=channels, y=optimal_values, marker_color=colors_optimal)
        ])

        for i, (curr, opt) in enumerate(zip(current_values, optimal_values)):
            change = opt - curr
            change_pct = (change / curr) * 100 if curr != 0 else 0
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
            template="plotly_white",
        )

        return fig
