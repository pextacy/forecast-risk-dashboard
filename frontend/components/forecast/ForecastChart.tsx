import React, { useMemo } from 'react';
import dynamic from 'next/dynamic';
import { formatCurrency, formatDate } from '../../lib/utils';
import LoadingSpinner from '../ui/LoadingSpinner';

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), {
  ssr: false,
  loading: () => <div className="flex justify-center p-8"><LoadingSpinner size="large" /></div>
});

interface ForecastData {
  symbol: string;
  forecast_horizon: number;
  generated_at: string;
  forecasts: {
    [modelName: string]: {
      forecast: number[];
      confidence_intervals?: {
        [level: string]: {
          lower: number[];
          upper: number[];
        };
      };
      metrics?: {
        mae?: number;
        rmse?: number;
        mape?: number;
      };
    };
  };
  data_points_used: number;
  last_price: number;
}

interface ForecastChartProps {
  data: ForecastData;
  historicalData?: Array<{
    time: string;
    price: number;
  }>;
  selectedModel?: string;
}

export default function ForecastChart({ data, historicalData = [], selectedModel }: ForecastChartProps) {
  const chartData = useMemo(() => {
    if (!data?.forecasts) return [];

    const traces: any[] = [];
    const modelToUse = selectedModel || 'ensemble' || Object.keys(data.forecasts)[0];
    const forecastData = data.forecasts[modelToUse];

    if (!forecastData?.forecast) return [];

    // Historical data trace
    if (historicalData.length > 0) {
      traces.push({
        x: historicalData.map(d => d.time),
        y: historicalData.map(d => d.price),
        type: 'scatter',
        mode: 'lines',
        name: 'Historical Prices',
        line: { color: '#6366f1', width: 2 },
        hovertemplate: '<b>%{x}</b><br>Price: $%{y:,.2f}<extra></extra>',
      });
    }

    // Generate forecast dates
    const startDate = new Date();
    const forecastDates = Array.from({ length: data.forecast_horizon }, (_, i) => {
      const date = new Date(startDate);
      date.setDate(startDate.getDate() + i + 1);
      return date.toISOString().split('T')[0];
    });

    // Forecast line
    traces.push({
      x: forecastDates,
      y: forecastData.forecast,
      type: 'scatter',
      mode: 'lines+markers',
      name: `${modelToUse.toUpperCase()} Forecast`,
      line: { color: '#10b981', width: 3, dash: 'dash' },
      marker: { size: 6, color: '#10b981' },
      hovertemplate: '<b>%{x}</b><br>Predicted: $%{y:,.2f}<extra></extra>',
    });

    // Add confidence intervals if available
    if (forecastData.confidence_intervals) {
      Object.entries(forecastData.confidence_intervals).forEach(([level, intervals]) => {
        const opacity = level === '95%' ? 0.3 : 0.2;
        const color = level === '95%' ? '#10b981' : '#6366f1';

        // Upper bound
        traces.push({
          x: forecastDates,
          y: intervals.upper,
          type: 'scatter',
          mode: 'lines',
          name: `${level} Upper`,
          line: { color: color, width: 1 },
          showlegend: false,
          hoverinfo: 'skip',
        });

        // Lower bound with fill
        traces.push({
          x: forecastDates,
          y: intervals.lower,
          type: 'scatter',
          mode: 'lines',
          name: `${level} Confidence`,
          line: { color: color, width: 1 },
          fill: 'tonexty',
          fillcolor: color.replace(')', `, ${opacity})`).replace('rgb', 'rgba'),
          hovertemplate: `<b>%{x}</b><br>${level} Range: $%{y:,.2f}<extra></extra>`,
        });
      });
    }

    // Add current price marker
    if (historicalData.length > 0) {
      const lastHistoricalDate = historicalData[historicalData.length - 1].time;
      traces.push({
        x: [lastHistoricalDate, forecastDates[0]],
        y: [data.last_price, forecastData.forecast[0]],
        type: 'scatter',
        mode: 'lines',
        name: 'Transition',
        line: { color: '#6b7280', width: 2, dash: 'dot' },
        showlegend: false,
        hoverinfo: 'skip',
      });
    }

    return traces;
  }, [data, historicalData, selectedModel]);

  const layout = {
    title: {
      text: `${data.symbol.toUpperCase()} Price Forecast`,
      font: { size: 18, color: '#111827' },
    },
    xaxis: {
      title: 'Date',
      type: 'date',
      showgrid: true,
      gridcolor: '#f3f4f6',
    },
    yaxis: {
      title: 'Price (USD)',
      tickformat: '$,.0f',
      showgrid: true,
      gridcolor: '#f3f4f6',
    },
    plot_bgcolor: 'white',
    paper_bgcolor: 'white',
    showlegend: true,
    legend: {
      orientation: 'h',
      x: 0,
      y: -0.2,
    },
    margin: { t: 50, r: 50, b: 80, l: 80 },
    hovermode: 'x unified',
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
    toImageButtonOptions: {
      format: 'png',
      filename: `${data.symbol}_forecast_${data.generated_at}`,
      height: 500,
      width: 900,
      scale: 2,
    },
  };

  if (!data?.forecasts || Object.keys(data.forecasts).length === 0) {
    return (
      <div className="text-center py-8">
        <p className="text-gray-500">No forecast data available</p>
      </div>
    );
  }

  return (
    <div className="w-full h-96">
      <Plot
        data={chartData}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}