import React, { useMemo } from 'react';
import dynamic from 'next/dynamic';
import { formatCurrency } from '../../lib/utils';
import LoadingSpinner from '../ui/LoadingSpinner';

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), {
  ssr: false,
  loading: () => <div className="flex justify-center p-8"><LoadingSpinner size="large" /></div>
});

interface PortfolioBalance {
  symbol: string;
  balance: number;
  usd_value: number;
  wallet_address?: string;
}

interface PortfolioChartProps {
  balances: PortfolioBalance[];
}

export default function PortfolioChart({ balances }: PortfolioChartProps) {
  const chartData = useMemo(() => {
    if (!balances || balances.length === 0) return [];

    const totalValue = balances.reduce((sum, balance) => sum + (balance.usd_value || 0), 0);

    // Prepare data for pie chart
    const labels = balances.map(balance => balance.symbol.toUpperCase());
    const values = balances.map(balance => balance.usd_value || 0);
    const percentages = balances.map(balance => ((balance.usd_value || 0) / totalValue) * 100);

    // Generate colors - you can customize this
    const colors = [
      '#8B5CF6', '#06B6D4', '#10B981', '#F59E0B', '#EF4444',
      '#EC4899', '#6366F1', '#84CC16', '#F97316', '#14B8A6'
    ];

    const hoverText = balances.map((balance, idx) =>
      `${balance.symbol.toUpperCase()}<br>` +
      `${formatCurrency(balance.usd_value)}<br>` +
      `${percentages[idx].toFixed(1)}%<br>` +
      `${balance.balance.toFixed(6)} tokens`
    );

    return [{
      type: 'pie',
      labels,
      values,
      hovertemplate: '%{hovertext}<extra></extra>',
      hovertext: hoverText,
      textinfo: 'label+percent',
      textposition: 'auto',
      marker: {
        colors: colors.slice(0, balances.length),
        line: {
          color: '#ffffff',
          width: 2
        }
      },
      hole: 0.4, // Creates a donut chart
    }];
  }, [balances]);

  const layout = {
    title: {
      text: '',
      font: { size: 16, color: '#111827' }
    },
    showlegend: true,
    legend: {
      orientation: 'v',
      x: 1.02,
      y: 0.5,
      font: { size: 12 }
    },
    margin: { t: 20, r: 120, b: 20, l: 20 },
    plot_bgcolor: 'transparent',
    paper_bgcolor: 'transparent',
    height: 400,
    annotations: [{
      font: {
        size: 16,
        color: '#111827'
      },
      showarrow: false,
      text: `Total<br>${formatCurrency(balances.reduce((sum, b) => sum + (b.usd_value || 0), 0))}`,
      x: 0.5,
      y: 0.5
    }]
  };

  const config = {
    responsive: true,
    displayModeBar: false,
  };

  if (!balances || balances.length === 0) {
    return (
      <div className="text-center py-8">
        <p className="text-gray-500">No portfolio data to display</p>
      </div>
    );
  }

  return (
    <div className="w-full">
      <Plot
        data={chartData}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '400px' }}
      />

      {/* Additional Portfolio Details */}
      <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div className="text-center">
          <div className="text-gray-500">Assets</div>
          <div className="font-semibold text-lg">{balances.length}</div>
        </div>
        <div className="text-center">
          <div className="text-gray-500">Largest Position</div>
          <div className="font-semibold text-lg">
            {balances.length > 0 ?
              `${((Math.max(...balances.map(b => b.usd_value || 0)) / balances.reduce((sum, b) => sum + (b.usd_value || 0), 0)) * 100).toFixed(1)}%`
              : '0%'
            }
          </div>
        </div>
        <div className="text-center">
          <div className="text-gray-500">Total Tokens</div>
          <div className="font-semibold text-lg">
            {balances.reduce((sum, b) => sum + b.balance, 0).toFixed(2)}
          </div>
        </div>
        <div className="text-center">
          <div className="text-gray-500">Avg. Position</div>
          <div className="font-semibold text-lg">
            {formatCurrency(balances.reduce((sum, b) => sum + (b.usd_value || 0), 0) / balances.length)}
          </div>
        </div>
      </div>
    </div>
  );
}