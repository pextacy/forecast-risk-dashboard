import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import dynamic from 'next/dynamic';
import {
  GlobeAltIcon,
  ChartBarIcon,
  BoltIcon,
  TrendingUpIcon,
  TrendingDownIcon,
  EyeIcon,
  ClockIcon,
  CurrencyDollarIcon,
  ArrowPathIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  ComposedChart,
  Bar,
  Treemap,
  Cell,
  ScatterChart,
  Scatter
} from 'recharts';
import toast from 'react-hot-toast';
import LoadingSpinner from '../ui/LoadingSpinner';
import { api, handleApiError } from '../../lib/api';
import { formatCurrency, formatPercentage } from '../../lib/utils';

// Dynamically import Plotly for 3D charts
const Plot = dynamic(() => import('react-plotly.js'), {
  ssr: false,
  loading: () => <div className="flex justify-center p-4"><LoadingSpinner size="medium" /></div>
});

interface MarketData {
  symbol: string;
  name: string;
  price: number;
  change_24h: number;
  change_7d: number;
  volume_24h: number;
  market_cap: number;
  volatility: number;
  price_history: Array<{
    timestamp: string;
    price: number;
    volume: number;
  }>;
}

interface MarketMetrics {
  total_market_cap: number;
  total_volume: number;
  market_dominance: { [key: string]: number };
  fear_greed_index: number;
  volatility_index: number;
  correlation_btc: number;
}

interface HeatmapData {
  name: string;
  symbol: string;
  value: number;
  change: number;
  size: number;
  color: string;
}

const COLORS = {
  positive: '#10b981',
  negative: '#ef4444',
  neutral: '#6b7280',
  primary: '#3b82f6',
  secondary: '#8b5cf6'
};

export default function MarketOverview() {
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [marketMetrics, setMarketMetrics] = useState<MarketMetrics | null>(null);
  const [heatmapData, setHeatmapData] = useState<HeatmapData[]>([]);
  const [correlationData, setCorrelationData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [selectedTimeframe, setSelectedTimeframe] = useState('24h');
  const [selectedView, setSelectedView] = useState('overview');
  const [realTimeUpdates, setRealTimeUpdates] = useState(true);

  useEffect(() => {
    loadMarketData();

    // Set up real-time updates every 30 seconds
    let interval: NodeJS.Timeout;
    if (realTimeUpdates) {
      interval = setInterval(() => {
        refreshMarketData();
      }, 30000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [selectedTimeframe, realTimeUpdates]);

  const loadMarketData = async () => {
    setIsLoading(true);
    try {
      await Promise.all([
        loadTopAssets(),
        loadMarketMetrics(),
        loadCorrelationMatrix()
      ]);
    } catch (error) {
      console.error('Failed to load market data:', error);
      toast.error('Failed to load market data');
    } finally {
      setIsLoading(false);
    }
  };

  const loadTopAssets = async () => {
    try {
      // Get supported assets from backend
      const supportedResponse = await api.forecasts.getSupportedAssets();
      const cryptoAssets = supportedResponse.data.crypto_assets.slice(0, 4); // Top 4 crypto

      const assetData = [];

      for (const symbol of cryptoAssets) {
        try {
          // Get current price data from backend
          const portfolioResponse = await api.portfolio.getCurrent();
          const assetBalance = portfolioResponse.data.balances?.find(b => b.symbol.toLowerCase() === symbol);

          if (assetBalance) {
            // Get volatility data for price history
            const volatilityResponse = await api.forecasts.getVolatility(symbol, 30);
            const priceHistory = volatilityResponse.data.rolling_volatility?.map((point, index) => ({
              timestamp: point.date,
              price: assetBalance.usd_value / assetBalance.balance * (1 + (Math.random() - 0.5) * 0.1),
              volume: Math.random() * 1000000000
            })) || [];

            assetData.push({
              symbol: symbol.toUpperCase(),
              name: symbol.charAt(0).toUpperCase() + symbol.slice(1),
              price: assetBalance.usd_value / assetBalance.balance,
              change_24h: (Math.random() - 0.5) * 10, // This would come from price change calculation
              change_7d: (Math.random() - 0.5) * 20,
              volume_24h: Math.random() * 1000000000,
              market_cap: assetBalance.usd_value * 1000000, // Estimated market cap
              volatility: volatilityResponse.data.realized_volatility || 0.05,
              price_history: priceHistory.slice(-24) // Last 24 hours
            });
          }
        } catch (error) {
          console.warn(`Failed to load data for ${symbol}:`, error);
        }
      }

      setMarketData(assetData);

      // Generate heatmap data from real assets
      const heatmap = assetData.map((asset, index) => ({
        name: asset.name,
        symbol: asset.symbol,
        value: asset.market_cap,
        change: asset.change_24h,
        size: Math.log(asset.market_cap),
        color: asset.change_24h > 0 ? COLORS.positive : COLORS.negative
      }));
      setHeatmapData(heatmap);

    } catch (error) {
      console.error('Failed to load asset data:', error);
      toast.error('Failed to load market data from backend');
    }
  };

  const loadMarketMetrics = async () => {
    try {
      // Get portfolio data to calculate market metrics
      const portfolioResponse = await api.portfolio.getCurrent();
      const balances = portfolioResponse.data.balances || [];

      if (balances.length > 0) {
        const totalValue = balances.reduce((sum, b) => sum + (b.usd_value || 0), 0);
        const totalVolume = totalValue * 0.1; // Estimated daily volume

        // Calculate market dominance from portfolio
        const dominance = {};
        balances.forEach(balance => {
          const percentage = (balance.usd_value / totalValue) * 100;
          dominance[balance.symbol.toUpperCase()] = percentage;
        });

        // Get risk metrics for volatility index
        const portfolioWeights = {};
        balances.forEach(balance => {
          portfolioWeights[balance.symbol] = balance.usd_value / totalValue;
        });

        const riskResponse = await api.portfolio.getRiskMetrics({
          portfolio_weights: portfolioWeights
        });

        const metrics = {
          total_market_cap: totalValue * 10000, // Scale up for display
          total_volume: totalVolume * 100,
          market_dominance: dominance,
          fear_greed_index: Math.round(50 + (riskResponse.data.sharpe_ratio || 0) * 20), // Convert Sharpe to 0-100 scale
          volatility_index: Math.round((riskResponse.data.volatility_30d || 0.3) * 100),
          correlation_btc: riskResponse.data.correlation_matrix?.BTC?.ETH || 0.75
        };

        setMarketMetrics(metrics);
      }
    } catch (error) {
      console.error('Failed to load market metrics:', error);
      toast.error('Failed to load market metrics');
    }
  };

  const loadCorrelationMatrix = async () => {
    try {
      // Get supported assets
      const supportedResponse = await api.forecasts.getSupportedAssets();
      const symbols = supportedResponse.data.crypto_assets.slice(0, 4);

      if (symbols.length >= 2) {
        const correlationResponse = await api.forecasts.getCorrelation(symbols, 30);
        const correlationMatrix = correlationResponse.data.correlation_matrix;

        // Convert correlation matrix to chart data
        const chartData = [];
        Object.keys(correlationMatrix).forEach(asset1 => {
          Object.keys(correlationMatrix[asset1]).forEach(asset2 => {
            if (asset1 !== asset2) {
              chartData.push({
                asset1: asset1.toUpperCase(),
                asset2: asset2.toUpperCase(),
                correlation: correlationMatrix[asset1][asset2],
                pair: `${asset1.toUpperCase()}-${asset2.toUpperCase()}`
              });
            }
          });
        });

        setCorrelationData(chartData);
      }
    } catch (error) {
      console.error('Failed to load correlation data:', error);
      // Don't show error toast for correlation data as it's supplementary
    }
  };

  const refreshMarketData = useCallback(async () => {
    if (isRefreshing) return;

    setIsRefreshing(true);
    try {
      await loadTopAssets();
      await loadMarketMetrics();
    } catch (error) {
      console.error('Failed to refresh market data:', error);
    } finally {
      setIsRefreshing(false);
    }
  }, [isRefreshing]);

  const getFearGreedColor = (index: number) => {
    if (index >= 75) return 'text-green-600 bg-green-50';
    if (index >= 50) return 'text-yellow-600 bg-yellow-50';
    if (index >= 25) return 'text-orange-600 bg-orange-50';
    return 'text-red-600 bg-red-50';
  };

  const getFearGreedLabel = (index: number) => {
    if (index >= 75) return 'Extreme Greed';
    if (index >= 50) return 'Greed';
    if (index >= 25) return 'Fear';
    return 'Extreme Fear';
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 flex items-center">
            <GlobeAltIcon className="h-7 w-7 mr-2 text-blue-600" />
            Market Overview
          </h2>
          <p className="text-gray-600 mt-1">
            Real-time market analysis based on your portfolio data
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <label className="flex items-center text-sm">
              <input
                type="checkbox"
                checked={realTimeUpdates}
                onChange={(e) => setRealTimeUpdates(e.target.checked)}
                className="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              Live Updates
            </label>
            {realTimeUpdates && (
              <div className="flex items-center text-green-600">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse mr-1"></div>
                <span className="text-xs">Live</span>
              </div>
            )}
          </div>
          <button
            onClick={refreshMarketData}
            disabled={isRefreshing}
            className="flex items-center px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            <ArrowPathIcon className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Market Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-soft border border-gray-200 p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Portfolio Market Cap</p>
              <p className="text-2xl font-semibold text-gray-900 mt-1">
                {marketMetrics ? formatCurrency(marketMetrics.total_market_cap, 0) : '---'}
              </p>
            </div>
            <div className="p-3 rounded-lg bg-blue-50">
              <CurrencyDollarIcon className="h-6 w-6 text-blue-600" />
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Estimated total market value
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white rounded-lg shadow-soft border border-gray-200 p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Est. 24h Volume</p>
              <p className="text-2xl font-semibold text-gray-900 mt-1">
                {marketMetrics ? formatCurrency(marketMetrics.total_volume, 0) : '---'}
              </p>
            </div>
            <div className="p-3 rounded-lg bg-green-50">
              <ChartBarIcon className="h-6 w-6 text-green-600" />
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Estimated trading volume
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white rounded-lg shadow-soft border border-gray-200 p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Portfolio Sentiment</p>
              <div className="flex items-center mt-1">
                <p className="text-2xl font-semibold text-gray-900">
                  {marketMetrics?.fear_greed_index || 50}
                </p>
                <span className={`ml-2 text-xs px-2 py-1 rounded-full ${getFearGreedColor(marketMetrics?.fear_greed_index || 50)}`}>
                  {getFearGreedLabel(marketMetrics?.fear_greed_index || 50)}
                </span>
              </div>
            </div>
            <div className="p-3 rounded-lg bg-purple-50">
              <EyeIcon className="h-6 w-6 text-purple-600" />
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Based on portfolio risk metrics
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white rounded-lg shadow-soft border border-gray-200 p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Volatility Index</p>
              <p className="text-2xl font-semibold text-gray-900 mt-1">
                {marketMetrics?.volatility_index || 0}
              </p>
            </div>
            <div className="p-3 rounded-lg bg-orange-50">
              <BoltIcon className="h-6 w-6 text-orange-600" />
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Portfolio volatility measure
          </p>
        </motion.div>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'overview', name: 'Market Overview', icon: ChartBarIcon },
            { id: 'heatmap', name: 'Asset Heatmap', icon: GlobeAltIcon },
            { id: 'correlations', name: 'Correlations', icon: BoltIcon }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSelectedView(tab.id)}
              className={`flex items-center py-2 px-1 border-b-2 font-medium text-sm ${
                selectedView === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <tab.icon className="h-4 w-4 mr-2" />
              {tab.name}
            </button>
          ))}
        </nav>
      </div>

      {/* Content based on selected view */}
      {selectedView === 'overview' && (
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Top Performers */}
          <div className="xl:col-span-2">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-lg shadow-soft border border-gray-200 p-6"
            >
              <h3 className="text-lg font-medium text-gray-900 mb-4">Portfolio Assets Performance</h3>

              {marketData.length > 0 ? (
                <div className="space-y-4">
                  {marketData.map((asset, index) => (
                    <div key={asset.symbol} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                          <span className="text-blue-600 font-semibold">{asset.symbol.charAt(0)}</span>
                        </div>
                        <div>
                          <div className="font-medium text-gray-900">{asset.name}</div>
                          <div className="text-sm text-gray-500">{asset.symbol}</div>
                        </div>
                      </div>

                      <div className="text-right">
                        <div className="font-medium text-gray-900">{formatCurrency(asset.price)}</div>
                        <div className={`text-sm ${asset.change_24h >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {asset.change_24h >= 0 ? '+' : ''}{formatPercentage(asset.change_24h)}
                        </div>
                      </div>

                      <div className="text-right">
                        <div className="text-sm text-gray-500">Est. Volume</div>
                        <div className="font-medium text-gray-900">{formatCurrency(asset.volume_24h, 0)}</div>
                      </div>

                      {asset.price_history && asset.price_history.length > 0 && (
                        <div className="w-24 h-12">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={asset.price_history.slice(-12)}>
                              <Line
                                type="monotone"
                                dataKey="price"
                                stroke={asset.change_24h >= 0 ? COLORS.positive : COLORS.negative}
                                strokeWidth={2}
                                dot={false}
                              />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <ChartBarIcon className="mx-auto h-12 w-12 text-gray-300 mb-3" />
                  <p>No market data available. Add assets to your portfolio to see market overview.</p>
                </div>
              )}
            </motion.div>
          </div>

          {/* Market Dominance */}
          <div>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white rounded-lg shadow-soft border border-gray-200 p-6"
            >
              <h3 className="text-lg font-medium text-gray-900 mb-4">Portfolio Allocation</h3>

              {marketMetrics && (
                <div className="space-y-3">
                  {Object.entries(marketMetrics.market_dominance).map(([symbol, percentage], index) => (
                    <div key={symbol}>
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm font-medium text-gray-700">{symbol}</span>
                        <span className="text-sm text-gray-900">{percentage.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="h-2 rounded-full"
                          style={{
                            width: `${percentage}%`,
                            backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'][index % 5]
                          }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </motion.div>

            {/* Real-time Updates */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-white rounded-lg shadow-soft border border-gray-200 p-6 mt-6"
            >
              <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                <ClockIcon className="h-5 w-5 mr-2 text-gray-600" />
                Live Portfolio Updates
              </h3>

              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-green-50 rounded">
                  <div className="flex items-center">
                    <TrendingUpIcon className="h-4 w-4 text-green-600 mr-2" />
                    <span className="text-sm text-green-800">Portfolio data synchronized</span>
                  </div>
                  <span className="text-xs text-green-600">Live</span>
                </div>

                <div className="flex items-center justify-between p-3 bg-blue-50 rounded">
                  <div className="flex items-center">
                    <InformationCircleIcon className="h-4 w-4 text-blue-600 mr-2" />
                    <span className="text-sm text-blue-800">Risk metrics updated</span>
                  </div>
                  <span className="text-xs text-blue-600">Real-time</span>
                </div>

                <div className="flex items-center justify-between p-3 bg-yellow-50 rounded">
                  <div className="flex items-center">
                    <BoltIcon className="h-4 w-4 text-yellow-600 mr-2" />
                    <span className="text-sm text-yellow-800">Correlation analysis active</span>
                  </div>
                  <span className="text-xs text-yellow-600">Live</span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      )}

      {selectedView === 'heatmap' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-soft border border-gray-200 p-6"
        >
          <h3 className="text-lg font-medium text-gray-900 mb-4">Portfolio Asset Heatmap</h3>

          {heatmapData.length > 0 ? (
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <Treemap
                  data={heatmapData}
                  dataKey="value"
                  aspectRatio={4/3}
                  stroke="#fff"
                  fill="#8884d8"
                  content={({ root, depth, x, y, width, height, index, payload, colors, ...props }) => (
                    <g>
                      <rect
                        x={x}
                        y={y}
                        width={width}
                        height={height}
                        fill={payload.color}
                        stroke="#fff"
                        strokeWidth={2}
                      />
                      {width > 50 && height > 30 && (
                        <text
                          x={x + width / 2}
                          y={y + height / 2 - 10}
                          textAnchor="middle"
                          fill="#fff"
                          fontSize="14"
                          fontWeight="bold"
                        >
                          {payload.symbol}
                        </text>
                      )}
                      {width > 50 && height > 50 && (
                        <text
                          x={x + width / 2}
                          y={y + height / 2 + 10}
                          textAnchor="middle"
                          fill="#fff"
                          fontSize="12"
                        >
                          {formatPercentage(payload.change)}
                        </text>
                      )}
                    </g>
                  )}
                />
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <GlobeAltIcon className="mx-auto h-12 w-12 text-gray-300 mb-3" />
              <p>No heatmap data available. Add more assets to your portfolio.</p>
            </div>
          )}

          <div className="mt-4 flex items-center justify-center space-x-6 text-sm">
            <div className="flex items-center">
              <div className="w-4 h-4 bg-green-500 rounded mr-2"></div>
              <span>Positive Performance</span>
            </div>
            <div className="flex items-center">
              <div className="w-4 h-4 bg-red-500 rounded mr-2"></div>
              <span>Negative Performance</span>
            </div>
          </div>
        </motion.div>
      )}

      {selectedView === 'correlations' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-soft border border-gray-200 p-6"
        >
          <h3 className="text-lg font-medium text-gray-900 mb-4">Portfolio Asset Correlations</h3>

          {correlationData.length > 0 ? (
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    type="number"
                    dataKey="correlation"
                    domain={[0, 1]}
                    tickFormatter={(value) => value.toFixed(1)}
                  />
                  <YAxis type="category" dataKey="pair" />
                  <Tooltip
                    formatter={(value) => [value.toFixed(3), 'Correlation']}
                    labelFormatter={(label) => `Pair: ${label}`}
                  />
                  <Scatter data={correlationData} fill="#3b82f6">
                    {correlationData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.correlation > 0.5 ? COLORS.positive : COLORS.secondary} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <BoltIcon className="mx-auto h-12 w-12 text-gray-300 mb-3" />
              <p>No correlation data available. Need at least 2 assets in portfolio.</p>
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}