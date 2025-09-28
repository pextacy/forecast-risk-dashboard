import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  BriefcaseIcon,
  CurrencyDollarIcon,
  PlusIcon,
  TrashIcon,
  ChartPieIcon,
  CalculatorIcon,
  TrendingUpIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';
import LoadingSpinner from '../ui/LoadingSpinner';
import PortfolioChart from './PortfolioChart';
import { api, handleApiError } from '../../lib/api';
import { formatCurrency, formatPercentage, formatDate, getRiskLevelColor } from '../../lib/utils';

interface PortfolioBalance {
  symbol: string;
  balance: number;
  usd_value: number;
  wallet_address?: string;
}

interface RiskMetrics {
  portfolio_value: number;
  var_95: {
    historical: number;
    parametric: number;
  };
  expected_shortfall: number;
  sharpe_ratio: number;
  volatility_30d: number;
  max_drawdown: number;
  beta: number;
  concentration_ratio: number;
  risk_level: string;
}

export default function PortfolioPanel() {
  const [balances, setBalances] = useState<PortfolioBalance[]>([]);
  const [newBalance, setNewBalance] = useState({ symbol: '', balance: '', wallet_address: '' });
  const [isLoading, setIsLoading] = useState(false);
  const [isAddingBalance, setIsAddingBalance] = useState(false);
  const [supportedAssets, setSupportedAssets] = useState([]);
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics | null>(null);
  const [performanceData, setPerformanceData] = useState(null);
  const [isCalculatingRisk, setIsCalculatingRisk] = useState(false);
  const [walletAddress, setWalletAddress] = useState('');

  useEffect(() => {
    loadSupportedAssets();
    loadCurrentBalances();
  }, []);

  const loadSupportedAssets = async () => {
    try {
      const response = await api.forecasts.getSupportedAssets();
      const allAssets = [
        ...response.data.crypto_assets.map(id => ({ id, name: id.toUpperCase(), type: 'crypto' })),
        ...response.data.stock_assets.map(id => ({ id, name: id, type: 'stock' }))
      ];
      setSupportedAssets(allAssets);
    } catch (error) {
      console.error('Failed to load supported assets:', error);
    }
  };

  const loadCurrentBalances = async () => {
    setIsLoading(true);
    try {
      const response = await api.portfolio.getCurrent();
      setBalances(response.data.balances || []);

      if (response.data.balances && response.data.balances.length > 0) {
        await calculateRiskMetrics(response.data.balances);
      }
    } catch (error) {
      console.error('Failed to load current balances:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const calculateRiskMetrics = async (currentBalances = balances) => {
    if (currentBalances.length === 0) return;

    setIsCalculatingRisk(true);
    try {
      // Convert balances to portfolio weights
      const totalValue = currentBalances.reduce((sum, b) => sum + (b.usd_value || 0), 0);
      const portfolioWeights = {};
      currentBalances.forEach(balance => {
        portfolioWeights[balance.symbol] = (balance.usd_value || 0) / totalValue;
      });

      const response = await api.portfolio.getRiskMetrics({ portfolio_weights: portfolioWeights });
      setRiskMetrics(response.data);

      // Also load performance data
      const performanceResponse = await api.portfolio.getPerformance(portfolioWeights, 30);
      setPerformanceData(performanceResponse.data);

    } catch (error) {
      console.error('Failed to calculate risk metrics:', error);
      toast.error('Failed to calculate risk metrics');
    } finally {
      setIsCalculatingRisk(false);
    }
  };

  const handleAddBalance = async () => {
    if (!newBalance.symbol || !newBalance.balance) {
      toast.error('Please enter symbol and balance');
      return;
    }

    setIsAddingBalance(true);
    try {
      const response = await api.portfolio.updateBalances({
        balances: [{
          symbol: newBalance.symbol,
          balance: parseFloat(newBalance.balance),
          wallet_address: newBalance.wallet_address || 'default'
        }]
      });

      toast.success('Balance added successfully');
      setNewBalance({ symbol: '', balance: '', wallet_address: '' });
      await loadCurrentBalances();

    } catch (error) {
      const errorMessage = handleApiError(error, 'Failed to add balance');
      toast.error(errorMessage);
    } finally {
      setIsAddingBalance(false);
    }
  };

  const handleRemoveBalance = async (symbol: string, walletAddress: string = 'default') => {
    try {
      // Set balance to 0 to remove
      await api.portfolio.updateBalances({
        balances: [{
          symbol,
          balance: 0,
          wallet_address: walletAddress
        }]
      });

      toast.success('Balance removed successfully');
      await loadCurrentBalances();

    } catch (error) {
      const errorMessage = handleApiError(error, 'Failed to remove balance');
      toast.error(errorMessage);
    }
  };

  const handleOptimizePortfolio = async () => {
    if (balances.length < 2) {
      toast.error('Need at least 2 assets for optimization');
      return;
    }

    try {
      const totalValue = balances.reduce((sum, b) => sum + (b.usd_value || 0), 0);
      const portfolioWeights = {};
      balances.forEach(balance => {
        portfolioWeights[balance.symbol] = (balance.usd_value || 0) / totalValue;
      });

      const response = await api.portfolio.optimize({
        current_portfolio: portfolioWeights,
        optimization_type: 'sharpe',
        constraints: {
          max_weight: 0.4,
          min_weight: 0.05
        }
      });

      toast.success('Portfolio optimization completed');
      // Handle optimization results (could show in a modal or separate section)
      console.log('Optimization results:', response.data);

    } catch (error) {
      const errorMessage = handleApiError(error, 'Failed to optimize portfolio');
      toast.error(errorMessage);
    }
  };

  const totalPortfolioValue = balances.reduce((sum, balance) => sum + (balance.usd_value || 0), 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 flex items-center">
            <BriefcaseIcon className="h-7 w-7 mr-2 text-purple-600" />
            Portfolio Management
          </h2>
          <p className="text-gray-600 mt-1">
            Track portfolio performance, allocation, and optimization suggestions
          </p>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-gray-900">
            {formatCurrency(totalPortfolioValue)}
          </div>
          <div className="text-sm text-gray-500">Total Portfolio Value</div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Portfolio Overview */}
        <div className="xl:col-span-2 space-y-6">
          {/* Add Balance Form */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-lg shadow-soft border border-gray-200 p-6"
          >
            <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
              <PlusIcon className="h-5 w-5 mr-2 text-green-600" />
              Add Asset
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Asset</label>
                <select
                  value={newBalance.symbol}
                  onChange={(e) => setNewBalance({ ...newBalance, symbol: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                >
                  <option value="">Select Asset</option>
                  {supportedAssets.map((asset) => (
                    <option key={asset.id} value={asset.id}>
                      {asset.name} ({asset.type})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Balance</label>
                <input
                  type="number"
                  step="any"
                  value={newBalance.balance}
                  onChange={(e) => setNewBalance({ ...newBalance, balance: e.target.value })}
                  placeholder="0.00"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Wallet (Optional)</label>
                <input
                  type="text"
                  value={newBalance.wallet_address}
                  onChange={(e) => setNewBalance({ ...newBalance, wallet_address: e.target.value })}
                  placeholder="default"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                />
              </div>

              <div className="flex items-end">
                <button
                  onClick={handleAddBalance}
                  disabled={isAddingBalance || !newBalance.symbol || !newBalance.balance}
                  className="w-full flex items-center justify-center px-4 py-2 bg-purple-600 text-white font-medium rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {isAddingBalance ? (
                    <LoadingSpinner size="small" color="white" className="mr-2" />
                  ) : (
                    <PlusIcon className="h-4 w-4 mr-2" />
                  )}
                  Add
                </button>
              </div>
            </div>
          </motion.div>

          {/* Current Balances */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-white rounded-lg shadow-soft border border-gray-200 p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900 flex items-center">
                <ChartPieIcon className="h-5 w-5 mr-2 text-blue-600" />
                Current Holdings
              </h3>
              <div className="flex space-x-2">
                <button
                  onClick={() => calculateRiskMetrics()}
                  disabled={isCalculatingRisk || balances.length === 0}
                  className="text-sm px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                >
                  {isCalculatingRisk ? 'Calculating...' : 'Analyze Risk'}
                </button>
                <button
                  onClick={handleOptimizePortfolio}
                  disabled={balances.length < 2}
                  className="text-sm px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
                >
                  Optimize
                </button>
              </div>
            </div>

            {isLoading ? (
              <div className="flex justify-center py-8">
                <LoadingSpinner size="large" />
              </div>
            ) : balances.length > 0 ? (
              <div className="space-y-3">
                {balances.map((balance, idx) => {
                  const percentage = totalPortfolioValue > 0 ? (balance.usd_value / totalPortfolioValue) * 100 : 0;

                  return (
                    <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center">
                          <span className="text-purple-600 font-semibold text-sm">
                            {balance.symbol.charAt(0).toUpperCase()}
                          </span>
                        </div>
                        <div>
                          <div className="font-medium text-gray-900">{balance.symbol.toUpperCase()}</div>
                          <div className="text-sm text-gray-500">
                            {balance.balance.toFixed(6)} tokens
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center space-x-4">
                        <div className="text-right">
                          <div className="font-medium text-gray-900">
                            {formatCurrency(balance.usd_value)}
                          </div>
                          <div className="text-sm text-gray-500">
                            {percentage.toFixed(1)}%
                          </div>
                        </div>
                        <button
                          onClick={() => handleRemoveBalance(balance.symbol, balance.wallet_address)}
                          className="text-red-600 hover:text-red-800 p-1"
                        >
                          <TrashIcon className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <BriefcaseIcon className="mx-auto h-12 w-12 text-gray-300 mb-3" />
                <p>No assets in portfolio. Add your first asset above.</p>
              </div>
            )}
          </motion.div>

          {/* Portfolio Chart */}
          {balances.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-white rounded-lg shadow-soft border border-gray-200 p-6"
            >
              <h3 className="text-lg font-medium text-gray-900 mb-4">Portfolio Allocation</h3>
              <PortfolioChart balances={balances} />
            </motion.div>
          )}
        </div>

        {/* Risk Metrics & Performance */}
        <div className="space-y-6">
          {/* Risk Metrics */}
          {riskMetrics && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white rounded-lg shadow-soft border border-gray-200 p-6"
            >
              <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                <ExclamationTriangleIcon className="h-5 w-5 mr-2 text-yellow-600" />
                Risk Analysis
              </h3>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Risk Level</span>
                  <span className={`text-sm px-2 py-1 rounded border ${getRiskLevelColor(riskMetrics.risk_level)}`}>
                    {riskMetrics.risk_level.toUpperCase()}
                  </span>
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">VaR (95%)</span>
                    <span className="font-medium">{formatPercentage(Math.abs(riskMetrics.var_95.historical * 100))}</span>
                  </div>
                  <div className="text-xs text-gray-500">Maximum 1-day loss with 95% confidence</div>
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Expected Shortfall</span>
                    <span className="font-medium">{formatPercentage(Math.abs(riskMetrics.expected_shortfall * 100))}</span>
                  </div>
                  <div className="text-xs text-gray-500">Average loss beyond VaR</div>
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Sharpe Ratio</span>
                    <span className="font-medium">{riskMetrics.sharpe_ratio.toFixed(2)}</span>
                  </div>
                  <div className="text-xs text-gray-500">Risk-adjusted returns</div>
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">30-Day Volatility</span>
                    <span className="font-medium">{formatPercentage(riskMetrics.volatility_30d * 100)}</span>
                  </div>
                  <div className="text-xs text-gray-500">Annualized volatility</div>
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Max Drawdown</span>
                    <span className="font-medium text-red-600">{formatPercentage(Math.abs(riskMetrics.max_drawdown * 100))}</span>
                  </div>
                  <div className="text-xs text-gray-500">Largest peak-to-trough decline</div>
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Concentration</span>
                    <span className="font-medium">{formatPercentage(riskMetrics.concentration_ratio * 100)}</span>
                  </div>
                  <div className="text-xs text-gray-500">Largest single position</div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Performance Metrics */}
          {performanceData && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-white rounded-lg shadow-soft border border-gray-200 p-6"
            >
              <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                <TrendingUpIcon className="h-5 w-5 mr-2 text-green-600" />
                Performance (30D)
              </h3>

              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Total Return</span>
                  <span className={`text-sm font-medium ${performanceData.total_return >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {performanceData.total_return >= 0 ? '+' : ''}{formatPercentage(performanceData.total_return * 100)}
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Annualized Return</span>
                  <span className={`text-sm font-medium ${performanceData.annualized_return >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {performanceData.annualized_return >= 0 ? '+' : ''}{formatPercentage(performanceData.annualized_return * 100)}
                  </span>
                </div>

                {performanceData.benchmark_return && (
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">vs Benchmark</span>
                    <span className={`text-sm font-medium ${performanceData.excess_return >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {performanceData.excess_return >= 0 ? '+' : ''}{formatPercentage(performanceData.excess_return * 100)}
                    </span>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}