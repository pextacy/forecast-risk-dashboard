import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  ShieldCheckIcon,
  CogIcon,
  ChartBarIcon,
  LightBulbIcon,
  PlayIcon,
  ArrowPathIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';
import LoadingSpinner from '../ui/LoadingSpinner';
import { api, handleApiError } from '../../lib/api';
import { formatCurrency, formatPercentage, formatDate, getRiskLevelColor } from '../../lib/utils';

interface HedgeSuggestion {
  id: number;
  risk_level: string;
  suggestion_type: string;
  current_allocation: { [key: string]: number };
  suggested_allocation: { [key: string]: number };
  rationale: string;
  expected_risk_reduction: number;
  implementation_cost: number;
  confidence_score: number;
  created_at: string;
}

interface SimulationResult {
  current_portfolio: {
    value: number;
    risk_metrics: any;
  };
  suggested_portfolio: {
    value: number;
    risk_metrics: any;
  };
  expected_outcomes: {
    risk_reduction: number;
    return_impact: number;
    implementation_cost: number;
  };
  backtest_results: any;
}

export default function HedgePanel() {
  const [portfolioWeights, setPortfolioWeights] = useState({});
  const [riskTolerance, setRiskTolerance] = useState('medium');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);
  const [hedgeSuggestions, setHedgeSuggestions] = useState<HedgeSuggestion[]>([]);
  const [selectedSuggestion, setSelectedSuggestion] = useState<HedgeSuggestion | null>(null);
  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null);
  const [recentSuggestions, setRecentSuggestions] = useState([]);
  const [supportedAssets, setSupportedAssets] = useState([]);
  const [manualWeights, setManualWeights] = useState({ symbol: '', weight: '' });

  useEffect(() => {
    loadSupportedAssets();
    loadRecentSuggestions();
    loadCurrentPortfolio();
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

  const loadRecentSuggestions = async () => {
    try {
      const response = await api.hedge.getHistory(10);
      setRecentSuggestions(response.data.suggestions || []);
    } catch (error) {
      console.error('Failed to load recent suggestions:', error);
    }
  };

  const loadCurrentPortfolio = async () => {
    try {
      const response = await api.portfolio.getCurrent();
      if (response.data.balances && response.data.balances.length > 0) {
        const totalValue = response.data.balances.reduce((sum, b) => sum + (b.usd_value || 0), 0);
        const weights = {};
        response.data.balances.forEach(balance => {
          weights[balance.symbol] = (balance.usd_value || 0) / totalValue;
        });
        setPortfolioWeights(weights);
      }
    } catch (error) {
      console.error('Failed to load current portfolio:', error);
    }
  };

  const handleAddManualWeight = () => {
    if (!manualWeights.symbol || !manualWeights.weight) {
      toast.error('Please enter symbol and weight');
      return;
    }

    const weight = parseFloat(manualWeights.weight) / 100; // Convert percentage to decimal
    if (weight <= 0 || weight > 1) {
      toast.error('Weight must be between 0.1% and 100%');
      return;
    }

    setPortfolioWeights(prev => ({
      ...prev,
      [manualWeights.symbol]: weight
    }));

    setManualWeights({ symbol: '', weight: '' });
  };

  const removeWeight = (symbol: string) => {
    setPortfolioWeights(prev => {
      const newWeights = { ...prev };
      delete newWeights[symbol];
      return newWeights;
    });
  };

  const normalizeWeights = () => {
    const totalWeight = Object.values(portfolioWeights).reduce((sum: number, weight: number) => sum + weight, 0);
    if (totalWeight === 0) return;

    const normalizedWeights = {};
    Object.entries(portfolioWeights).forEach(([symbol, weight]) => {
      normalizedWeights[symbol] = weight / totalWeight;
    });
    setPortfolioWeights(normalizedWeights);
    toast.success('Weights normalized to 100%');
  };

  const handleGenerateHedgeSuggestions = async () => {
    const totalWeight = Object.values(portfolioWeights).reduce((sum: number, weight: number) => sum + weight, 0);
    if (totalWeight === 0) {
      toast.error('Please add portfolio weights');
      return;
    }

    setIsGenerating(true);
    try {
      const response = await api.hedge.suggest({
        portfolio_weights: portfolioWeights,
        risk_tolerance: riskTolerance,
        constraints: {
          max_single_asset_weight: 0.4,
          min_single_asset_weight: 0.01,
          target_num_assets: Math.max(5, Object.keys(portfolioWeights).length)
        }
      });

      setHedgeSuggestions([response.data]);
      toast.success('Hedge suggestions generated successfully');
      loadRecentSuggestions();

    } catch (error) {
      const errorMessage = handleApiError(error, 'Failed to generate hedge suggestions');
      toast.error(errorMessage);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSimulateSuggestion = async (suggestion: HedgeSuggestion) => {
    setIsSimulating(true);
    setSelectedSuggestion(suggestion);

    try {
      const response = await api.hedge.simulate({
        current_allocation: suggestion.current_allocation,
        suggested_allocation: suggestion.suggested_allocation,
        simulation_days: 90,
        monte_carlo_runs: 1000
      });

      setSimulationResult(response.data);
      toast.success('Simulation completed successfully');

    } catch (error) {
      const errorMessage = handleApiError(error, 'Failed to simulate hedge suggestion');
      toast.error(errorMessage);
    } finally {
      setIsSimulating(false);
    }
  };

  const totalWeight = Object.values(portfolioWeights).reduce((sum: number, weight: number) => sum + weight, 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 flex items-center">
            <ShieldCheckIcon className="h-7 w-7 mr-2 text-green-600" />
            Hedge Suggestions
          </h2>
          <p className="text-gray-600 mt-1">
            AI-powered portfolio rebalancing and hedging strategies to optimize risk-return profile
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Configuration Panel */}
        <div className="xl:col-span-2 space-y-6">
          {/* Portfolio Input */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-lg shadow-soft border border-gray-200 p-6"
          >
            <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
              <ChartBarIcon className="h-5 w-5 mr-2 text-blue-600" />
              Portfolio Configuration
            </h3>

            {/* Add Manual Weight */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-1">Asset</label>
                <select
                  value={manualWeights.symbol}
                  onChange={(e) => setManualWeights({ ...manualWeights, symbol: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
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
                <label className="block text-sm font-medium text-gray-700 mb-1">Weight (%)</label>
                <input
                  type="number"
                  step="0.1"
                  value={manualWeights.weight}
                  onChange={(e) => setManualWeights({ ...manualWeights, weight: e.target.value })}
                  placeholder="0.0"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
                />
              </div>
              <div className="flex items-end">
                <button
                  onClick={handleAddManualWeight}
                  disabled={!manualWeights.symbol || !manualWeights.weight}
                  className="w-full px-4 py-2 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Add
                </button>
              </div>
            </div>

            {/* Current Weights */}
            <div className="mb-6">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-medium text-gray-900">Current Portfolio Weights</h4>
                <div className="flex space-x-2">
                  <button
                    onClick={() => loadCurrentPortfolio()}
                    className="text-sm px-3 py-1 bg-gray-600 text-white rounded hover:bg-gray-700"
                  >
                    Load Current
                  </button>
                  <button
                    onClick={normalizeWeights}
                    disabled={totalWeight === 0}
                    className="text-sm px-3 py-1 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50"
                  >
                    Normalize
                  </button>
                </div>
              </div>

              {Object.keys(portfolioWeights).length > 0 ? (
                <div className="space-y-2">
                  {Object.entries(portfolioWeights).map(([symbol, weight]) => (
                    <div key={symbol} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                      <span className="font-medium">{symbol.toUpperCase()}</span>
                      <div className="flex items-center space-x-2">
                        <span>{formatPercentage(weight * 100)}</span>
                        <button
                          onClick={() => removeWeight(symbol)}
                          className="text-red-600 hover:text-red-800"
                        >
                          ×
                        </button>
                      </div>
                    </div>
                  ))}
                  <div className="text-right text-sm text-gray-500">
                    Total: {formatPercentage(totalWeight * 100)}
                  </div>
                </div>
              ) : (
                <div className="text-center py-4 text-gray-500">
                  No portfolio weights configured. Add assets above or load current portfolio.
                </div>
              )}
            </div>

            {/* Risk Tolerance */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">Risk Tolerance</label>
              <div className="grid grid-cols-3 gap-2">
                {['conservative', 'medium', 'aggressive'].map((level) => (
                  <button
                    key={level}
                    onClick={() => setRiskTolerance(level)}
                    className={`px-3 py-2 text-sm font-medium rounded-lg border ${
                      riskTolerance === level
                        ? 'bg-green-600 text-white border-green-600'
                        : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                    }`}
                  >
                    {level.charAt(0).toUpperCase() + level.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Generate Button */}
            <button
              onClick={handleGenerateHedgeSuggestions}
              disabled={isGenerating || totalWeight === 0}
              className="w-full flex items-center justify-center px-4 py-3 bg-green-600 text-white font-medium rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isGenerating ? (
                <>
                  <LoadingSpinner size="small" color="white" className="mr-2" />
                  Generating Suggestions...
                </>
              ) : (
                <>
                  <LightBulbIcon className="h-5 w-5 mr-2" />
                  Generate Hedge Suggestions
                </>
              )}
            </button>
          </motion.div>

          {/* Hedge Suggestions */}
          {hedgeSuggestions.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-lg shadow-soft border border-gray-200 p-6"
            >
              <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                <ShieldCheckIcon className="h-5 w-5 mr-2 text-green-600" />
                Recommendations
              </h3>

              <div className="space-y-4">
                {hedgeSuggestions.map((suggestion, idx) => (
                  <div key={idx} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <div className="flex items-center space-x-2 mb-1">
                          <span className={`text-sm px-2 py-1 rounded border ${getRiskLevelColor(suggestion.risk_level)}`}>
                            {suggestion.risk_level.toUpperCase()}
                          </span>
                          <span className="text-sm text-gray-500">{suggestion.suggestion_type}</span>
                        </div>
                        <div className="text-xs text-gray-500">
                          Confidence: {formatPercentage(suggestion.confidence_score * 100)}
                        </div>
                      </div>
                      <button
                        onClick={() => handleSimulateSuggestion(suggestion)}
                        disabled={isSimulating}
                        className="flex items-center px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:opacity-50"
                      >
                        {isSimulating && selectedSuggestion?.id === suggestion.id ? (
                          <LoadingSpinner size="small" color="white" className="mr-1" />
                        ) : (
                          <PlayIcon className="h-4 w-4 mr-1" />
                        )}
                        Simulate
                      </button>
                    </div>

                    <div className="text-sm text-gray-700 mb-3">{suggestion.rationale}</div>

                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <span className="text-gray-500">Expected Risk Reduction:</span>
                        <div className="font-medium text-green-600">
                          -{formatPercentage(suggestion.expected_risk_reduction * 100)}
                        </div>
                      </div>
                      <div>
                        <span className="text-gray-500">Implementation Cost:</span>
                        <div className="font-medium">
                          {formatPercentage(suggestion.implementation_cost * 100)}
                        </div>
                      </div>
                      <div>
                        <span className="text-gray-500">Rebalancing Changes:</span>
                        <div className="font-medium">
                          {Object.keys(suggestion.suggested_allocation).length} assets
                        </div>
                      </div>
                    </div>

                    {/* Allocation Changes */}
                    <div className="mt-3 pt-3 border-t border-gray-100">
                      <h5 className="text-sm font-medium text-gray-900 mb-2">Suggested Changes:</h5>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        {Object.entries(suggestion.suggested_allocation).map(([symbol, newWeight]) => {
                          const currentWeight = suggestion.current_allocation[symbol] || 0;
                          const change = newWeight - currentWeight;

                          return (
                            <div key={symbol} className="flex justify-between">
                              <span>{symbol.toUpperCase()}:</span>
                              <span className={change >= 0 ? 'text-green-600' : 'text-red-600'}>
                                {formatPercentage(currentWeight * 100)} → {formatPercentage(newWeight * 100)}
                                ({change >= 0 ? '+' : ''}{formatPercentage(change * 100)})
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </div>

        {/* Simulation Results & Recent Suggestions */}
        <div className="space-y-6">
          {/* Simulation Results */}
          {simulationResult && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white rounded-lg shadow-soft border border-gray-200 p-6"
            >
              <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                <CogIcon className="h-5 w-5 mr-2 text-purple-600" />
                Simulation Results
              </h3>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="text-center p-3 bg-red-50 rounded">
                    <div className="text-red-600 font-medium">Current Portfolio</div>
                    <div className="text-xl font-bold text-red-700">
                      {formatPercentage(simulationResult.current_portfolio.risk_metrics?.volatility * 100 || 0)}
                    </div>
                    <div className="text-xs text-red-600">Risk Level</div>
                  </div>
                  <div className="text-center p-3 bg-green-50 rounded">
                    <div className="text-green-600 font-medium">Suggested Portfolio</div>
                    <div className="text-xl font-bold text-green-700">
                      {formatPercentage(simulationResult.suggested_portfolio.risk_metrics?.volatility * 100 || 0)}
                    </div>
                    <div className="text-xs text-green-600">Risk Level</div>
                  </div>
                </div>

                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Risk Reduction:</span>
                    <span className="font-medium text-green-600">
                      -{formatPercentage(simulationResult.expected_outcomes.risk_reduction * 100)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Return Impact:</span>
                    <span className={`font-medium ${simulationResult.expected_outcomes.return_impact >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {simulationResult.expected_outcomes.return_impact >= 0 ? '+' : ''}{formatPercentage(simulationResult.expected_outcomes.return_impact * 100)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Implementation Cost:</span>
                    <span className="font-medium">
                      {formatPercentage(simulationResult.expected_outcomes.implementation_cost * 100)}
                    </span>
                  </div>
                </div>

                {simulationResult.backtest_results && (
                  <div className="pt-3 border-t border-gray-100">
                    <h6 className="text-sm font-medium text-gray-900 mb-2">Backtest Performance</h6>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="flex justify-between">
                        <span>Sharpe Ratio:</span>
                        <span className="font-medium">{simulationResult.backtest_results.sharpe_ratio?.toFixed(2) || 'N/A'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Max Drawdown:</span>
                        <span className="font-medium text-red-600">
                          {formatPercentage(Math.abs(simulationResult.backtest_results.max_drawdown || 0) * 100)}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {/* Recent Suggestions */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-white rounded-lg shadow-soft border border-gray-200 p-6"
          >
            <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Suggestions</h3>

            {recentSuggestions.length > 0 ? (
              <div className="space-y-3">
                {recentSuggestions.slice(0, 5).map((suggestion, idx) => (
                  <div key={idx} className="text-sm border border-gray-200 rounded p-3">
                    <div className="flex justify-between items-start mb-1">
                      <span className={`text-xs px-2 py-1 rounded ${getRiskLevelColor(suggestion.risk_level)}`}>
                        {suggestion.risk_level?.toUpperCase()}
                      </span>
                      <span className="text-xs text-gray-500">
                        {formatDate(suggestion.created_at, 'relative')}
                      </span>
                    </div>
                    <div className="text-gray-600 text-xs mb-2">
                      {suggestion.suggestion_type} • {formatPercentage((suggestion.confidence_score || 0) * 100)} confidence
                    </div>
                    <div className="text-gray-500 text-xs line-clamp-2">
                      {suggestion.rationale}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-6 text-gray-500">
                <ShieldCheckIcon className="mx-auto h-8 w-8 text-gray-300 mb-2" />
                <p className="text-sm">No recent suggestions. Generate your first hedge strategy above.</p>
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
}