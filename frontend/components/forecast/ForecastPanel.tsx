import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ChartBarIcon, CpuChipIcon, TrendingUpIcon, InformationCircleIcon } from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';
import LoadingSpinner from '../ui/LoadingSpinner';
import ForecastChart from './ForecastChart';
import { api, handleApiError } from '../../lib/api';
import { formatCurrency, formatPercentage, formatDate } from '../../lib/utils';

export default function ForecastPanel() {
  const [selectedSymbol, setSelectedSymbol] = useState('bitcoin');
  const [forecastHorizon, setForecastHorizon] = useState(30);
  const [selectedModels, setSelectedModels] = useState(['arima', 'prophet']);
  const [isGenerating, setIsGenerating] = useState(false);
  const [forecastData, setForecastData] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);
  const [supportedAssets, setSupportedAssets] = useState([]);
  const [recentForecasts, setRecentForecasts] = useState([]);
  const [selectedModel, setSelectedModel] = useState('ensemble');
  const [includeExplanation, setIncludeExplanation] = useState(true);
  const [isLoadingData, setIsLoadingData] = useState(false);

  // Load supported assets on component mount
  useEffect(() => {
    loadSupportedAssets();
    loadRecentForecasts();
  }, []);

  const loadSupportedAssets = async () => {
    try {
      const response = await api.forecasts.getSupportedAssets();
      const cryptoAssets = response.data.crypto_assets.map(id => ({
        id,
        name: id.charAt(0).toUpperCase() + id.slice(1),
        type: 'crypto',
        icon: id === 'bitcoin' ? '₿' : id === 'ethereum' ? 'Ξ' : '🪙'
      }));
      const stockAssets = response.data.stock_assets.map(id => ({
        id,
        name: id,
        type: 'stock',
        icon: '📈'
      }));
      setSupportedAssets([...cryptoAssets, ...stockAssets]);
    } catch (error) {
      console.error('Failed to load supported assets:', error);
      toast.error('Failed to load supported assets');
    }
  };

  const loadRecentForecasts = async () => {
    try {
      const response = await api.forecasts.getLatest(6);
      setRecentForecasts(response.data.latest_forecasts);
    } catch (error) {
      console.error('Failed to load recent forecasts:', error);
    }
  };

  const models = [
    { id: 'arima', name: 'ARIMA', description: 'Statistical time series model' },
    { id: 'prophet', name: 'Prophet', description: 'Facebook\'s forecasting tool' },
    { id: 'ensemble', name: 'Ensemble', description: 'Combined model approach' },
  ];

  const handleGenerateForecast = async () => {
    if (selectedModels.length === 0) {
      toast.error('Please select at least one model');
      return;
    }

    setIsGenerating(true);
    setForecastData(null);

    try {
      // Generate the forecast
      const forecastResponse = await api.forecasts.generate({
        symbol: selectedSymbol,
        horizon_days: forecastHorizon,
        models: selectedModels,
        include_explanation: includeExplanation
      });

      setForecastData(forecastResponse.data);

      // Load historical data for visualization
      try {
        const volatilityResponse = await api.forecasts.getVolatility(selectedSymbol, 90);
        if (volatilityResponse.data.volatility_analysis?.price_data) {
          setHistoricalData(volatilityResponse.data.volatility_analysis.price_data.slice(-30));
        }
      } catch (error) {
        console.warn('Failed to load historical data:', error);
      }

      toast.success(`Forecast generated successfully for ${selectedSymbol}`);
      loadRecentForecasts(); // Refresh recent forecasts

    } catch (error) {
      const errorMessage = handleApiError(error, 'Failed to generate forecast');
      toast.error(errorMessage);
      console.error('Forecast generation error:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRefreshData = async (symbol: string) => {
    setIsLoadingData(true);
    try {
      await api.forecasts.refreshData(symbol);
      toast.success(`Data refresh started for ${symbol}`);
    } catch (error) {
      toast.error(handleApiError(error, 'Failed to refresh data'));
    } finally {
      setIsLoadingData(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 flex items-center">
            <ChartBarIcon className="h-7 w-7 mr-2 text-blue-600" />
            AI Forecasting
          </h2>
          <p className="text-gray-600 mt-1">
            Generate 30-day price predictions using advanced machine learning models
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Configuration Panel */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="bg-white rounded-lg shadow-soft border border-gray-200 p-6"
        >
          <h3 className="text-lg font-medium text-gray-900 mb-4">Forecast Configuration</h3>

          {/* Symbol Selection */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Asset Symbol
            </label>
            <div className="space-y-2">
              <select
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                {supportedAssets.map((asset) => (
                  <option key={asset.id} value={asset.id}>
                    {asset.icon} {asset.name} ({asset.type})
                  </option>
                ))}
              </select>
              <button
                onClick={() => handleRefreshData(selectedSymbol)}
                disabled={isLoadingData}
                className="text-xs text-blue-600 hover:text-blue-800 disabled:opacity-50"
              >
                {isLoadingData ? 'Refreshing...' : 'Refresh Data'}
              </button>
            </div>
          </div>

          {/* Forecast Horizon */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Forecast Horizon: {forecastHorizon} days
            </label>
            <input
              type="range"
              min="7"
              max="90"
              value={forecastHorizon}
              onChange={(e) => setForecastHorizon(parseInt(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>7 days</span>
              <span>90 days</span>
            </div>
          </div>

          {/* Model Selection */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Models to Use
            </label>
            <div className="space-y-2">
              {models.map((model) => (
                <label key={model.id} className="flex items-start">
                  <input
                    type="checkbox"
                    checked={selectedModels.includes(model.id)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedModels([...selectedModels, model.id]);
                      } else {
                        setSelectedModels(selectedModels.filter(m => m !== model.id));
                      }
                    }}
                    className="mt-1 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                  <div className="ml-3">
                    <div className="text-sm font-medium text-gray-900">{model.name}</div>
                    <div className="text-xs text-gray-500">{model.description}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Include Explanation */}
          <div className="mb-6">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={includeExplanation}
                onChange={(e) => setIncludeExplanation(e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <span className="ml-2 text-sm text-gray-700">Include AI explanation</span>
            </label>
          </div>

          {/* Generate Button */}
          <button
            onClick={handleGenerateForecast}
            disabled={isGenerating || selectedModels.length === 0}
            className="w-full flex items-center justify-center px-4 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isGenerating ? (
              <>
                <LoadingSpinner size="small" color="white" className="mr-2" />
                Generating Forecast...
              </>
            ) : (
              <>
                <CpuChipIcon className="h-5 w-5 mr-2" />
                Generate Forecast
              </>
            )}
          </button>
        </motion.div>

        {/* Results Panel */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="lg:col-span-2 bg-white rounded-lg shadow-soft border border-gray-200 p-6"
        >
          <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
            <TrendingUpIcon className="h-5 w-5 mr-2 text-green-600" />
            Forecast Results
          </h3>

          {isGenerating ? (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <LoadingSpinner size="large" />
                <p className="mt-4 text-gray-600">Generating AI forecast...</p>
                <p className="text-sm text-gray-500">Analyzing {forecastHorizon} days using {selectedModels.join(', ')} models</p>
              </div>
            </div>
          ) : forecastData ? (
            <div className="space-y-6">
              {/* Model Selection for Visualization */}
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-medium text-gray-900">Forecast Results</h4>
                  <p className="text-sm text-gray-500">
                    Generated {formatDate(forecastData.generated_at, 'relative')} • {forecastData.data_points_used} data points
                  </p>
                </div>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="text-sm border border-gray-300 rounded px-2 py-1"
                >
                  {Object.keys(forecastData.forecasts).map(model => (
                    <option key={model} value={model}>
                      {model.toUpperCase()}
                    </option>
                  ))}
                </select>
              </div>

              {/* Chart */}
              <ForecastChart
                data={forecastData}
                historicalData={historicalData}
                selectedModel={selectedModel}
              />

              {/* Forecast Summary */}
              {forecastData.forecasts[selectedModel] && (
                <div className="bg-gray-50 rounded-lg p-4">
                  <h5 className="font-medium text-gray-900 mb-2">Forecast Summary</h5>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-gray-500">Current Price:</span>
                      <div className="font-medium">{formatCurrency(forecastData.last_price)}</div>
                    </div>
                    <div>
                      <span className="text-gray-500">Final Predicted:</span>
                      <div className="font-medium">
                        {formatCurrency(forecastData.forecasts[selectedModel].forecast[forecastData.forecast_horizon - 1])}
                      </div>
                    </div>
                    <div>
                      <span className="text-gray-500">Expected Change:</span>
                      <div className="font-medium">
                        {formatPercentage(
                          ((forecastData.forecasts[selectedModel].forecast[forecastData.forecast_horizon - 1] - forecastData.last_price) / forecastData.last_price) * 100
                        )}
                      </div>
                    </div>
                    {forecastData.forecasts[selectedModel].metrics?.mae && (
                      <div>
                        <span className="text-gray-500">Model MAE:</span>
                        <div className="font-medium">{forecastData.forecasts[selectedModel].metrics.mae.toFixed(2)}</div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* AI Explanation */}
              {forecastData.explanation && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <div className="flex items-start">
                    <InformationCircleIcon className="h-5 w-5 text-blue-600 mt-0.5 mr-2 flex-shrink-0" />
                    <div>
                      <h5 className="font-medium text-blue-900 mb-2">AI Analysis</h5>
                      <p className="text-blue-800 text-sm leading-relaxed">{forecastData.explanation.narrative}</p>
                      {forecastData.explanation.key_insights && forecastData.explanation.key_insights.length > 0 && (
                        <div className="mt-3">
                          <h6 className="font-medium text-blue-900 text-sm mb-1">Key Insights:</h6>
                          <ul className="text-blue-800 text-sm space-y-1">
                            {forecastData.explanation.key_insights.map((insight, idx) => (
                              <li key={idx} className="flex items-start">
                                <span className="mr-2">•</span>
                                <span>{insight}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-16">
              <ChartBarIcon className="mx-auto h-16 w-16 text-gray-300" />
              <h3 className="mt-4 text-lg font-medium text-gray-900">
                No Forecast Generated
              </h3>
              <p className="mt-2 text-gray-500">
                Configure your parameters and click "Generate Forecast" to create AI-powered predictions.
              </p>
            </div>
          )}
        </motion.div>
      </div>

      {/* Recent Forecasts */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-white rounded-lg shadow-soft border border-gray-200 p-6"
      >
        <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Forecasts</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {recentForecasts.length > 0 ? recentForecasts.map((forecast, idx) => {
            const forecastValues = forecast.forecast_data?.values || [];
            const finalValue = forecastValues[forecastValues.length - 1];
            const initialValue = forecastValues[0];
            const percentageChange = initialValue ? ((finalValue - initialValue) / initialValue) * 100 : 0;

            return (
              <div key={idx} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
                   onClick={() => setSelectedSymbol(forecast.symbol)}>
                <div className="flex justify-between items-start mb-2">
                  <div className="text-sm font-medium text-gray-900">
                    {forecast.symbol.toUpperCase()}
                  </div>
                  {forecast.accuracy_score && (
                    <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full">
                      {Math.round(forecast.accuracy_score * 100)}% Accuracy
                    </span>
                  )}
                </div>
                <div className="text-xs text-gray-500 mb-2">
                  {forecast.forecast_horizon}-day forecast • {forecast.model_name}
                </div>
                <div className="text-sm text-gray-600">
                  Predicted: <span className={`font-medium ${percentageChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {percentageChange >= 0 ? '+' : ''}{percentageChange.toFixed(1)}%
                  </span> change
                </div>
                <div className="text-xs text-gray-400 mt-1">
                  Generated {formatDate(forecast.created_at, 'relative')}
                </div>
              </div>
            );
          }) : (
            <div className="col-span-full text-center py-8 text-gray-500">
              No recent forecasts available. Generate your first forecast above.
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}