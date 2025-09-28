import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  SparklesIcon,
  LightBulbIcon,
  ExclamationTriangleIcon,
  TrendingUpIcon,
  TrendingDownIcon,
  InformationCircleIcon,
  BoltIcon,
  ShieldCheckIcon,
  CurrencyDollarIcon,
  ChartBarIcon,
  ClockIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart, BarChart, Bar } from 'recharts';
import toast from 'react-hot-toast';
import LoadingSpinner from '../ui/LoadingSpinner';
import { api, handleApiError } from '../../lib/api';
import { formatCurrency, formatPercentage, formatDate } from '../../lib/utils';

interface AIInsight {
  id: string;
  type: 'opportunity' | 'risk' | 'trend' | 'recommendation' | 'alert';
  title: string;
  description: string;
  confidence: number;
  impact: 'high' | 'medium' | 'low';
  timeframe: string;
  data_points: any[];
  visual_data?: {
    chart_type: 'line' | 'area' | 'bar';
    data: any[];
    x_axis: string;
    y_axis: string;
  };
  action_items?: string[];
  related_assets?: string[];
  timestamp: string;
}

interface MarketAnalysis {
  overall_sentiment: 'bullish' | 'bearish' | 'neutral';
  market_phase: string;
  volatility_level: 'low' | 'medium' | 'high';
  key_trends: string[];
  risk_factors: string[];
  opportunities: string[];
  ai_summary: string;
  confidence_score: number;
}

interface PredictiveAlert {
  type: 'price_movement' | 'volatility_spike' | 'volume_anomaly' | 'correlation_break';
  asset: string;
  probability: number;
  timeframe: string;
  description: string;
  suggested_action: string;
  risk_level: 'low' | 'medium' | 'high';
}

export default function AIInsights() {
  const [insights, setInsights] = useState<AIInsight[]>([]);
  const [marketAnalysis, setMarketAnalysis] = useState<MarketAnalysis | null>(null);
  const [predictiveAlerts, setPredictiveAlerts] = useState<PredictiveAlert[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedInsightType, setSelectedInsightType] = useState<string>('all');
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    loadAIInsights();

    // Auto-refresh every 5 minutes
    let interval: NodeJS.Timeout;
    if (autoRefresh) {
      interval = setInterval(() => {
        refreshInsights();
      }, 300000); // 5 minutes
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  const loadAIInsights = async () => {
    setIsLoading(true);
    try {
      await Promise.all([
        loadInsights(),
        loadMarketAnalysis(),
        loadPredictiveAlerts()
      ]);
    } catch (error) {
      console.error('Failed to load AI insights:', error);
      toast.error('Failed to load AI insights');
    } finally {
      setIsLoading(false);
    }
  };

  const loadInsights = async () => {
    try {
      // Get portfolio data and generate insights based on real data
      const portfolioResponse = await api.portfolio.getCurrent();
      const balances = portfolioResponse.data.balances || [];

      if (balances.length === 0) {
        setInsights([]);
        return;
      }

      const totalValue = balances.reduce((sum, b) => sum + (b.usd_value || 0), 0);
      const portfolioWeights = {};
      balances.forEach(balance => {
        portfolioWeights[balance.symbol] = (balance.usd_value || 0) / totalValue;
      });

      // Get real insights from backend
      const insightsData: AIInsight[] = [];

      // Risk Analysis Insight
      try {
        const riskResponse = await api.portfolio.getRiskMetrics({
          portfolio_weights: portfolioWeights
        });

        const riskData = riskResponse.data;
        if (riskData.sharpe_ratio < 1.0) {
          insightsData.push({
            id: 'risk-1',
            type: 'risk',
            title: 'Suboptimal Risk-Adjusted Returns',
            description: `Portfolio Sharpe ratio of ${riskData.sharpe_ratio?.toFixed(2)} indicates suboptimal risk-adjusted returns. Consider rebalancing to improve efficiency.`,
            confidence: 0.85,
            impact: 'medium',
            timeframe: '1-2 weeks',
            data_points: [],
            visual_data: {
              chart_type: 'bar',
              data: [
                { metric: 'Current Sharpe', value: riskData.sharpe_ratio || 0 },
                { metric: 'Target Sharpe', value: 1.5 },
                { metric: 'Volatility', value: (riskData.volatility_30d || 0) * 100 }
              ],
              x_axis: 'metric',
              y_axis: 'value'
            },
            action_items: [
              'Review high-volatility positions',
              'Consider diversification into stable assets',
              'Evaluate correlation between holdings'
            ],
            related_assets: Object.keys(portfolioWeights),
            timestamp: new Date().toISOString()
          });
        }

        if (riskData.volatility_30d > 0.5) {
          insightsData.push({
            id: 'risk-2',
            type: 'alert',
            title: 'High Portfolio Volatility Detected',
            description: `30-day portfolio volatility of ${formatPercentage((riskData.volatility_30d || 0) * 100)} exceeds recommended thresholds for most risk profiles.`,
            confidence: 0.92,
            impact: 'high',
            timeframe: 'Immediate',
            data_points: [],
            action_items: [
              'Reduce position sizes in volatile assets',
              'Add hedging instruments',
              'Consider stablecoin allocation'
            ],
            related_assets: Object.keys(portfolioWeights),
            timestamp: new Date().toISOString()
          });
        }
      } catch (error) {
        console.warn('Failed to load risk insights:', error);
      }

      // Portfolio Allocation Insight
      const largestPosition = balances.reduce((max, balance) =>
        (balance.usd_value || 0) > (max.usd_value || 0) ? balance : max
      );

      if (largestPosition && (largestPosition.usd_value || 0) / totalValue > 0.6) {
        insightsData.push({
          id: 'recommendation-1',
          type: 'recommendation',
          title: 'Portfolio Concentration Risk',
          description: `${largestPosition.symbol.toUpperCase()} represents ${formatPercentage((largestPosition.usd_value || 0) / totalValue * 100)} of your portfolio. Consider diversification to reduce concentration risk.`,
          confidence: 0.88,
          impact: 'medium',
          timeframe: 'Next rebalancing',
          data_points: [],
          visual_data: {
            chart_type: 'area',
            data: balances.map(balance => ({
              asset: balance.symbol.toUpperCase(),
              allocation: (balance.usd_value || 0) / totalValue * 100,
              recommended: Math.min((balance.usd_value || 0) / totalValue * 100, 40)
            })),
            x_axis: 'asset',
            y_axis: 'allocation'
          },
          action_items: [
            `Reduce ${largestPosition.symbol.toUpperCase()} allocation to below 40%`,
            'Diversify into uncorrelated assets',
            'Consider dollar-cost averaging for rebalancing'
          ],
          related_assets: [largestPosition.symbol.toUpperCase()],
          timestamp: new Date().toISOString()
        });
      }

      // Forecasting Opportunities
      try {
        const supportedResponse = await api.forecasts.getSupportedAssets();
        const availableAssets = supportedResponse.data.crypto_assets;

        for (const symbol of Object.keys(portfolioWeights).slice(0, 2)) {
          if (availableAssets.includes(symbol.toLowerCase())) {
            try {
              const forecastResponse = await api.forecasts.generate({
                symbol: symbol.toLowerCase(),
                horizon_days: 30,
                models: ['ensemble'],
                include_explanation: true
              });

              const forecast = forecastResponse.data;
              const finalPrice = forecast.forecasts.ensemble?.forecast?.[forecast.forecast_horizon - 1];
              const expectedReturn = finalPrice ? (finalPrice - forecast.last_price) / forecast.last_price : 0;

              if (Math.abs(expectedReturn) > 0.1) {
                insightsData.push({
                  id: `forecast-${symbol}`,
                  type: expectedReturn > 0 ? 'opportunity' : 'risk',
                  title: `${symbol.toUpperCase()} ${expectedReturn > 0 ? 'Upward' : 'Downward'} Trend Detected`,
                  description: `AI forecasting models predict ${formatPercentage(Math.abs(expectedReturn) * 100)} ${expectedReturn > 0 ? 'gain' : 'loss'} for ${symbol.toUpperCase()} over the next 30 days.`,
                  confidence: 0.75,
                  impact: Math.abs(expectedReturn) > 0.2 ? 'high' : 'medium',
                  timeframe: '30 days',
                  data_points: [],
                  visual_data: {
                    chart_type: 'line',
                    data: forecast.forecasts.ensemble?.forecast?.map((price, index) => ({
                      day: index + 1,
                      predicted_price: price,
                      current_price: forecast.last_price
                    })) || [],
                    x_axis: 'day',
                    y_axis: 'predicted_price'
                  },
                  action_items: [
                    expectedReturn > 0 ? `Consider increasing ${symbol.toUpperCase()} position` : `Consider reducing ${symbol.toUpperCase()} exposure`,
                    'Monitor forecast accuracy over time',
                    'Set price alerts for key levels'
                  ],
                  related_assets: [symbol.toUpperCase()],
                  timestamp: new Date().toISOString()
                });
              }
            } catch (error) {
              console.warn(`Failed to generate forecast for ${symbol}:`, error);
            }
          }
        }
      } catch (error) {
        console.warn('Failed to load forecasting insights:', error);
      }

      setInsights(insightsData);
    } catch (error) {
      console.error('Failed to load insights:', error);
      setInsights([]);
    }
  };

  const loadMarketAnalysis = async () => {
    try {
      // Get portfolio data for analysis
      const portfolioResponse = await api.portfolio.getCurrent();
      const balances = portfolioResponse.data.balances || [];

      if (balances.length === 0) {
        setMarketAnalysis(null);
        return;
      }

      const totalValue = balances.reduce((sum, b) => sum + (b.usd_value || 0), 0);
      const portfolioWeights = {};
      balances.forEach(balance => {
        portfolioWeights[balance.symbol] = (balance.usd_value || 0) / totalValue;
      });

      // Get risk metrics to determine market phase
      const riskResponse = await api.portfolio.getRiskMetrics({
        portfolio_weights: portfolioWeights
      });

      const riskData = riskResponse.data;
      const sharpeRatio = riskData.sharpe_ratio || 0;
      const volatility = riskData.volatility_30d || 0.3;

      // Determine sentiment based on Sharpe ratio and volatility
      let sentiment: 'bullish' | 'bearish' | 'neutral' = 'neutral';
      if (sharpeRatio > 1.2 && volatility < 0.4) sentiment = 'bullish';
      else if (sharpeRatio < 0.8 || volatility > 0.6) sentiment = 'bearish';

      const analysis: MarketAnalysis = {
        overall_sentiment: sentiment,
        market_phase: sentiment === 'bullish' ? 'Risk-On Environment' : sentiment === 'bearish' ? 'Risk-Off Environment' : 'Consolidation Phase',
        volatility_level: volatility < 0.3 ? 'low' : volatility > 0.5 ? 'high' : 'medium',
        key_trends: [
          `Portfolio Sharpe ratio at ${sharpeRatio.toFixed(2)}`,
          `30-day volatility at ${formatPercentage(volatility * 100)}`,
          `${balances.length} assets in portfolio`,
          'Real-time risk monitoring active'
        ],
        risk_factors: [
          volatility > 0.5 ? 'High portfolio volatility' : 'Moderate volatility levels',
          'Concentration risk in top holdings',
          'Market correlation during stress events',
          'Liquidity constraints in volatile markets'
        ],
        opportunities: [
          'Rebalancing based on risk metrics',
          'Volatility-adjusted position sizing',
          'Correlation-based diversification',
          'Risk-adjusted return optimization'
        ],
        ai_summary: `Portfolio analysis indicates a ${sentiment} outlook with ${volatility > 0.5 ? 'elevated' : 'manageable'} risk levels. Current Sharpe ratio of ${sharpeRatio.toFixed(2)} suggests ${sharpeRatio > 1.0 ? 'adequate' : 'suboptimal'} risk-adjusted performance. Recommend ${volatility > 0.5 ? 'defensive positioning' : 'maintaining current allocation'} with ongoing risk monitoring.`,
        confidence_score: 0.82
      };

      setMarketAnalysis(analysis);
    } catch (error) {
      console.error('Failed to load market analysis:', error);
      setMarketAnalysis(null);
    }
  };

  const loadPredictiveAlerts = async () => {
    try {
      // Get portfolio data for predictive analysis
      const portfolioResponse = await api.portfolio.getCurrent();
      const balances = portfolioResponse.data.balances || [];

      if (balances.length === 0) {
        setPredictiveAlerts([]);
        return;
      }

      const alerts: PredictiveAlert[] = [];

      // Generate volatility alerts based on real risk metrics
      const totalValue = balances.reduce((sum, b) => sum + (b.usd_value || 0), 0);
      const portfolioWeights = {};
      balances.forEach(balance => {
        portfolioWeights[balance.symbol] = (balance.usd_value || 0) / totalValue;
      });

      try {
        const riskResponse = await api.portfolio.getRiskMetrics({
          portfolio_weights: portfolioWeights
        });

        const volatility = riskResponse.data.volatility_30d || 0;
        if (volatility > 0.4) {
          alerts.push({
            type: 'volatility_spike',
            asset: 'Portfolio',
            probability: Math.min(volatility / 0.6, 0.95),
            timeframe: 'Next 48 hours',
            description: `High volatility environment detected with potential for further increases`,
            suggested_action: 'Consider reducing position sizes or adding hedging',
            risk_level: volatility > 0.6 ? 'high' : 'medium'
          });
        }
      } catch (error) {
        console.warn('Failed to load risk-based alerts:', error);
      }

      // Generate correlation-based alerts
      try {
        const symbols = balances.slice(0, 4).map(b => b.symbol.toLowerCase());
        if (symbols.length >= 2) {
          const correlationResponse = await api.forecasts.getCorrelation(symbols, 30);
          const correlationMatrix = correlationResponse.data.correlation_matrix;

          // Check for high correlations
          const highCorrelations = [];
          Object.keys(correlationMatrix).forEach(asset1 => {
            Object.keys(correlationMatrix[asset1]).forEach(asset2 => {
              if (asset1 !== asset2 && correlationMatrix[asset1][asset2] > 0.85) {
                highCorrelations.push({
                  asset1: asset1.toUpperCase(),
                  asset2: asset2.toUpperCase(),
                  correlation: correlationMatrix[asset1][asset2]
                });
              }
            });
          });

          if (highCorrelations.length > 0) {
            alerts.push({
              type: 'correlation_break',
              asset: highCorrelations[0].asset1,
              probability: 0.75,
              timeframe: 'Next 7 days',
              description: `High correlation (${highCorrelations[0].correlation.toFixed(2)}) with ${highCorrelations[0].asset2} may break during market stress`,
              suggested_action: 'Monitor for diversification opportunities',
              risk_level: 'medium'
            });
          }
        }
      } catch (error) {
        console.warn('Failed to load correlation alerts:', error);
      }

      setPredictiveAlerts(alerts);
    } catch (error) {
      console.error('Failed to load predictive alerts:', error);
      setPredictiveAlerts([]);
    }
  };

  const refreshInsights = async () => {
    if (isGenerating) return;

    setIsGenerating(true);
    try {
      await loadInsights();
      await loadPredictiveAlerts();
      toast.success('AI insights updated');
    } catch (error) {
      console.error('Failed to refresh insights:', error);
      toast.error('Failed to refresh insights');
    } finally {
      setIsGenerating(false);
    }
  };

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'opportunity': return TrendingUpIcon;
      case 'risk': return ExclamationTriangleIcon;
      case 'trend': return ChartBarIcon;
      case 'recommendation': return LightBulbIcon;
      case 'alert': return BoltIcon;
      default: return InformationCircleIcon;
    }
  };

  const getInsightColor = (type: string) => {
    switch (type) {
      case 'opportunity': return 'text-green-600 bg-green-50 border-green-200';
      case 'risk': return 'text-red-600 bg-red-50 border-red-200';
      case 'trend': return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'recommendation': return 'text-purple-600 bg-purple-50 border-purple-200';
      case 'alert': return 'text-orange-600 bg-orange-50 border-orange-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600 bg-green-100';
    if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const filteredInsights = selectedInsightType === 'all'
    ? insights
    : insights.filter(insight => insight.type === selectedInsightType);

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
            <SparklesIcon className="h-7 w-7 mr-2 text-purple-600" />
            AI Insights
          </h2>
          <p className="text-gray-600 mt-1">
            Intelligent analysis based on your portfolio data and risk metrics
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <label className="flex items-center text-sm">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="mr-2 h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
              />
              Auto-refresh
            </label>
            {autoRefresh && (
              <div className="flex items-center text-purple-600">
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse mr-1"></div>
                <span className="text-xs">Live</span>
              </div>
            )}
          </div>
          <button
            onClick={refreshInsights}
            disabled={isGenerating}
            className="flex items-center px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50"
          >
            <ArrowPathIcon className={`h-4 w-4 mr-2 ${isGenerating ? 'animate-spin' : ''}`} />
            {isGenerating ? 'Analyzing...' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* Market Analysis Overview */}
      {marketAnalysis && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900 flex items-center">
              <SparklesIcon className="h-5 w-5 mr-2 text-purple-600" />
              Portfolio AI Analysis
            </h3>
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor(marketAnalysis.confidence_score)}`}>
              {formatPercentage(marketAnalysis.confidence_score * 100)} Confidence
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <div className="flex items-center mb-2">
                <TrendingUpIcon className="h-4 w-4 mr-2 text-green-600" />
                <span className="text-sm font-medium text-gray-700">Portfolio Sentiment</span>
              </div>
              <div className={`text-lg font-semibold ${
                marketAnalysis.overall_sentiment === 'bullish' ? 'text-green-600' :
                marketAnalysis.overall_sentiment === 'bearish' ? 'text-red-600' : 'text-gray-600'
              }`}>
                {marketAnalysis.overall_sentiment.charAt(0).toUpperCase() + marketAnalysis.overall_sentiment.slice(1)}
              </div>
              <div className="text-sm text-gray-600">{marketAnalysis.market_phase}</div>
            </div>

            <div>
              <div className="flex items-center mb-2">
                <BoltIcon className="h-4 w-4 mr-2 text-orange-600" />
                <span className="text-sm font-medium text-gray-700">Volatility Level</span>
              </div>
              <div className={`text-lg font-semibold ${
                marketAnalysis.volatility_level === 'low' ? 'text-green-600' :
                marketAnalysis.volatility_level === 'high' ? 'text-red-600' : 'text-yellow-600'
              }`}>
                {marketAnalysis.volatility_level.charAt(0).toUpperCase() + marketAnalysis.volatility_level.slice(1)}
              </div>
            </div>

            <div>
              <div className="flex items-center mb-2">
                <InformationCircleIcon className="h-4 w-4 mr-2 text-blue-600" />
                <span className="text-sm font-medium text-gray-700">Insights Available</span>
              </div>
              <div className="text-lg font-semibold text-blue-600">
                {insights.length}
              </div>
              <div className="text-sm text-gray-600">
                Based on real portfolio data
              </div>
            </div>
          </div>

          <div className="mt-4 p-4 bg-white bg-opacity-60 rounded">
            <p className="text-sm text-gray-700 leading-relaxed">{marketAnalysis.ai_summary}</p>
          </div>
        </motion.div>
      )}

      {/* Predictive Alerts */}
      {predictiveAlerts.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white rounded-lg shadow-soft border border-gray-200 p-6"
        >
          <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
            <BoltIcon className="h-5 w-5 mr-2 text-orange-600" />
            Predictive Alerts (Real-time Analysis)
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {predictiveAlerts.map((alert, index) => (
              <div key={index} className={`p-4 rounded-lg border ${
                alert.risk_level === 'high' ? 'border-red-200 bg-red-50' :
                alert.risk_level === 'medium' ? 'border-yellow-200 bg-yellow-50' :
                'border-green-200 bg-green-50'
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-900">{alert.asset}</span>
                  <span className={`text-xs px-2 py-1 rounded-full ${getConfidenceColor(alert.probability)}`}>
                    {formatPercentage(alert.probability * 100)}
                  </span>
                </div>
                <div className="text-sm text-gray-600 mb-2">{alert.description}</div>
                <div className="text-xs text-gray-500 mb-2">{alert.timeframe}</div>
                <div className="text-xs text-gray-700 font-medium">{alert.suggested_action}</div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Filter Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'all', name: 'All Insights' },
            { id: 'opportunity', name: 'Opportunities' },
            { id: 'risk', name: 'Risks' },
            { id: 'trend', name: 'Trends' },
            { id: 'recommendation', name: 'Recommendations' }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSelectedInsightType(tab.id)}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                selectedInsightType === tab.id
                  ? 'border-purple-500 text-purple-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.name}
            </button>
          ))}
        </nav>
      </div>

      {/* Insights Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {filteredInsights.map((insight, index) => {
          const IconComponent = getInsightIcon(insight.type);

          return (
            <motion.div
              key={insight.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`bg-white rounded-lg shadow-soft border p-6 ${getInsightColor(insight.type)}`}
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center">
                  <IconComponent className="h-5 w-5 mr-2" />
                  <div>
                    <h4 className="font-medium text-gray-900">{insight.title}</h4>
                    <div className="flex items-center space-x-2 mt-1">
                      <span className={`text-xs px-2 py-1 rounded-full ${getConfidenceColor(insight.confidence)}`}>
                        {formatPercentage(insight.confidence * 100)} Confidence
                      </span>
                      <span className="text-xs text-gray-500">{insight.timeframe}</span>
                    </div>
                  </div>
                </div>
                <ClockIcon className="h-4 w-4 text-gray-400" />
              </div>

              <p className="text-sm text-gray-700 mb-4">{insight.description}</p>

              {/* Visual Data */}
              {insight.visual_data && insight.visual_data.data.length > 0 && (
                <div className="mb-4">
                  <div className="h-32">
                    <ResponsiveContainer width="100%" height="100%">
                      {insight.visual_data.chart_type === 'line' && (
                        <LineChart data={insight.visual_data.data}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey={insight.visual_data.x_axis} fontSize={10} />
                          <YAxis fontSize={10} />
                          <Tooltip />
                          <Line
                            type="monotone"
                            dataKey={insight.visual_data.y_axis}
                            stroke="#8884d8"
                            strokeWidth={2}
                            dot={false}
                          />
                        </LineChart>
                      )}
                      {insight.visual_data.chart_type === 'area' && (
                        <AreaChart data={insight.visual_data.data}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey={insight.visual_data.x_axis} fontSize={10} />
                          <YAxis fontSize={10} />
                          <Tooltip />
                          <Area
                            type="monotone"
                            dataKey={insight.visual_data.y_axis}
                            stroke="#8884d8"
                            fill="#8884d8"
                            fillOpacity={0.6}
                          />
                        </AreaChart>
                      )}
                      {insight.visual_data.chart_type === 'bar' && (
                        <BarChart data={insight.visual_data.data}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey={insight.visual_data.x_axis} fontSize={10} />
                          <YAxis fontSize={10} />
                          <Tooltip />
                          <Bar dataKey={insight.visual_data.y_axis} fill="#8884d8" />
                        </BarChart>
                      )}
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Action Items */}
              {insight.action_items && insight.action_items.length > 0 && (
                <div className="border-t border-gray-200 pt-4">
                  <h6 className="text-sm font-medium text-gray-900 mb-2">Recommended Actions:</h6>
                  <ul className="text-sm text-gray-600 space-y-1">
                    {insight.action_items.map((action, actionIndex) => (
                      <li key={actionIndex} className="flex items-start">
                        <span className="mr-2">•</span>
                        <span>{action}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Related Assets */}
              {insight.related_assets && insight.related_assets.length > 0 && (
                <div className="mt-3 flex items-center">
                  <span className="text-xs text-gray-500 mr-2">Related:</span>
                  <div className="flex space-x-1">
                    {insight.related_assets.map((asset, assetIndex) => (
                      <span key={assetIndex} className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded">
                        {asset}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div className="mt-3 text-xs text-gray-400">
                Generated {formatDate(insight.timestamp, 'relative')}
              </div>
            </motion.div>
          );
        })}
      </div>

      {filteredInsights.length === 0 && (
        <div className="text-center py-12">
          <SparklesIcon className="mx-auto h-12 w-12 text-gray-300 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            {insights.length === 0 ? 'No portfolio data for insights' : 'No insights for this category'}
          </h3>
          <p className="text-gray-500">
            {insights.length === 0
              ? 'Add assets to your portfolio to generate AI insights.'
              : 'AI is analyzing your portfolio data to generate relevant insights.'}
          </p>
        </div>
      )}
    </div>
  );
}