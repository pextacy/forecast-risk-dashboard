import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import { motion } from 'framer-motion';
import { Toaster } from 'react-hot-toast';
import DashboardLayout from '../components/layout/DashboardLayout';
import OverviewCards from '../components/dashboard/OverviewCards';
import ForecastPanel from '../components/forecast/ForecastPanel';
import RiskPanel from '../components/risk/RiskPanel';
import PortfolioPanel from '../components/portfolio/PortfolioPanel';
import HedgePanel from '../components/hedge/HedgePanel';
import MarketOverview from '../components/dashboard/MarketOverview';
import AIInsights from '../components/dashboard/AIInsights';
import { useDashboardData } from '../hooks/useDashboardData';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import ErrorBoundary from '../components/ui/ErrorBoundary';

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('overview');
  const { data: dashboardData, isLoading, error } = useDashboardData();

  const tabs = [
    { id: 'overview', name: 'Overview', icon: '📊' },
    { id: 'market', name: 'Market Overview', icon: '🌍' },
    { id: 'forecasts', name: 'Forecasts', icon: '🔮' },
    { id: 'risk', name: 'Risk Analysis', icon: '⚠️' },
    { id: 'portfolio', name: 'Portfolio', icon: '💼' },
    { id: 'hedge', name: 'Hedge Suggestions', icon: '🛡️' },
    { id: 'ai-insights', name: 'AI Insights', icon: '✨' }
  ];

  if (isLoading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center h-96">
          <LoadingSpinner size="large" />
        </div>
      </DashboardLayout>
    );
  }

  if (error) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="text-6xl mb-4">❌</div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Dashboard Error</h2>
            <p className="text-gray-600 mb-4">{error.message}</p>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
            >
              Retry
            </button>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  return (
    <>
      <Head>
        <title>Treasury Risk Dashboard - AI-Powered Financial Analytics</title>
        <meta name="description" content="Comprehensive treasury management with AI-powered forecasting, risk analysis, and hedge suggestions." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <DashboardLayout>
        <div className="space-y-6">
          {/* Header */}
          <div className="bg-white shadow-soft rounded-lg p-6">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h1 className="text-3xl font-bold text-gray-900">Treasury Risk Dashboard</h1>
                <p className="mt-1 text-sm text-gray-500">
                  AI-powered forecasting and risk management for treasury operations
                </p>
              </div>
              <div className="mt-4 sm:mt-0">
                <div className="flex items-center space-x-2 text-sm text-gray-500">
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
                    Live Data
                  </div>
                  <span>•</span>
                  <span>Last updated: {new Date().toLocaleTimeString()}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="bg-white shadow-soft rounded-lg">
            <div className="border-b border-gray-200">
              <nav className="-mb-px flex space-x-8 px-6 overflow-x-auto" aria-label="Tabs">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`
                      flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm whitespace-nowrap
                      ${activeTab === tab.id
                        ? 'border-primary-500 text-primary-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                      }
                    `}
                  >
                    <span className="text-lg">{tab.icon}</span>
                    <span>{tab.name}</span>
                  </button>
                ))}
              </nav>
            </div>

            {/* Tab Content */}
            <div className="p-6">
              <ErrorBoundary>
                <motion.div
                  key={activeTab}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  {activeTab === 'overview' && (
                    <div className="space-y-6">
                      <OverviewCards data={dashboardData} />

                      {/* Quick Actions */}
                      <div className="bg-gray-50 rounded-lg p-6">
                        <h3 className="text-lg font-medium text-gray-900 mb-4">Quick Actions</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                          <button
                            onClick={() => setActiveTab('market')}
                            className="flex items-center p-4 bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow border border-gray-200 hover:border-primary-300"
                          >
                            <div className="text-2xl mr-3">🌍</div>
                            <div className="text-left">
                              <div className="font-medium text-gray-900">Market Overview</div>
                              <div className="text-sm text-gray-500">Real-time market analysis</div>
                            </div>
                          </button>

                          <button
                            onClick={() => setActiveTab('forecasts')}
                            className="flex items-center p-4 bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow border border-gray-200 hover:border-primary-300"
                          >
                            <div className="text-2xl mr-3">🔮</div>
                            <div className="text-left">
                              <div className="font-medium text-gray-900">Generate Forecast</div>
                              <div className="text-sm text-gray-500">Create 30-day price predictions</div>
                            </div>
                          </button>

                          <button
                            onClick={() => setActiveTab('risk')}
                            className="flex items-center p-4 bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow border border-gray-200 hover:border-primary-300"
                          >
                            <div className="text-2xl mr-3">⚠️</div>
                            <div className="text-left">
                              <div className="font-medium text-gray-900">Analyze Risk</div>
                              <div className="text-sm text-gray-500">Calculate portfolio metrics</div>
                            </div>
                          </button>

                          <button
                            onClick={() => setActiveTab('ai-insights')}
                            className="flex items-center p-4 bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow border border-gray-200 hover:border-primary-300"
                          >
                            <div className="text-2xl mr-3">✨</div>
                            <div className="text-left">
                              <div className="font-medium text-gray-900">AI Insights</div>
                              <div className="text-sm text-gray-500">Smart recommendations</div>
                            </div>
                          </button>
                        </div>
                      </div>
                    </div>
                  )}

                  {activeTab === 'market' && <MarketOverview />}
                  {activeTab === 'forecasts' && <ForecastPanel />}
                  {activeTab === 'risk' && <RiskPanel />}
                  {activeTab === 'portfolio' && <PortfolioPanel />}
                  {activeTab === 'hedge' && <HedgePanel />}
                  {activeTab === 'ai-insights' && <AIInsights />}
                </motion.div>
              </ErrorBoundary>
            </div>
          </div>
        </div>
      </DashboardLayout>

      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
          success: {
            duration: 3000,
            style: {
              background: '#10b981',
            },
          },
          error: {
            duration: 5000,
            style: {
              background: '#ef4444',
            },
          },
        }}
      />
    </>
  );
}