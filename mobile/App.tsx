/**
 * AI Options Trading System - Mobile Application
 * Main App Component
 */

import React, { useEffect } from 'react';
import { StatusBar, Platform } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { Provider } from 'react-redux';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { GestureHandlerRootView } from 'react-native-gesture-handler';

import { store } from './src/store/store';
import { AppNavigator } from './src/navigation/AppNavigator';
import { NotificationService } from './src/services/notifications/NotificationService';
import { BiometricAuthService } from './src/services/auth/BiometricAuthService';
import { OfflineManager } from './src/services/offline/OfflineManager';
import { PerformanceMonitor } from './src/services/performance/PerformanceMonitor';
import { ErrorBoundary } from './src/components/common/ErrorBoundary';
import { LoadingProvider } from './src/components/common/LoadingProvider';
import { ThemeProvider } from './src/components/common/ThemeProvider';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
  },
});

const App: React.FC = () => {
  useEffect(() => {
    // Initialize app services
    const initializeApp = async () => {
      try {
        // Initialize performance monitoring
        await PerformanceMonitor.initialize();
        
        // Initialize notification service
        await NotificationService.initialize();
        
        // Initialize biometric authentication
        await BiometricAuthService.initialize();
        
        // Initialize offline manager
        await OfflineManager.initialize();
        
        console.log('App services initialized successfully');
      } catch (error) {
        console.error('Failed to initialize app services:', error);
      }
    };

    initializeApp();
  }, []);

  return (
    <ErrorBoundary>
      <Provider store={store}>
        <QueryClientProvider client={queryClient}>
          <SafeAreaProvider>
            <GestureHandlerRootView style={{ flex: 1 }}>
              <ThemeProvider>
                <LoadingProvider>
                  <NavigationContainer>
                    <StatusBar
                      barStyle={Platform.OS === 'ios' ? 'dark-content' : 'light-content'}
                      backgroundColor={Platform.OS === 'android' ? '#1a1a1a' : undefined}
                    />
                    <AppNavigator />
                  </NavigationContainer>
                </LoadingProvider>
              </ThemeProvider>
            </GestureHandlerRootView>
          </SafeAreaProvider>
        </QueryClientProvider>
      </Provider>
    </ErrorBoundary>
  );
};

export default App;

