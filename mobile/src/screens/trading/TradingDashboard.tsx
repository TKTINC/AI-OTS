/**
 * Trading Dashboard Screen
 * Main trading interface with signal dashboard and execution
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  TouchableOpacity,
  Dimensions,
  Alert,
  Vibration,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { useQuery, useMutation } from '@tanstack/react-query';
import { useSelector, useDispatch } from 'react-redux';
import Animated, { 
  useSharedValue, 
  useAnimatedStyle, 
  withSpring,
  withTiming,
  runOnJS,
} from 'react-native-reanimated';
import { PanGestureHandler, State } from 'react-native-gesture-handler';

import { RootState } from '../../store/store';
import { SignalCard } from '../../components/trading/SignalCard';
import { QuickTradeModal } from '../../components/trading/QuickTradeModal';
import { PortfolioSummary } from '../../components/trading/PortfolioSummary';
import { MarketStatus } from '../../components/trading/MarketStatus';
import { TradingService } from '../../services/api/TradingService';
import { HapticFeedback } from '../../utils/HapticFeedback';
import { formatCurrency, formatPercentage } from '../../utils/formatters';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

interface Signal {
  id: string;
  symbol: string;
  strategy: string;
  signal_type: string;
  confidence: number;
  expected_return: number;
  risk_level: string;
  priority: string;
  entry_price: number;
  target_price: number;
  stop_loss: number;
  expiration: string;
  created_at: string;
}

export const TradingDashboard: React.FC = () => {
  const dispatch = useDispatch();
  const { user, portfolios } = useSelector((state: RootState) => state.auth);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedSignal, setSelectedSignal] = useState<Signal | null>(null);
  const [quickTradeVisible, setQuickTradeVisible] = useState(false);
  const [filter, setFilter] = useState<'all' | 'high' | 'critical'>('all');

  // Animated values for pull-to-refresh
  const pullDistance = useSharedValue(0);
  const refreshOpacity = useSharedValue(0);

  // Fetch active signals
  const {
    data: signals = [],
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['signals', 'active', filter],
    queryFn: () => TradingService.getActiveSignals({ priority: filter }),
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  // Execute signal mutation
  const executeSignalMutation = useMutation({
    mutationFn: TradingService.executeSignal,
    onSuccess: (data) => {
      HapticFeedback.success();
      Alert.alert(
        'Signal Executed',
        `Successfully executed ${data.signal.symbol} ${data.signal.signal_type}`,
        [{ text: 'OK' }]
      );
      refetch();
    },
    onError: (error: any) => {
      HapticFeedback.error();
      Alert.alert(
        'Execution Failed',
        error.message || 'Failed to execute signal',
        [{ text: 'OK' }]
      );
    },
  });

  // Handle pull-to-refresh
  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    HapticFeedback.light();
    try {
      await refetch();
    } finally {
      setRefreshing(false);
    }
  }, [refetch]);

  // Handle signal execution
  const handleExecuteSignal = useCallback((signal: Signal) => {
    HapticFeedback.medium();
    Alert.alert(
      'Execute Signal',
      `Execute ${signal.symbol} ${signal.signal_type}?\n\nConfidence: ${formatPercentage(signal.confidence)}\nExpected Return: ${formatPercentage(signal.expected_return)}`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Execute',
          style: 'default',
          onPress: () => {
            executeSignalMutation.mutate({
              signalId: signal.id,
              quantity: 100, // Default quantity
            });
          },
        },
      ]
    );
  }, [executeSignalMutation]);

  // Handle signal card press
  const handleSignalPress = useCallback((signal: Signal) => {
    HapticFeedback.light();
    setSelectedSignal(signal);
    setQuickTradeVisible(true);
  }, []);

  // Filter signals based on selected filter
  const filteredSignals = signals.filter((signal: Signal) => {
    if (filter === 'all') return true;
    return signal.priority.toLowerCase() === filter;
  });

  // Animated style for pull-to-refresh indicator
  const refreshIndicatorStyle = useAnimatedStyle(() => {
    return {
      opacity: refreshOpacity.value,
      transform: [{ translateY: pullDistance.value * 0.5 }],
    };
  });

  // Pan gesture handler for pull-to-refresh
  const onPanGestureEvent = (event: any) => {
    const { translationY, state } = event.nativeEvent;
    
    if (state === State.ACTIVE && translationY > 0) {
      pullDistance.value = Math.min(translationY, 100);
      refreshOpacity.value = Math.min(translationY / 100, 1);
    } else if (state === State.END) {
      if (translationY > 80) {
        runOnJS(onRefresh)();
      }
      pullDistance.value = withSpring(0);
      refreshOpacity.value = withTiming(0);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <Text style={styles.headerTitle}>Trading</Text>
          <Text style={styles.headerSubtitle}>
            {filteredSignals.length} active signals
          </Text>
        </View>
        <TouchableOpacity
          style={styles.quickTradeButton}
          onPress={() => setQuickTradeVisible(true)}
        >
          <Icon name="flash-on" size={24} color="#FFFFFF" />
        </TouchableOpacity>
      </View>

      {/* Market Status */}
      <MarketStatus />

      {/* Portfolio Summary */}
      <PortfolioSummary portfolios={portfolios} />

      {/* Filter Tabs */}
      <View style={styles.filterContainer}>
        {['all', 'high', 'critical'].map((filterOption) => (
          <TouchableOpacity
            key={filterOption}
            style={[
              styles.filterTab,
              filter === filterOption && styles.filterTabActive,
            ]}
            onPress={() => {
              HapticFeedback.light();
              setFilter(filterOption as any);
            }}
          >
            <Text
              style={[
                styles.filterTabText,
                filter === filterOption && styles.filterTabTextActive,
              ]}
            >
              {filterOption.charAt(0).toUpperCase() + filterOption.slice(1)}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Pull-to-refresh indicator */}
      <Animated.View style={[styles.refreshIndicator, refreshIndicatorStyle]}>
        <Icon name="refresh" size={24} color="#007AFF" />
      </Animated.View>

      {/* Signals List */}
      <PanGestureHandler onGestureEvent={onPanGestureEvent}>
        <Animated.View style={{ flex: 1 }}>
          <ScrollView
            style={styles.signalsList}
            showsVerticalScrollIndicator={false}
            refreshControl={
              <RefreshControl
                refreshing={refreshing}
                onRefresh={onRefresh}
                tintColor="#007AFF"
                colors={['#007AFF']}
              />
            }
          >
            {isLoading ? (
              <View style={styles.loadingContainer}>
                <Text style={styles.loadingText}>Loading signals...</Text>
              </View>
            ) : error ? (
              <View style={styles.errorContainer}>
                <Icon name="error" size={48} color="#FF3B30" />
                <Text style={styles.errorText}>Failed to load signals</Text>
                <TouchableOpacity style={styles.retryButton} onPress={() => refetch()}>
                  <Text style={styles.retryButtonText}>Retry</Text>
                </TouchableOpacity>
              </View>
            ) : filteredSignals.length === 0 ? (
              <View style={styles.emptyContainer}>
                <Icon name="trending-up" size={64} color="#C7C7CC" />
                <Text style={styles.emptyText}>No signals available</Text>
                <Text style={styles.emptySubtext}>
                  Pull down to refresh or check back later
                </Text>
              </View>
            ) : (
              filteredSignals.map((signal: Signal, index: number) => (
                <SignalCard
                  key={signal.id}
                  signal={signal}
                  onPress={() => handleSignalPress(signal)}
                  onExecute={() => handleExecuteSignal(signal)}
                  style={[
                    styles.signalCard,
                    { marginTop: index === 0 ? 10 : 0 },
                  ]}
                />
              ))
            )}
          </ScrollView>
        </Animated.View>
      </PanGestureHandler>

      {/* Quick Trade Modal */}
      <QuickTradeModal
        visible={quickTradeVisible}
        signal={selectedSignal}
        onClose={() => {
          setQuickTradeVisible(false);
          setSelectedSignal(null);
        }}
        onExecute={(signalId, quantity) => {
          executeSignalMutation.mutate({ signalId, quantity });
          setQuickTradeVisible(false);
          setSelectedSignal(null);
        }}
      />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F2F2F7',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 15,
    backgroundColor: '#FFFFFF',
    borderBottomWidth: 1,
    borderBottomColor: '#E5E5EA',
  },
  headerLeft: {
    flex: 1,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#000000',
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#8E8E93',
    marginTop: 2,
  },
  quickTradeButton: {
    backgroundColor: '#007AFF',
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
  },
  filterContainer: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    paddingVertical: 10,
    backgroundColor: '#FFFFFF',
    borderBottomWidth: 1,
    borderBottomColor: '#E5E5EA',
  },
  filterTab: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginRight: 10,
    borderRadius: 16,
    backgroundColor: '#F2F2F7',
  },
  filterTabActive: {
    backgroundColor: '#007AFF',
  },
  filterTabText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#8E8E93',
  },
  filterTabTextActive: {
    color: '#FFFFFF',
  },
  refreshIndicator: {
    position: 'absolute',
    top: 120,
    left: screenWidth / 2 - 12,
    zIndex: 1000,
  },
  signalsList: {
    flex: 1,
    paddingHorizontal: 20,
  },
  signalCard: {
    marginBottom: 12,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 60,
  },
  loadingText: {
    fontSize: 16,
    color: '#8E8E93',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 60,
  },
  errorText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#FF3B30',
    marginTop: 16,
    marginBottom: 8,
  },
  retryButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
    marginTop: 16,
  },
  retryButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 80,
  },
  emptyText: {
    fontSize: 20,
    fontWeight: '600',
    color: '#8E8E93',
    marginTop: 16,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#C7C7CC',
    marginTop: 8,
    textAlign: 'center',
  },
});

