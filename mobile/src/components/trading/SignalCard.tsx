/**
 * Signal Card Component
 * Touch-optimized signal display with swipe gestures
 */

import React, { useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Dimensions,
  ViewStyle,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  useAnimatedGestureHandler,
  withSpring,
  withTiming,
  runOnJS,
} from 'react-native-reanimated';
import { PanGestureHandler, State } from 'react-native-gesture-handler';
import LinearGradient from 'react-native-linear-gradient';

import { HapticFeedback } from '../../utils/HapticFeedback';
import { formatCurrency, formatPercentage, formatTime } from '../../utils/formatters';

const { width: screenWidth } = Dimensions.get('window');
const SWIPE_THRESHOLD = screenWidth * 0.25;

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

interface SignalCardProps {
  signal: Signal;
  onPress: () => void;
  onExecute: () => void;
  onDismiss?: () => void;
  style?: ViewStyle;
}

export const SignalCard: React.FC<SignalCardProps> = ({
  signal,
  onPress,
  onExecute,
  onDismiss,
  style,
}) => {
  const translateX = useSharedValue(0);
  const opacity = useSharedValue(1);
  const scale = useSharedValue(1);

  // Get priority color
  const getPriorityColor = (priority: string) => {
    switch (priority.toLowerCase()) {
      case 'critical':
        return ['#FF3B30', '#FF6B6B'];
      case 'high':
        return ['#FF9500', '#FFB84D'];
      case 'medium':
        return ['#007AFF', '#4DA6FF'];
      default:
        return ['#34C759', '#5DD579'];
    }
  };

  // Get signal type icon
  const getSignalIcon = (signalType: string) => {
    switch (signalType.toLowerCase()) {
      case 'buy_call':
        return 'call-made';
      case 'buy_put':
        return 'call-received';
      case 'sell_call':
        return 'call-split';
      case 'sell_put':
        return 'call-merge';
      default:
        return 'trending-up';
    }
  };

  // Get risk level color
  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel.toLowerCase()) {
      case 'high':
        return '#FF3B30';
      case 'medium':
        return '#FF9500';
      default:
        return '#34C759';
    }
  };

  // Handle swipe gesture
  const gestureHandler = useAnimatedGestureHandler({
    onStart: () => {
      runOnJS(HapticFeedback.light)();
    },
    onActive: (event) => {
      translateX.value = event.translationX;
      
      // Scale down slightly when swiping
      const progress = Math.abs(event.translationX) / SWIPE_THRESHOLD;
      scale.value = 1 - progress * 0.05;
    },
    onEnd: (event) => {
      const { translationX, velocityX } = event;
      const shouldSwipe = Math.abs(translationX) > SWIPE_THRESHOLD || Math.abs(velocityX) > 500;

      if (shouldSwipe) {
        // Swipe right to execute
        if (translationX > 0) {
          translateX.value = withTiming(screenWidth, { duration: 300 });
          opacity.value = withTiming(0, { duration: 300 });
          runOnJS(HapticFeedback.success)();
          runOnJS(onExecute)();
        }
        // Swipe left to dismiss
        else if (onDismiss) {
          translateX.value = withTiming(-screenWidth, { duration: 300 });
          opacity.value = withTiming(0, { duration: 300 });
          runOnJS(HapticFeedback.medium)();
          runOnJS(onDismiss)();
        } else {
          // Bounce back if no dismiss handler
          translateX.value = withSpring(0);
          scale.value = withSpring(1);
        }
      } else {
        // Bounce back
        translateX.value = withSpring(0);
        scale.value = withSpring(1);
      }
    },
  });

  // Animated styles
  const cardStyle = useAnimatedStyle(() => {
    return {
      transform: [
        { translateX: translateX.value },
        { scale: scale.value },
      ],
      opacity: opacity.value,
    };
  });

  // Background action styles
  const executeActionStyle = useAnimatedStyle(() => {
    const progress = Math.max(0, translateX.value / SWIPE_THRESHOLD);
    return {
      opacity: progress,
      transform: [{ scale: 0.8 + progress * 0.2 }],
    };
  });

  const dismissActionStyle = useAnimatedStyle(() => {
    const progress = Math.max(0, -translateX.value / SWIPE_THRESHOLD);
    return {
      opacity: progress,
      transform: [{ scale: 0.8 + progress * 0.2 }],
    };
  });

  const priorityColors = getPriorityColor(signal.priority);

  return (
    <View style={[styles.container, style]}>
      {/* Background Actions */}
      <View style={styles.actionsContainer}>
        {/* Execute Action (Right) */}
        <Animated.View style={[styles.executeAction, executeActionStyle]}>
          <Icon name="play-arrow" size={32} color="#FFFFFF" />
          <Text style={styles.actionText}>Execute</Text>
        </Animated.View>

        {/* Dismiss Action (Left) */}
        {onDismiss && (
          <Animated.View style={[styles.dismissAction, dismissActionStyle]}>
            <Icon name="close" size={32} color="#FFFFFF" />
            <Text style={styles.actionText}>Dismiss</Text>
          </Animated.View>
        )}
      </View>

      {/* Main Card */}
      <PanGestureHandler onGestureEvent={gestureHandler}>
        <Animated.View style={cardStyle}>
          <TouchableOpacity
            style={styles.card}
            onPress={() => {
              HapticFeedback.light();
              onPress();
            }}
            activeOpacity={0.95}
          >
            <LinearGradient
              colors={['#FFFFFF', '#F8F9FA']}
              style={styles.cardGradient}
            >
              {/* Header */}
              <View style={styles.header}>
                <View style={styles.headerLeft}>
                  <View style={styles.symbolContainer}>
                    <Icon 
                      name={getSignalIcon(signal.signal_type)} 
                      size={20} 
                      color={priorityColors[0]} 
                    />
                    <Text style={styles.symbol}>{signal.symbol}</Text>
                  </View>
                  <Text style={styles.strategy}>{signal.strategy}</Text>
                </View>
                <View style={styles.headerRight}>
                  <LinearGradient
                    colors={priorityColors}
                    style={styles.priorityBadge}
                  >
                    <Text style={styles.priorityText}>
                      {signal.priority.toUpperCase()}
                    </Text>
                  </LinearGradient>
                </View>
              </View>

              {/* Signal Details */}
              <View style={styles.details}>
                <View style={styles.detailRow}>
                  <View style={styles.detailItem}>
                    <Text style={styles.detailLabel}>Confidence</Text>
                    <Text style={[styles.detailValue, { color: priorityColors[0] }]}>
                      {formatPercentage(signal.confidence)}
                    </Text>
                  </View>
                  <View style={styles.detailItem}>
                    <Text style={styles.detailLabel}>Expected Return</Text>
                    <Text style={[styles.detailValue, { color: '#34C759' }]}>
                      {formatPercentage(signal.expected_return)}
                    </Text>
                  </View>
                  <View style={styles.detailItem}>
                    <Text style={styles.detailLabel}>Risk</Text>
                    <Text style={[styles.detailValue, { color: getRiskColor(signal.risk_level) }]}>
                      {signal.risk_level}
                    </Text>
                  </View>
                </View>

                <View style={styles.detailRow}>
                  <View style={styles.detailItem}>
                    <Text style={styles.detailLabel}>Entry</Text>
                    <Text style={styles.detailValue}>
                      {formatCurrency(signal.entry_price)}
                    </Text>
                  </View>
                  <View style={styles.detailItem}>
                    <Text style={styles.detailLabel}>Target</Text>
                    <Text style={styles.detailValue}>
                      {formatCurrency(signal.target_price)}
                    </Text>
                  </View>
                  <View style={styles.detailItem}>
                    <Text style={styles.detailLabel}>Stop Loss</Text>
                    <Text style={styles.detailValue}>
                      {formatCurrency(signal.stop_loss)}
                    </Text>
                  </View>
                </View>
              </View>

              {/* Footer */}
              <View style={styles.footer}>
                <Text style={styles.timestamp}>
                  {formatTime(signal.created_at)}
                </Text>
                <Text style={styles.expiration}>
                  Expires: {formatTime(signal.expiration)}
                </Text>
              </View>

              {/* Swipe Hint */}
              <View style={styles.swipeHint}>
                <Icon name="swipe" size={16} color="#C7C7CC" />
                <Text style={styles.swipeHintText}>Swipe to execute</Text>
              </View>
            </LinearGradient>
          </TouchableOpacity>
        </Animated.View>
      </PanGestureHandler>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'relative',
  },
  actionsContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  executeAction: {
    backgroundColor: '#34C759',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  dismissAction: {
    backgroundColor: '#FF3B30',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  actionText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '600',
    marginTop: 4,
  },
  card: {
    borderRadius: 16,
    overflow: 'hidden',
    elevation: 3,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
  },
  cardGradient: {
    padding: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  headerLeft: {
    flex: 1,
  },
  symbolContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  symbol: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#000000',
    marginLeft: 8,
  },
  strategy: {
    fontSize: 14,
    color: '#8E8E93',
  },
  headerRight: {
    marginLeft: 12,
  },
  priorityBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  priorityText: {
    fontSize: 10,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
  details: {
    marginBottom: 12,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  detailItem: {
    flex: 1,
    alignItems: 'center',
  },
  detailLabel: {
    fontSize: 12,
    color: '#8E8E93',
    marginBottom: 2,
  },
  detailValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#000000',
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#E5E5EA',
  },
  timestamp: {
    fontSize: 12,
    color: '#8E8E93',
  },
  expiration: {
    fontSize: 12,
    color: '#FF9500',
  },
  swipeHint: {
    position: 'absolute',
    bottom: 8,
    right: 16,
    flexDirection: 'row',
    alignItems: 'center',
  },
  swipeHintText: {
    fontSize: 10,
    color: '#C7C7CC',
    marginLeft: 4,
  },
});

