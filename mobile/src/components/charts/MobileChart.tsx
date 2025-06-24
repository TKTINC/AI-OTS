import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
  PanGestureHandler,
  PinchGestureHandler,
  TapGestureHandler,
  State,
} from 'react-native';
import { LineChart, CandlestickChart, BarChart } from 'react-native-chart-kit';
import Svg, { Line, Circle, Text as SvgText, Rect } from 'react-native-svg';
import Animated, {
  useAnimatedGestureHandler,
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  runOnJS,
} from 'react-native-reanimated';
import { offlineStorageService } from '../services/storage/OfflineStorageService';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

export interface ChartDataPoint {
  timestamp: number;
  open?: number;
  high?: number;
  low?: number;
  close: number;
  volume?: number;
  value?: number;
}

export interface ChartConfig {
  type: 'line' | 'candlestick' | 'bar' | 'area';
  timeframe: '1m' | '5m' | '15m' | '1h' | '4h' | '1d' | '1w';
  indicators: string[];
  showVolume: boolean;
  showGrid: boolean;
  showCrosshair: boolean;
  theme: 'light' | 'dark';
}

export interface TouchPoint {
  x: number;
  y: number;
  timestamp: number;
  value: number;
}

interface MobileChartProps {
  symbol: string;
  data: ChartDataPoint[];
  config: ChartConfig;
  width?: number;
  height?: number;
  onDataPointPress?: (point: ChartDataPoint) => void;
  onZoomChange?: (zoomLevel: number) => void;
  onPanChange?: (offset: number) => void;
}

export const MobileChart: React.FC<MobileChartProps> = ({
  symbol,
  data,
  config,
  width = screenWidth - 32,
  height = 300,
  onDataPointPress,
  onZoomChange,
  onPanChange,
}) => {
  const [chartData, setChartData] = useState<ChartDataPoint[]>(data);
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: data.length });
  const [crosshair, setCrosshair] = useState<TouchPoint | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Animated values for gestures
  const scale = useSharedValue(1);
  const translateX = useSharedValue(0);
  const focalX = useSharedValue(0);
  const focalY = useSharedValue(0);

  // Refs for gesture handlers
  const panRef = useRef();
  const pinchRef = useRef();
  const tapRef = useRef();

  useEffect(() => {
    setChartData(data);
    setVisibleRange({ start: 0, end: data.length });
  }, [data]);

  // Calculate visible data based on zoom and pan
  const getVisibleData = useCallback(() => {
    const totalPoints = chartData.length;
    const visiblePoints = Math.max(10, Math.floor(totalPoints / scale.value));
    const centerIndex = Math.floor(totalPoints / 2) + Math.floor(translateX.value / 10);
    
    const start = Math.max(0, centerIndex - Math.floor(visiblePoints / 2));
    const end = Math.min(totalPoints, start + visiblePoints);
    
    return chartData.slice(start, end);
  }, [chartData, scale.value, translateX.value]);

  // Pan gesture handler
  const panGestureHandler = useAnimatedGestureHandler({
    onStart: (_, context) => {
      context.translateX = translateX.value;
    },
    onActive: (event, context) => {
      translateX.value = context.translateX + event.translationX;
      
      // Update visible range
      runOnJS(updateVisibleRange)();
    },
    onEnd: () => {
      // Snap to boundaries if needed
      const maxTranslate = (chartData.length - 10) * 10;
      if (translateX.value > 0) {
        translateX.value = withSpring(0);
      } else if (translateX.value < -maxTranslate) {
        translateX.value = withSpring(-maxTranslate);
      }
    },
  });

  // Pinch gesture handler
  const pinchGestureHandler = useAnimatedGestureHandler({
    onStart: (_, context) => {
      context.scale = scale.value;
    },
    onActive: (event, context) => {
      scale.value = Math.max(0.5, Math.min(5, context.scale * event.scale));
      focalX.value = event.focalX;
      focalY.value = event.focalY;
      
      // Notify zoom change
      runOnJS(handleZoomChange)(scale.value);
    },
    onEnd: () => {
      // Snap to reasonable zoom levels
      if (scale.value < 0.8) {
        scale.value = withSpring(0.5);
      } else if (scale.value > 4) {
        scale.value = withSpring(4);
      }
    },
  });

  // Tap gesture handler
  const tapGestureHandler = useAnimatedGestureHandler({
    onStart: (event) => {
      focalX.value = event.x;
      focalY.value = event.y;
      
      // Show crosshair and data point info
      runOnJS(handleTap)(event.x, event.y);
    },
  });

  const updateVisibleRange = () => {
    const totalPoints = chartData.length;
    const visiblePoints = Math.max(10, Math.floor(totalPoints / scale.value));
    const centerIndex = Math.floor(totalPoints / 2) + Math.floor(translateX.value / 10);
    
    const start = Math.max(0, centerIndex - Math.floor(visiblePoints / 2));
    const end = Math.min(totalPoints, start + visiblePoints);
    
    setVisibleRange({ start, end });
    
    if (onPanChange) {
      onPanChange(translateX.value);
    }
  };

  const handleZoomChange = (zoomLevel: number) => {
    if (onZoomChange) {
      onZoomChange(zoomLevel);
    }
  };

  const handleTap = (x: number, y: number) => {
    const visibleData = getVisibleData();
    const pointWidth = width / visibleData.length;
    const pointIndex = Math.floor(x / pointWidth);
    
    if (pointIndex >= 0 && pointIndex < visibleData.length) {
      const dataPoint = visibleData[pointIndex];
      
      // Calculate value based on y position
      const chartHeight = height - 40; // Account for padding
      const minValue = Math.min(...visibleData.map(d => d.close));
      const maxValue = Math.max(...visibleData.map(d => d.close));
      const valueRange = maxValue - minValue;
      const relativeY = (chartHeight - y) / chartHeight;
      const value = minValue + (relativeY * valueRange);
      
      const touchPoint: TouchPoint = {
        x,
        y,
        timestamp: dataPoint.timestamp,
        value: dataPoint.close,
      };
      
      setCrosshair(touchPoint);
      
      if (onDataPointPress) {
        onDataPointPress(dataPoint);
      }
      
      // Hide crosshair after 3 seconds
      setTimeout(() => setCrosshair(null), 3000);
    }
  };

  // Animated style for chart container
  const animatedStyle = useAnimatedStyle(() => {
    return {
      transform: [
        { translateX: translateX.value },
        { scale: scale.value },
      ],
    };
  });

  // Prepare chart data based on type
  const prepareChartData = () => {
    const visibleData = getVisibleData();
    
    switch (config.type) {
      case 'line':
        return {
          labels: visibleData.map(d => formatTimestamp(d.timestamp, config.timeframe)),
          datasets: [{
            data: visibleData.map(d => d.close),
            color: (opacity = 1) => `rgba(0, 122, 255, ${opacity})`,
            strokeWidth: 2,
          }],
        };
      
      case 'candlestick':
        return visibleData.map(d => ({
          shadowH: d.high || d.close,
          shadowL: d.low || d.close,
          open: d.open || d.close,
          close: d.close,
        }));
      
      case 'bar':
        return {
          labels: visibleData.map(d => formatTimestamp(d.timestamp, config.timeframe)),
          datasets: [{
            data: visibleData.map(d => d.volume || d.value || d.close),
          }],
        };
      
      default:
        return {
          labels: visibleData.map(d => formatTimestamp(d.timestamp, config.timeframe)),
          datasets: [{
            data: visibleData.map(d => d.close),
          }],
        };
    }
  };

  const formatTimestamp = (timestamp: number, timeframe: string): string => {
    const date = new Date(timestamp);
    
    switch (timeframe) {
      case '1m':
      case '5m':
      case '15m':
        return date.toLocaleTimeString('en-US', { 
          hour: '2-digit', 
          minute: '2-digit',
          hour12: false 
        });
      case '1h':
      case '4h':
        return date.toLocaleTimeString('en-US', { 
          hour: '2-digit',
          hour12: false 
        });
      case '1d':
        return date.toLocaleDateString('en-US', { 
          month: 'short', 
          day: 'numeric' 
        });
      case '1w':
        return date.toLocaleDateString('en-US', { 
          month: 'short', 
          day: 'numeric' 
        });
      default:
        return date.toLocaleTimeString('en-US', { 
          hour: '2-digit', 
          minute: '2-digit' 
        });
    }
  };

  const chartConfig = {
    backgroundColor: config.theme === 'dark' ? '#1E1E1E' : '#FFFFFF',
    backgroundGradientFrom: config.theme === 'dark' ? '#1E1E1E' : '#FFFFFF',
    backgroundGradientTo: config.theme === 'dark' ? '#2D2D2D' : '#F8F8F8',
    decimalPlaces: 2,
    color: (opacity = 1) => config.theme === 'dark' 
      ? `rgba(255, 255, 255, ${opacity})` 
      : `rgba(0, 0, 0, ${opacity})`,
    labelColor: (opacity = 1) => config.theme === 'dark' 
      ? `rgba(255, 255, 255, ${opacity})` 
      : `rgba(0, 0, 0, ${opacity})`,
    style: {
      borderRadius: 16,
    },
    propsForDots: {
      r: '6',
      strokeWidth: '2',
      stroke: '#007AFF',
    },
    propsForBackgroundLines: {
      strokeDasharray: config.showGrid ? '5,5' : '0,0',
      stroke: config.theme === 'dark' ? '#444444' : '#E0E0E0',
      strokeWidth: 1,
    },
  };

  const renderChart = () => {
    const chartData = prepareChartData();
    
    switch (config.type) {
      case 'line':
        return (
          <LineChart
            data={chartData}
            width={width}
            height={height}
            chartConfig={chartConfig}
            bezier={false}
            withDots={false}
            withShadow={false}
            withScrollableDot={false}
            withInnerLines={config.showGrid}
            withOuterLines={config.showGrid}
            withVerticalLines={config.showGrid}
            withHorizontalLines={config.showGrid}
          />
        );
      
      case 'bar':
        return (
          <BarChart
            data={chartData}
            width={width}
            height={height}
            chartConfig={chartConfig}
            withInnerLines={config.showGrid}
            showBarTops={false}
            showValuesOnTopOfBars={false}
          />
        );
      
      case 'candlestick':
        return (
          <CandlestickChart
            data={chartData}
            width={width}
            height={height}
            chartConfig={chartConfig}
          />
        );
      
      default:
        return (
          <LineChart
            data={chartData}
            width={width}
            height={height}
            chartConfig={chartConfig}
          />
        );
    }
  };

  const renderCrosshair = () => {
    if (!crosshair || !config.showCrosshair) return null;
    
    return (
      <Svg
        style={StyleSheet.absoluteFill}
        width={width}
        height={height}
      >
        {/* Vertical line */}
        <Line
          x1={crosshair.x}
          y1={0}
          x2={crosshair.x}
          y2={height}
          stroke={config.theme === 'dark' ? '#FFFFFF' : '#000000'}
          strokeWidth={1}
          strokeDasharray="5,5"
          opacity={0.7}
        />
        
        {/* Horizontal line */}
        <Line
          x1={0}
          y1={crosshair.y}
          x2={width}
          y2={crosshair.y}
          stroke={config.theme === 'dark' ? '#FFFFFF' : '#000000'}
          strokeWidth={1}
          strokeDasharray="5,5"
          opacity={0.7}
        />
        
        {/* Value indicator */}
        <Circle
          cx={crosshair.x}
          cy={crosshair.y}
          r={4}
          fill="#007AFF"
          stroke={config.theme === 'dark' ? '#FFFFFF' : '#000000'}
          strokeWidth={2}
        />
        
        {/* Value label */}
        <Rect
          x={crosshair.x + 10}
          y={crosshair.y - 15}
          width={80}
          height={30}
          fill={config.theme === 'dark' ? '#000000' : '#FFFFFF'}
          stroke={config.theme === 'dark' ? '#FFFFFF' : '#000000'}
          strokeWidth={1}
          rx={4}
          opacity={0.9}
        />
        
        <SvgText
          x={crosshair.x + 50}
          y={crosshair.y}
          textAnchor="middle"
          fontSize={12}
          fill={config.theme === 'dark' ? '#FFFFFF' : '#000000'}
        >
          ${crosshair.value.toFixed(2)}
        </SvgText>
      </Svg>
    );
  };

  const renderIndicators = () => {
    // This would render technical indicators like MA, RSI, etc.
    // For now, we'll return null as it would require additional calculation logic
    return null;
  };

  return (
    <View style={[styles.container, { width, height }]}>
      {/* Chart Header */}
      <View style={styles.header}>
        <Text style={[styles.symbol, { color: config.theme === 'dark' ? '#FFFFFF' : '#000000' }]}>
          {symbol}
        </Text>
        <Text style={[styles.timeframe, { color: config.theme === 'dark' ? '#CCCCCC' : '#666666' }]}>
          {config.timeframe.toUpperCase()}
        </Text>
      </View>
      
      {/* Gesture Handlers */}
      <TapGestureHandler
        ref={tapRef}
        onGestureEvent={tapGestureHandler}
        numberOfTaps={1}
      >
        <PinchGestureHandler
          ref={pinchRef}
          onGestureEvent={pinchGestureHandler}
          simultaneousHandlers={panRef}
        >
          <PanGestureHandler
            ref={panRef}
            onGestureEvent={panGestureHandler}
            simultaneousHandlers={pinchRef}
            minPointers={1}
            maxPointers={1}
          >
            <Animated.View style={[styles.chartContainer, animatedStyle]}>
              {renderChart()}
              {renderCrosshair()}
              {renderIndicators()}
            </Animated.View>
          </PanGestureHandler>
        </PinchGestureHandler>
      </TapGestureHandler>
      
      {/* Loading Overlay */}
      {isLoading && (
        <View style={styles.loadingOverlay}>
          <Text style={styles.loadingText}>Loading chart data...</Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'transparent',
    borderRadius: 12,
    overflow: 'hidden',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: 'rgba(0, 0, 0, 0.05)',
  },
  symbol: {
    fontSize: 16,
    fontWeight: '600',
  },
  timeframe: {
    fontSize: 14,
    fontWeight: '400',
  },
  chartContainer: {
    flex: 1,
    position: 'relative',
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '500',
  },
});

export default MobileChart;

