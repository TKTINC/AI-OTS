import { Platform, AppState, Dimensions, PixelRatio } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-netinfo/netinfo';
import { InteractionManager } from 'react-native';

export interface PerformanceMetrics {
  memoryUsage: number;
  cpuUsage: number;
  batteryLevel: number;
  networkType: string;
  renderTime: number;
  jsHeapSize: number;
  nativeHeapSize: number;
  imageMemory: number;
}

export interface OptimizationConfig {
  enableMemoryOptimization: boolean;
  enableBatteryOptimization: boolean;
  enableNetworkOptimization: boolean;
  enableRenderOptimization: boolean;
  maxMemoryUsage: number; // MB
  maxImageCacheSize: number; // MB
  networkTimeout: number; // ms
  renderBudget: number; // ms per frame
}

export interface NetworkOptimization {
  enableCompression: boolean;
  enableCaching: boolean;
  batchRequests: boolean;
  prioritizeRequests: boolean;
  adaptiveQuality: boolean;
}

export interface MemoryOptimization {
  enableImageOptimization: boolean;
  enableComponentUnmounting: boolean;
  enableDataPagination: boolean;
  enableGarbageCollection: boolean;
  maxCacheEntries: number;
}

export interface BatteryOptimization {
  enableBackgroundThrottling: boolean;
  enableLocationOptimization: boolean;
  enableAnimationReduction: boolean;
  enableRefreshRateAdaptation: boolean;
  enableCPUThrottling: boolean;
}

export class PerformanceOptimizationService {
  private static instance: PerformanceOptimizationService;
  private config: OptimizationConfig;
  private metrics: PerformanceMetrics;
  private isMonitoring = false;
  private monitoringInterval: NodeJS.Timeout | null = null;
  private imageCache = new Map<string, { data: any; timestamp: number; size: number }>();
  private requestQueue: Array<{ request: () => Promise<any>; priority: number }> = [];
  private isProcessingQueue = false;
  private renderTimes: number[] = [];
  private memoryWarningCount = 0;

  private constructor() {
    this.config = this.getDefaultConfig();
    this.metrics = this.getDefaultMetrics();
    this.initialize();
  }

  static getInstance(): PerformanceOptimizationService {
    if (!PerformanceOptimizationService.instance) {
      PerformanceOptimizationService.instance = new PerformanceOptimizationService();
    }
    return PerformanceOptimizationService.instance;
  }

  private getDefaultConfig(): OptimizationConfig {
    return {
      enableMemoryOptimization: true,
      enableBatteryOptimization: true,
      enableNetworkOptimization: true,
      enableRenderOptimization: true,
      maxMemoryUsage: 150, // 150MB
      maxImageCacheSize: 50, // 50MB
      networkTimeout: 10000, // 10 seconds
      renderBudget: 16.67, // 60 FPS
    };
  }

  private getDefaultMetrics(): PerformanceMetrics {
    return {
      memoryUsage: 0,
      cpuUsage: 0,
      batteryLevel: 100,
      networkType: 'unknown',
      renderTime: 0,
      jsHeapSize: 0,
      nativeHeapSize: 0,
      imageMemory: 0,
    };
  }

  private async initialize(): Promise<void> {
    try {
      // Load saved configuration
      await this.loadConfig();
      
      // Set up app state monitoring
      AppState.addEventListener('change', this.handleAppStateChange);
      
      // Set up memory warning listener
      if (Platform.OS === 'ios') {
        // iOS memory warning handling would go here
      }
      
      // Start performance monitoring
      this.startMonitoring();
      
      console.log('Performance optimization service initialized');
    } catch (error) {
      console.error('Failed to initialize performance optimization service:', error);
    }
  }

  // Memory Optimization
  async optimizeMemoryUsage(): Promise<void> {
    if (!this.config.enableMemoryOptimization) return;

    try {
      // Clear expired cache entries
      await this.clearExpiredCache();
      
      // Optimize image cache
      await this.optimizeImageCache();
      
      // Force garbage collection if available
      if (global.gc && this.metrics.memoryUsage > this.config.maxMemoryUsage * 0.8) {
        global.gc();
        console.log('Forced garbage collection');
      }
      
      // Clear unnecessary data
      await this.clearUnnecessaryData();
      
      console.log('Memory optimization completed');
    } catch (error) {
      console.error('Memory optimization failed:', error);
    }
  }

  private async clearExpiredCache(): Promise<void> {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const cacheKeys = keys.filter(key => key.startsWith('cache_'));
      const now = Date.now();
      
      for (const key of cacheKeys) {
        const cached = await AsyncStorage.getItem(key);
        if (cached) {
          const data = JSON.parse(cached);
          if (data.timestamp && now - data.timestamp > data.ttl) {
            await AsyncStorage.removeItem(key);
          }
        }
      }
    } catch (error) {
      console.error('Failed to clear expired cache:', error);
    }
  }

  private async optimizeImageCache(): Promise<void> {
    const maxSize = this.config.maxImageCacheSize * 1024 * 1024; // Convert to bytes
    let currentSize = 0;
    
    // Calculate current cache size
    for (const [key, value] of this.imageCache) {
      currentSize += value.size;
    }
    
    if (currentSize > maxSize) {
      // Sort by timestamp (oldest first)
      const entries = Array.from(this.imageCache.entries())
        .sort((a, b) => a[1].timestamp - b[1].timestamp);
      
      // Remove oldest entries until under limit
      for (const [key, value] of entries) {
        if (currentSize <= maxSize) break;
        
        this.imageCache.delete(key);
        currentSize -= value.size;
      }
      
      console.log(`Image cache optimized: ${entries.length - this.imageCache.size} entries removed`);
    }
  }

  private async clearUnnecessaryData(): Promise<void> {
    try {
      // Clear old notification history
      const notifications = await AsyncStorage.getItem('notifications');
      if (notifications) {
        const parsed = JSON.parse(notifications);
        if (parsed.length > 100) {
          const recent = parsed.slice(0, 100);
          await AsyncStorage.setItem('notifications', JSON.stringify(recent));
        }
      }
      
      // Clear old performance metrics
      this.renderTimes = this.renderTimes.slice(-100); // Keep last 100 measurements
      
    } catch (error) {
      console.error('Failed to clear unnecessary data:', error);
    }
  }

  // Battery Optimization
  async optimizeBatteryUsage(): Promise<void> {
    if (!this.config.enableBatteryOptimization) return;

    try {
      const batteryLevel = await this.getBatteryLevel();
      
      if (batteryLevel < 20) {
        // Enable aggressive battery saving
        await this.enableAggressiveBatterySaving();
      } else if (batteryLevel < 50) {
        // Enable moderate battery saving
        await this.enableModerateBatterySaving();
      }
      
      console.log('Battery optimization applied for level:', batteryLevel);
    } catch (error) {
      console.error('Battery optimization failed:', error);
    }
  }

  private async enableAggressiveBatterySaving(): Promise<void> {
    // Reduce update frequencies
    await AsyncStorage.setItem('update_interval', '60000'); // 1 minute
    
    // Disable animations
    await AsyncStorage.setItem('animations_enabled', 'false');
    
    // Reduce chart quality
    await AsyncStorage.setItem('chart_quality', 'low');
    
    // Disable background sync
    await AsyncStorage.setItem('background_sync', 'false');
    
    console.log('Aggressive battery saving enabled');
  }

  private async enableModerateBatterySaving(): Promise<void> {
    // Moderate update frequencies
    await AsyncStorage.setItem('update_interval', '30000'); // 30 seconds
    
    // Reduce chart quality
    await AsyncStorage.setItem('chart_quality', 'medium');
    
    // Limit background sync
    await AsyncStorage.setItem('background_sync_interval', '300000'); // 5 minutes
    
    console.log('Moderate battery saving enabled');
  }

  private async getBatteryLevel(): Promise<number> {
    try {
      // This would use a battery info library
      // For now, return a mock value
      return 75;
    } catch (error) {
      console.error('Failed to get battery level:', error);
      return 100;
    }
  }

  // Network Optimization
  async optimizeNetworkUsage(): Promise<void> {
    if (!this.config.enableNetworkOptimization) return;

    try {
      const networkInfo = await NetInfo.fetch();
      
      if (networkInfo.type === 'cellular') {
        await this.enableCellularOptimization();
      } else if (networkInfo.type === 'wifi') {
        await this.enableWiFiOptimization();
      }
      
      // Process request queue
      await this.processRequestQueue();
      
      console.log('Network optimization applied for:', networkInfo.type);
    } catch (error) {
      console.error('Network optimization failed:', error);
    }
  }

  private async enableCellularOptimization(): Promise<void> {
    // Reduce image quality
    await AsyncStorage.setItem('image_quality', 'low');
    
    // Enable request batching
    await AsyncStorage.setItem('batch_requests', 'true');
    
    // Reduce update frequency
    await AsyncStorage.setItem('cellular_update_interval', '60000');
    
    // Enable compression
    await AsyncStorage.setItem('enable_compression', 'true');
    
    console.log('Cellular optimization enabled');
  }

  private async enableWiFiOptimization(): Promise<void> {
    // High image quality
    await AsyncStorage.setItem('image_quality', 'high');
    
    // Disable request batching
    await AsyncStorage.setItem('batch_requests', 'false');
    
    // Normal update frequency
    await AsyncStorage.setItem('wifi_update_interval', '5000');
    
    console.log('WiFi optimization enabled');
  }

  async addToRequestQueue(request: () => Promise<any>, priority: number = 1): Promise<void> {
    this.requestQueue.push({ request, priority });
    this.requestQueue.sort((a, b) => b.priority - a.priority); // Higher priority first
    
    if (!this.isProcessingQueue) {
      await this.processRequestQueue();
    }
  }

  private async processRequestQueue(): Promise<void> {
    if (this.isProcessingQueue || this.requestQueue.length === 0) return;
    
    this.isProcessingQueue = true;
    
    try {
      const batchSize = await this.getBatchSize();
      const batch = this.requestQueue.splice(0, batchSize);
      
      // Process requests in parallel
      const promises = batch.map(item => item.request());
      await Promise.allSettled(promises);
      
      // Continue processing if more requests exist
      if (this.requestQueue.length > 0) {
        setTimeout(() => this.processRequestQueue(), 100);
      }
    } catch (error) {
      console.error('Request queue processing failed:', error);
    } finally {
      this.isProcessingQueue = false;
    }
  }

  private async getBatchSize(): Promise<number> {
    const networkInfo = await NetInfo.fetch();
    
    switch (networkInfo.type) {
      case 'wifi':
        return 10;
      case 'cellular':
        return 3;
      default:
        return 1;
    }
  }

  // Render Optimization
  async optimizeRenderPerformance(): Promise<void> {
    if (!this.config.enableRenderOptimization) return;

    try {
      const averageRenderTime = this.getAverageRenderTime();
      
      if (averageRenderTime > this.config.renderBudget) {
        await this.enableRenderOptimizations();
      }
      
      console.log('Render optimization applied, average time:', averageRenderTime);
    } catch (error) {
      console.error('Render optimization failed:', error);
    }
  }

  recordRenderTime(renderTime: number): void {
    this.renderTimes.push(renderTime);
    
    // Keep only last 100 measurements
    if (this.renderTimes.length > 100) {
      this.renderTimes = this.renderTimes.slice(-100);
    }
    
    // Update metrics
    this.metrics.renderTime = renderTime;
  }

  private getAverageRenderTime(): number {
    if (this.renderTimes.length === 0) return 0;
    
    const sum = this.renderTimes.reduce((acc, time) => acc + time, 0);
    return sum / this.renderTimes.length;
  }

  private async enableRenderOptimizations(): Promise<void> {
    // Reduce animation complexity
    await AsyncStorage.setItem('animation_complexity', 'low');
    
    // Enable view recycling
    await AsyncStorage.setItem('enable_view_recycling', 'true');
    
    // Reduce chart update frequency
    await AsyncStorage.setItem('chart_update_interval', '1000');
    
    // Enable lazy loading
    await AsyncStorage.setItem('enable_lazy_loading', 'true');
    
    console.log('Render optimizations enabled');
  }

  // Performance Monitoring
  private startMonitoring(): void {
    if (this.isMonitoring) return;
    
    this.isMonitoring = true;
    this.monitoringInterval = setInterval(async () => {
      await this.updateMetrics();
      await this.checkPerformanceThresholds();
    }, 5000); // Monitor every 5 seconds
    
    console.log('Performance monitoring started');
  }

  private stopMonitoring(): void {
    if (!this.isMonitoring) return;
    
    this.isMonitoring = false;
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
    
    console.log('Performance monitoring stopped');
  }

  private async updateMetrics(): Promise<void> {
    try {
      // Update memory usage
      this.metrics.memoryUsage = await this.getMemoryUsage();
      
      // Update network type
      const networkInfo = await NetInfo.fetch();
      this.metrics.networkType = networkInfo.type || 'unknown';
      
      // Update battery level
      this.metrics.batteryLevel = await this.getBatteryLevel();
      
      // Update image cache size
      let imageCacheSize = 0;
      for (const [key, value] of this.imageCache) {
        imageCacheSize += value.size;
      }
      this.metrics.imageMemory = imageCacheSize / (1024 * 1024); // Convert to MB
      
    } catch (error) {
      console.error('Failed to update metrics:', error);
    }
  }

  private async getMemoryUsage(): Promise<number> {
    try {
      // This would use a memory info library
      // For now, return a mock value based on image cache and other factors
      const baseMemory = 50; // Base app memory
      const cacheMemory = this.metrics.imageMemory;
      return baseMemory + cacheMemory;
    } catch (error) {
      console.error('Failed to get memory usage:', error);
      return 0;
    }
  }

  private async checkPerformanceThresholds(): Promise<void> {
    // Check memory usage
    if (this.metrics.memoryUsage > this.config.maxMemoryUsage) {
      console.warn('Memory usage threshold exceeded:', this.metrics.memoryUsage);
      await this.optimizeMemoryUsage();
      this.memoryWarningCount++;
    }
    
    // Check battery level
    if (this.metrics.batteryLevel < 20) {
      console.warn('Low battery detected:', this.metrics.batteryLevel);
      await this.optimizeBatteryUsage();
    }
    
    // Check render performance
    const averageRenderTime = this.getAverageRenderTime();
    if (averageRenderTime > this.config.renderBudget * 1.5) {
      console.warn('Render performance degraded:', averageRenderTime);
      await this.optimizeRenderPerformance();
    }
  }

  // App State Management
  private handleAppStateChange = async (nextAppState: string) => {
    if (nextAppState === 'background') {
      // App went to background
      await this.enableBackgroundOptimizations();
    } else if (nextAppState === 'active') {
      // App became active
      await this.disableBackgroundOptimizations();
    }
  };

  private async enableBackgroundOptimizations(): Promise<void> {
    // Reduce monitoring frequency
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = setInterval(async () => {
        await this.updateMetrics();
      }, 30000); // Monitor every 30 seconds in background
    }
    
    // Clear non-essential cache
    await this.clearExpiredCache();
    
    console.log('Background optimizations enabled');
  }

  private async disableBackgroundOptimizations(): Promise<void> {
    // Restore normal monitoring frequency
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = setInterval(async () => {
        await this.updateMetrics();
        await this.checkPerformanceThresholds();
      }, 5000); // Monitor every 5 seconds when active
    }
    
    console.log('Background optimizations disabled');
  }

  // Configuration Management
  async updateConfig(updates: Partial<OptimizationConfig>): Promise<void> {
    this.config = { ...this.config, ...updates };
    await this.saveConfig();
    console.log('Performance configuration updated');
  }

  private async loadConfig(): Promise<void> {
    try {
      const stored = await AsyncStorage.getItem('performance_config');
      if (stored) {
        this.config = { ...this.config, ...JSON.parse(stored) };
      }
    } catch (error) {
      console.error('Failed to load performance config:', error);
    }
  }

  private async saveConfig(): Promise<void> {
    try {
      await AsyncStorage.setItem('performance_config', JSON.stringify(this.config));
    } catch (error) {
      console.error('Failed to save performance config:', error);
    }
  }

  // Public API
  getMetrics(): PerformanceMetrics {
    return { ...this.metrics };
  }

  getConfig(): OptimizationConfig {
    return { ...this.config };
  }

  async runFullOptimization(): Promise<void> {
    console.log('Running full performance optimization...');
    
    await Promise.all([
      this.optimizeMemoryUsage(),
      this.optimizeBatteryUsage(),
      this.optimizeNetworkUsage(),
      this.optimizeRenderPerformance(),
    ]);
    
    console.log('Full performance optimization completed');
  }

  // Cleanup
  cleanup(): void {
    this.stopMonitoring();
    AppState.removeEventListener('change', this.handleAppStateChange);
    this.imageCache.clear();
    this.requestQueue = [];
  }
}

// Singleton instance
export const performanceOptimizationService = PerformanceOptimizationService.getInstance();

