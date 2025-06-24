import { Platform, NativeModules, DeviceEventEmitter, AppState } from 'react-native';
import { Shortcuts } from 'react-native-siri-shortcut';
import AsyncStorage from '@react-native-async-storage/async-storage';

// iOS Siri Shortcuts Integration
export interface SiriShortcut {
  activityType: string;
  title: string;
  userInfo: Record<string, any>;
  keywords: string[];
  persistentIdentifier: string;
  isEligibleForSearch: boolean;
  isEligibleForPrediction: boolean;
  suggestedInvocationPhrase: string;
  needsSave: boolean;
}

// Android Quick Settings Integration
export interface QuickSettingsTile {
  id: string;
  label: string;
  subtitle?: string;
  icon: string;
  state: 'active' | 'inactive' | 'unavailable';
  action: string;
}

// Widget Configuration
export interface WidgetConfig {
  type: 'portfolio' | 'signals' | 'watchlist' | 'performance';
  size: 'small' | 'medium' | 'large';
  updateInterval: number; // in minutes
  data: any;
}

export class NativePlatformService {
  private static instance: NativePlatformService;
  private shortcuts: SiriShortcut[] = [];
  private quickSettingsTiles: QuickSettingsTile[] = [];
  private widgets: WidgetConfig[] = [];

  private constructor() {
    this.initialize();
  }

  static getInstance(): NativePlatformService {
    if (!NativePlatformService.instance) {
      NativePlatformService.instance = new NativePlatformService();
    }
    return NativePlatformService.instance;
  }

  private async initialize(): Promise<void> {
    try {
      if (Platform.OS === 'ios') {
        await this.initializeIOSIntegration();
      } else if (Platform.OS === 'android') {
        await this.initializeAndroidIntegration();
      }
      
      // Set up app state change listener
      AppState.addEventListener('change', this.handleAppStateChange);
      
      console.log('Native platform service initialized');
    } catch (error) {
      console.error('Failed to initialize native platform service:', error);
    }
  }

  // iOS Siri Shortcuts Integration
  private async initializeIOSIntegration(): Promise<void> {
    if (Platform.OS !== 'ios') return;

    try {
      // Create default shortcuts
      await this.createDefaultSiriShortcuts();
      
      // Set up shortcut handling
      DeviceEventEmitter.addListener('SiriShortcutActivated', this.handleSiriShortcut);
      
      console.log('iOS Siri Shortcuts initialized');
    } catch (error) {
      console.error('Failed to initialize iOS integration:', error);
    }
  }

  private async createDefaultSiriShortcuts(): Promise<void> {
    const defaultShortcuts: SiriShortcut[] = [
      {
        activityType: 'com.ai-ots.check-portfolio',
        title: 'Check Portfolio',
        userInfo: { action: 'check_portfolio' },
        keywords: ['portfolio', 'balance', 'positions', 'trading'],
        persistentIdentifier: 'check-portfolio',
        isEligibleForSearch: true,
        isEligibleForPrediction: true,
        suggestedInvocationPhrase: 'Check my trading portfolio',
        needsSave: true,
      },
      {
        activityType: 'com.ai-ots.view-signals',
        title: 'View Trading Signals',
        userInfo: { action: 'view_signals' },
        keywords: ['signals', 'trading', 'opportunities', 'alerts'],
        persistentIdentifier: 'view-signals',
        isEligibleForSearch: true,
        isEligibleForPrediction: true,
        suggestedInvocationPhrase: 'Show me trading signals',
        needsSave: true,
      },
      {
        activityType: 'com.ai-ots.quick-trade',
        title: 'Quick Trade',
        userInfo: { action: 'quick_trade' },
        keywords: ['trade', 'buy', 'sell', 'execute', 'order'],
        persistentIdentifier: 'quick-trade',
        isEligibleForSearch: true,
        isEligibleForPrediction: true,
        suggestedInvocationPhrase: 'Execute a quick trade',
        needsSave: true,
      },
      {
        activityType: 'com.ai-ots.market-status',
        title: 'Market Status',
        userInfo: { action: 'market_status' },
        keywords: ['market', 'status', 'open', 'closed', 'hours'],
        persistentIdentifier: 'market-status',
        isEligibleForSearch: true,
        isEligibleForPrediction: true,
        suggestedInvocationPhrase: 'Check market status',
        needsSave: true,
      },
      {
        activityType: 'com.ai-ots.risk-check',
        title: 'Risk Check',
        userInfo: { action: 'risk_check' },
        keywords: ['risk', 'exposure', 'limits', 'safety'],
        persistentIdentifier: 'risk-check',
        isEligibleForSearch: true,
        isEligibleForPrediction: true,
        suggestedInvocationPhrase: 'Check my risk exposure',
        needsSave: true,
      },
    ];

    for (const shortcut of defaultShortcuts) {
      try {
        await this.createSiriShortcut(shortcut);
      } catch (error) {
        console.error('Failed to create Siri shortcut:', shortcut.title, error);
      }
    }

    this.shortcuts = defaultShortcuts;
  }

  async createSiriShortcut(shortcut: SiriShortcut): Promise<void> {
    if (Platform.OS !== 'ios') return;

    try {
      await Shortcuts.donateShortcut({
        activityType: shortcut.activityType,
        title: shortcut.title,
        userInfo: shortcut.userInfo,
        keywords: shortcut.keywords,
        persistentIdentifier: shortcut.persistentIdentifier,
        isEligibleForSearch: shortcut.isEligibleForSearch,
        isEligibleForPrediction: shortcut.isEligibleForPrediction,
        suggestedInvocationPhrase: shortcut.suggestedInvocationPhrase,
        needsSave: shortcut.needsSave,
      });
      
      console.log('Siri shortcut created:', shortcut.title);
    } catch (error) {
      console.error('Failed to create Siri shortcut:', error);
    }
  }

  private handleSiriShortcut = (shortcutInfo: any) => {
    console.log('Siri shortcut activated:', shortcutInfo);
    
    const { userInfo } = shortcutInfo;
    if (userInfo && userInfo.action) {
      this.executeShortcutAction(userInfo.action, userInfo);
    }
  };

  private async executeShortcutAction(action: string, params: any): Promise<void> {
    try {
      switch (action) {
        case 'check_portfolio':
          await this.handleCheckPortfolio();
          break;
        case 'view_signals':
          await this.handleViewSignals();
          break;
        case 'quick_trade':
          await this.handleQuickTrade();
          break;
        case 'market_status':
          await this.handleMarketStatus();
          break;
        case 'risk_check':
          await this.handleRiskCheck();
          break;
        default:
          console.warn('Unknown shortcut action:', action);
      }
    } catch (error) {
      console.error('Failed to execute shortcut action:', action, error);
    }
  }

  // Android Quick Settings Integration
  private async initializeAndroidIntegration(): Promise<void> {
    if (Platform.OS !== 'android') return;

    try {
      // Create default quick settings tiles
      await this.createDefaultQuickSettingsTiles();
      
      // Set up tile handling
      DeviceEventEmitter.addListener('QuickSettingsTileClicked', this.handleQuickSettingsTile);
      
      console.log('Android Quick Settings initialized');
    } catch (error) {
      console.error('Failed to initialize Android integration:', error);
    }
  }

  private async createDefaultQuickSettingsTiles(): Promise<void> {
    const defaultTiles: QuickSettingsTile[] = [
      {
        id: 'portfolio_tile',
        label: 'Portfolio',
        subtitle: 'Check balance',
        icon: 'ic_portfolio',
        state: 'inactive',
        action: 'check_portfolio',
      },
      {
        id: 'signals_tile',
        label: 'Signals',
        subtitle: 'View alerts',
        icon: 'ic_signals',
        state: 'inactive',
        action: 'view_signals',
      },
      {
        id: 'trade_tile',
        label: 'Quick Trade',
        subtitle: 'Execute order',
        icon: 'ic_trade',
        state: 'inactive',
        action: 'quick_trade',
      },
      {
        id: 'market_tile',
        label: 'Market',
        subtitle: 'Check status',
        icon: 'ic_market',
        state: 'inactive',
        action: 'market_status',
      },
    ];

    this.quickSettingsTiles = defaultTiles;
    
    // Update tiles with current data
    await this.updateQuickSettingsTiles();
  }

  async updateQuickSettingsTile(tileId: string, updates: Partial<QuickSettingsTile>): Promise<void> {
    if (Platform.OS !== 'android') return;

    try {
      const tileIndex = this.quickSettingsTiles.findIndex(tile => tile.id === tileId);
      if (tileIndex >= 0) {
        this.quickSettingsTiles[tileIndex] = { ...this.quickSettingsTiles[tileIndex], ...updates };
        
        // Update native tile
        if (NativeModules.QuickSettingsModule) {
          await NativeModules.QuickSettingsModule.updateTile(tileId, updates);
        }
      }
    } catch (error) {
      console.error('Failed to update quick settings tile:', error);
    }
  }

  private async updateQuickSettingsTiles(): Promise<void> {
    try {
      // Update portfolio tile with current balance
      const portfolioData = await this.getPortfolioSummary();
      await this.updateQuickSettingsTile('portfolio_tile', {
        subtitle: `$${portfolioData.totalValue.toLocaleString()}`,
        state: portfolioData.isPositive ? 'active' : 'inactive',
      });

      // Update signals tile with active signal count
      const signalsData = await this.getSignalsSummary();
      await this.updateQuickSettingsTile('signals_tile', {
        subtitle: `${signalsData.activeCount} active`,
        state: signalsData.activeCount > 0 ? 'active' : 'inactive',
      });

      // Update market tile with market status
      const marketData = await this.getMarketStatus();
      await this.updateQuickSettingsTile('market_tile', {
        subtitle: marketData.isOpen ? 'Open' : 'Closed',
        state: marketData.isOpen ? 'active' : 'inactive',
      });

    } catch (error) {
      console.error('Failed to update quick settings tiles:', error);
    }
  }

  private handleQuickSettingsTile = (tileInfo: any) => {
    console.log('Quick settings tile clicked:', tileInfo);
    
    const { action } = tileInfo;
    if (action) {
      this.executeShortcutAction(action, tileInfo);
    }
  };

  // Widget Management
  async createWidget(config: WidgetConfig): Promise<string> {
    try {
      const widgetId = `widget_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const widget = { ...config, id: widgetId };
      
      this.widgets.push(widget);
      await this.saveWidgets();
      
      // Create native widget
      if (Platform.OS === 'ios' && NativeModules.WidgetModule) {
        await NativeModules.WidgetModule.createWidget(widgetId, config);
      } else if (Platform.OS === 'android' && NativeModules.AppWidgetModule) {
        await NativeModules.AppWidgetModule.createWidget(widgetId, config);
      }
      
      console.log('Widget created:', widgetId, config.type);
      return widgetId;
    } catch (error) {
      console.error('Failed to create widget:', error);
      throw error;
    }
  }

  async updateWidget(widgetId: string, data: any): Promise<void> {
    try {
      const widget = this.widgets.find(w => w.id === widgetId);
      if (!widget) {
        console.warn('Widget not found:', widgetId);
        return;
      }
      
      widget.data = data;
      await this.saveWidgets();
      
      // Update native widget
      if (Platform.OS === 'ios' && NativeModules.WidgetModule) {
        await NativeModules.WidgetModule.updateWidget(widgetId, data);
      } else if (Platform.OS === 'android' && NativeModules.AppWidgetModule) {
        await NativeModules.AppWidgetModule.updateWidget(widgetId, data);
      }
      
      console.log('Widget updated:', widgetId);
    } catch (error) {
      console.error('Failed to update widget:', error);
    }
  }

  async deleteWidget(widgetId: string): Promise<void> {
    try {
      this.widgets = this.widgets.filter(w => w.id !== widgetId);
      await this.saveWidgets();
      
      // Delete native widget
      if (Platform.OS === 'ios' && NativeModules.WidgetModule) {
        await NativeModules.WidgetModule.deleteWidget(widgetId);
      } else if (Platform.OS === 'android' && NativeModules.AppWidgetModule) {
        await NativeModules.AppWidgetModule.deleteWidget(widgetId);
      }
      
      console.log('Widget deleted:', widgetId);
    } catch (error) {
      console.error('Failed to delete widget:', error);
    }
  }

  // Data Fetching Methods
  private async getPortfolioSummary(): Promise<{ totalValue: number; isPositive: boolean }> {
    try {
      // This would fetch real portfolio data
      const portfolioData = await AsyncStorage.getItem('portfolio_summary');
      if (portfolioData) {
        return JSON.parse(portfolioData);
      }
      
      return { totalValue: 0, isPositive: false };
    } catch (error) {
      console.error('Failed to get portfolio summary:', error);
      return { totalValue: 0, isPositive: false };
    }
  }

  private async getSignalsSummary(): Promise<{ activeCount: number }> {
    try {
      // This would fetch real signals data
      const signalsData = await AsyncStorage.getItem('signals_summary');
      if (signalsData) {
        return JSON.parse(signalsData);
      }
      
      return { activeCount: 0 };
    } catch (error) {
      console.error('Failed to get signals summary:', error);
      return { activeCount: 0 };
    }
  }

  private async getMarketStatus(): Promise<{ isOpen: boolean }> {
    try {
      // This would fetch real market status
      const marketData = await AsyncStorage.getItem('market_status');
      if (marketData) {
        return JSON.parse(marketData);
      }
      
      // Default to market hours check
      const now = new Date();
      const hour = now.getHours();
      const day = now.getDay();
      
      // Simple market hours check (9:30 AM - 4:00 PM EST, Monday-Friday)
      const isWeekday = day >= 1 && day <= 5;
      const isMarketHours = hour >= 9 && hour < 16;
      
      return { isOpen: isWeekday && isMarketHours };
    } catch (error) {
      console.error('Failed to get market status:', error);
      return { isOpen: false };
    }
  }

  // Action Handlers
  private async handleCheckPortfolio(): Promise<void> {
    console.log('Handling check portfolio action');
    // This would navigate to portfolio screen or show portfolio data
  }

  private async handleViewSignals(): Promise<void> {
    console.log('Handling view signals action');
    // This would navigate to signals screen or show active signals
  }

  private async handleQuickTrade(): Promise<void> {
    console.log('Handling quick trade action');
    // This would open quick trade interface
  }

  private async handleMarketStatus(): Promise<void> {
    console.log('Handling market status action');
    // This would show market status information
  }

  private async handleRiskCheck(): Promise<void> {
    console.log('Handling risk check action');
    // This would show risk metrics and alerts
  }

  // App State Management
  private handleAppStateChange = (nextAppState: string) => {
    if (nextAppState === 'active') {
      // App became active, update widgets and tiles
      this.updateQuickSettingsTiles();
      this.updateAllWidgets();
    }
  };

  private async updateAllWidgets(): Promise<void> {
    for (const widget of this.widgets) {
      try {
        let data;
        
        switch (widget.type) {
          case 'portfolio':
            data = await this.getPortfolioSummary();
            break;
          case 'signals':
            data = await this.getSignalsSummary();
            break;
          case 'watchlist':
            data = await this.getWatchlistData();
            break;
          case 'performance':
            data = await this.getPerformanceData();
            break;
          default:
            continue;
        }
        
        await this.updateWidget(widget.id!, data);
      } catch (error) {
        console.error('Failed to update widget:', widget.id, error);
      }
    }
  }

  private async getWatchlistData(): Promise<any> {
    // This would fetch watchlist data
    return { symbols: [], count: 0 };
  }

  private async getPerformanceData(): Promise<any> {
    // This would fetch performance data
    return { dailyPnL: 0, totalReturn: 0 };
  }

  // Storage Methods
  private async saveWidgets(): Promise<void> {
    try {
      await AsyncStorage.setItem('widgets', JSON.stringify(this.widgets));
    } catch (error) {
      console.error('Failed to save widgets:', error);
    }
  }

  private async loadWidgets(): Promise<void> {
    try {
      const stored = await AsyncStorage.getItem('widgets');
      if (stored) {
        this.widgets = JSON.parse(stored);
      }
    } catch (error) {
      console.error('Failed to load widgets:', error);
    }
  }

  // Public API
  getShortcuts(): SiriShortcut[] {
    return this.shortcuts;
  }

  getQuickSettingsTiles(): QuickSettingsTile[] {
    return this.quickSettingsTiles;
  }

  getWidgets(): WidgetConfig[] {
    return this.widgets;
  }

  // Cleanup
  cleanup(): void {
    AppState.removeEventListener('change', this.handleAppStateChange);
    DeviceEventEmitter.removeAllListeners('SiriShortcutActivated');
    DeviceEventEmitter.removeAllListeners('QuickSettingsTileClicked');
  }
}

// Singleton instance
export const nativePlatformService = NativePlatformService.getInstance();

