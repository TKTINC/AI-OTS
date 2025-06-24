import { Platform } from 'react-native';
import PushNotification, { Importance } from 'react-native-push-notification';
import PushNotificationIOS from '@react-native-async-storage/async-storage';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { io, Socket } from 'socket.io-client';

export interface NotificationPreferences {
  enabled: boolean;
  signalAlerts: boolean;
  riskAlerts: boolean;
  portfolioUpdates: boolean;
  marketNews: boolean;
  quietHours: {
    enabled: boolean;
    start: string; // "22:00"
    end: string;   // "08:00"
  };
  priorities: {
    critical: boolean;
    high: boolean;
    medium: boolean;
    low: boolean;
  };
  channels: {
    push: boolean;
    inApp: boolean;
    sound: boolean;
    vibration: boolean;
  };
}

export interface SignalNotification {
  id: string;
  type: 'signal' | 'risk' | 'portfolio' | 'market';
  priority: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  message: string;
  data?: any;
  timestamp: number;
  read: boolean;
}

export class NotificationService {
  private socket: Socket | null = null;
  private preferences: NotificationPreferences;
  private notifications: SignalNotification[] = [];
  private isInitialized = false;

  constructor() {
    this.preferences = this.getDefaultPreferences();
    this.initializeNotifications();
  }

  private getDefaultPreferences(): NotificationPreferences {
    return {
      enabled: true,
      signalAlerts: true,
      riskAlerts: true,
      portfolioUpdates: true,
      marketNews: false,
      quietHours: {
        enabled: true,
        start: "22:00",
        end: "08:00"
      },
      priorities: {
        critical: true,
        high: true,
        medium: true,
        low: false
      },
      channels: {
        push: true,
        inApp: true,
        sound: true,
        vibration: true
      }
    };
  }

  async initializeNotifications(): Promise<void> {
    if (this.isInitialized) return;

    try {
      // Load saved preferences
      await this.loadPreferences();

      // Configure push notifications
      PushNotification.configure({
        onRegister: (token) => {
          console.log('Push notification token:', token);
          this.registerDeviceToken(token.token);
        },
        onNotification: (notification) => {
          console.log('Notification received:', notification);
          this.handleNotificationReceived(notification);
        },
        onAction: (notification) => {
          console.log('Notification action:', notification);
          this.handleNotificationAction(notification);
        },
        onRegistrationError: (err) => {
          console.error('Push notification registration error:', err);
        },
        permissions: {
          alert: true,
          badge: true,
          sound: true,
        },
        popInitialNotification: true,
        requestPermissions: Platform.OS === 'ios',
      });

      // Create notification channels for Android
      if (Platform.OS === 'android') {
        this.createNotificationChannels();
      }

      // Initialize WebSocket connection for real-time notifications
      await this.initializeWebSocket();

      this.isInitialized = true;
      console.log('Notification service initialized successfully');
    } catch (error) {
      console.error('Failed to initialize notification service:', error);
    }
  }

  private createNotificationChannels(): void {
    const channels = [
      {
        channelId: 'signals_critical',
        channelName: 'Critical Trading Signals',
        channelDescription: 'Critical trading signals requiring immediate attention',
        importance: Importance.HIGH,
        vibrate: true,
        sound: 'default',
      },
      {
        channelId: 'signals_high',
        channelName: 'High Priority Signals',
        channelDescription: 'High priority trading signals',
        importance: Importance.DEFAULT,
        vibrate: true,
        sound: 'default',
      },
      {
        channelId: 'signals_medium',
        channelName: 'Medium Priority Signals',
        channelDescription: 'Medium priority trading signals',
        importance: Importance.DEFAULT,
        vibrate: false,
        sound: 'default',
      },
      {
        channelId: 'risk_alerts',
        channelName: 'Risk Alerts',
        channelDescription: 'Portfolio risk and compliance alerts',
        importance: Importance.HIGH,
        vibrate: true,
        sound: 'default',
      },
      {
        channelId: 'portfolio_updates',
        channelName: 'Portfolio Updates',
        channelDescription: 'Portfolio performance and position updates',
        importance: Importance.LOW,
        vibrate: false,
        sound: null,
      },
    ];

    channels.forEach(channel => {
      PushNotification.createChannel(channel, () => {
        console.log(`Created notification channel: ${channel.channelId}`);
      });
    });
  }

  private async initializeWebSocket(): Promise<void> {
    try {
      const API_BASE_URL = await AsyncStorage.getItem('API_BASE_URL') || 'http://localhost:8004';
      
      this.socket = io(`${API_BASE_URL}`, {
        transports: ['websocket'],
        autoConnect: true,
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
      });

      this.socket.on('connect', () => {
        console.log('WebSocket connected for notifications');
        this.joinNotificationRoom();
      });

      this.socket.on('disconnect', () => {
        console.log('WebSocket disconnected');
      });

      this.socket.on('new_signal', (signal: any) => {
        this.handleSignalNotification(signal);
      });

      this.socket.on('risk_alert', (alert: any) => {
        this.handleRiskAlert(alert);
      });

      this.socket.on('portfolio_update', (update: any) => {
        this.handlePortfolioUpdate(update);
      });

    } catch (error) {
      console.error('Failed to initialize WebSocket:', error);
    }
  }

  private async joinNotificationRoom(): Promise<void> {
    if (!this.socket) return;

    try {
      const userId = await AsyncStorage.getItem('user_id') || 'mobile_user';
      this.socket.emit('join_room', { user_id: userId });
      console.log('Joined notification room for user:', userId);
    } catch (error) {
      console.error('Failed to join notification room:', error);
    }
  }

  private async registerDeviceToken(token: string): Promise<void> {
    try {
      await AsyncStorage.setItem('device_token', token);
      
      // Send token to backend for push notification targeting
      const API_BASE_URL = await AsyncStorage.getItem('API_BASE_URL') || 'http://localhost:8004';
      const userId = await AsyncStorage.getItem('user_id') || 'mobile_user';

      const response = await fetch(`${API_BASE_URL}/api/v1/notifications/register-device`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          device_token: token,
          platform: Platform.OS,
        }),
      });

      if (response.ok) {
        console.log('Device token registered successfully');
      }
    } catch (error) {
      console.error('Failed to register device token:', error);
    }
  }

  private handleNotificationReceived(notification: any): void {
    // Handle notification when app is in foreground
    if (notification.foreground) {
      this.addInAppNotification({
        id: notification.id || Date.now().toString(),
        type: notification.data?.type || 'signal',
        priority: notification.data?.priority || 'medium',
        title: notification.title,
        message: notification.message,
        data: notification.data,
        timestamp: Date.now(),
        read: false,
      });
    }

    // Mark notification as delivered
    if (Platform.OS === 'ios') {
      notification.finish(PushNotificationIOS.FetchResult.NoData);
    }
  }

  private handleNotificationAction(notification: any): void {
    // Handle notification tap/action
    console.log('Notification action:', notification.action);
    
    if (notification.action === 'execute_signal') {
      // Navigate to signal execution screen
      this.navigateToSignal(notification.data?.signal_id);
    } else if (notification.action === 'view_portfolio') {
      // Navigate to portfolio screen
      this.navigateToPortfolio();
    }
  }

  private async handleSignalNotification(signal: any): Promise<void> {
    if (!this.shouldShowNotification('signal', signal.priority)) {
      return;
    }

    const notification: SignalNotification = {
      id: signal.signal_id,
      type: 'signal',
      priority: signal.priority,
      title: `${signal.priority.toUpperCase()} Trading Signal`,
      message: `${signal.signal_type} ${signal.symbol} - ${signal.expected_return}% expected return`,
      data: signal,
      timestamp: Date.now(),
      read: false,
    };

    // Add to in-app notifications
    this.addInAppNotification(notification);

    // Send push notification
    if (this.preferences.channels.push) {
      this.sendPushNotification(notification);
    }
  }

  private async handleRiskAlert(alert: any): Promise<void> {
    if (!this.shouldShowNotification('risk', alert.severity)) {
      return;
    }

    const notification: SignalNotification = {
      id: alert.alert_id,
      type: 'risk',
      priority: alert.severity,
      title: `Risk Alert - ${alert.alert_type}`,
      message: alert.message,
      data: alert,
      timestamp: Date.now(),
      read: false,
    };

    this.addInAppNotification(notification);

    if (this.preferences.channels.push) {
      this.sendPushNotification(notification);
    }
  }

  private async handlePortfolioUpdate(update: any): Promise<void> {
    if (!this.preferences.portfolioUpdates) {
      return;
    }

    const notification: SignalNotification = {
      id: `portfolio_${Date.now()}`,
      type: 'portfolio',
      priority: 'low',
      title: 'Portfolio Update',
      message: `Portfolio value: ${update.total_value} (${update.daily_pnl >= 0 ? '+' : ''}${update.daily_pnl})`,
      data: update,
      timestamp: Date.now(),
      read: false,
    };

    this.addInAppNotification(notification);
  }

  private shouldShowNotification(type: string, priority: string): boolean {
    if (!this.preferences.enabled) return false;

    // Check type preferences
    switch (type) {
      case 'signal':
        if (!this.preferences.signalAlerts) return false;
        break;
      case 'risk':
        if (!this.preferences.riskAlerts) return false;
        break;
      case 'portfolio':
        if (!this.preferences.portfolioUpdates) return false;
        break;
      default:
        return false;
    }

    // Check priority preferences
    if (!this.preferences.priorities[priority as keyof typeof this.preferences.priorities]) {
      return false;
    }

    // Check quiet hours
    if (this.preferences.quietHours.enabled && priority !== 'critical') {
      if (this.isInQuietHours()) {
        return false;
      }
    }

    return true;
  }

  private isInQuietHours(): boolean {
    const now = new Date();
    const currentTime = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
    
    const { start, end } = this.preferences.quietHours;
    
    if (start < end) {
      // Same day quiet hours (e.g., 14:00 to 18:00)
      return currentTime >= start && currentTime <= end;
    } else {
      // Overnight quiet hours (e.g., 22:00 to 08:00)
      return currentTime >= start || currentTime <= end;
    }
  }

  private sendPushNotification(notification: SignalNotification): void {
    const channelId = this.getChannelId(notification.type, notification.priority);
    
    PushNotification.localNotification({
      id: notification.id,
      channelId,
      title: notification.title,
      message: notification.message,
      playSound: this.preferences.channels.sound,
      vibrate: this.preferences.channels.vibration,
      priority: notification.priority === 'critical' ? 'high' : 'default',
      importance: notification.priority === 'critical' ? 'high' : 'default',
      data: notification.data,
      actions: this.getNotificationActions(notification),
    });
  }

  private getChannelId(type: string, priority: string): string {
    if (type === 'risk') return 'risk_alerts';
    if (type === 'portfolio') return 'portfolio_updates';
    
    // Signal notifications
    switch (priority) {
      case 'critical': return 'signals_critical';
      case 'high': return 'signals_high';
      case 'medium': return 'signals_medium';
      default: return 'signals_medium';
    }
  }

  private getNotificationActions(notification: SignalNotification): string[] {
    const actions: string[] = [];
    
    if (notification.type === 'signal') {
      actions.push('execute_signal', 'dismiss');
    } else if (notification.type === 'portfolio') {
      actions.push('view_portfolio');
    }
    
    return actions;
  }

  private addInAppNotification(notification: SignalNotification): void {
    this.notifications.unshift(notification);
    
    // Keep only last 100 notifications
    if (this.notifications.length > 100) {
      this.notifications = this.notifications.slice(0, 100);
    }
    
    // Save to local storage
    this.saveNotifications();
  }

  private navigateToSignal(signalId: string): void {
    // This would be implemented with navigation service
    console.log('Navigate to signal:', signalId);
  }

  private navigateToPortfolio(): void {
    // This would be implemented with navigation service
    console.log('Navigate to portfolio');
  }

  // Public API methods
  async updatePreferences(preferences: Partial<NotificationPreferences>): Promise<void> {
    this.preferences = { ...this.preferences, ...preferences };
    await this.savePreferences();
  }

  async getPreferences(): Promise<NotificationPreferences> {
    return this.preferences;
  }

  getNotifications(): SignalNotification[] {
    return this.notifications;
  }

  getUnreadCount(): number {
    return this.notifications.filter(n => !n.read).length;
  }

  async markAsRead(notificationId: string): Promise<void> {
    const notification = this.notifications.find(n => n.id === notificationId);
    if (notification) {
      notification.read = true;
      await this.saveNotifications();
    }
  }

  async markAllAsRead(): Promise<void> {
    this.notifications.forEach(n => n.read = true);
    await this.saveNotifications();
  }

  async clearNotifications(): Promise<void> {
    this.notifications = [];
    await this.saveNotifications();
  }

  async requestPermissions(): Promise<boolean> {
    try {
      if (Platform.OS === 'ios') {
        const permissions = await PushNotificationIOS.requestPermissions({
          alert: true,
          badge: true,
          sound: true,
        });
        return permissions.alert && permissions.badge && permissions.sound;
      } else {
        // Android permissions are handled automatically
        return true;
      }
    } catch (error) {
      console.error('Failed to request notification permissions:', error);
      return false;
    }
  }

  async testNotification(): Promise<void> {
    const testNotification: SignalNotification = {
      id: `test_${Date.now()}`,
      type: 'signal',
      priority: 'high',
      title: 'Test Notification',
      message: 'This is a test notification from AI-OTS',
      timestamp: Date.now(),
      read: false,
    };

    this.addInAppNotification(testNotification);
    
    if (this.preferences.channels.push) {
      this.sendPushNotification(testNotification);
    }
  }

  // Private storage methods
  private async loadPreferences(): Promise<void> {
    try {
      const stored = await AsyncStorage.getItem('notification_preferences');
      if (stored) {
        this.preferences = { ...this.preferences, ...JSON.parse(stored) };
      }
    } catch (error) {
      console.error('Failed to load notification preferences:', error);
    }
  }

  private async savePreferences(): Promise<void> {
    try {
      await AsyncStorage.setItem('notification_preferences', JSON.stringify(this.preferences));
    } catch (error) {
      console.error('Failed to save notification preferences:', error);
    }
  }

  private async loadNotifications(): Promise<void> {
    try {
      const stored = await AsyncStorage.getItem('notifications');
      if (stored) {
        this.notifications = JSON.parse(stored);
      }
    } catch (error) {
      console.error('Failed to load notifications:', error);
    }
  }

  private async saveNotifications(): Promise<void> {
    try {
      await AsyncStorage.setItem('notifications', JSON.stringify(this.notifications));
    } catch (error) {
      console.error('Failed to save notifications:', error);
    }
  }

  // Cleanup
  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }
}

// Singleton instance
export const notificationService = new NotificationService();

