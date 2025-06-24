import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-netinfo/netinfo';
import { Platform } from 'react-native';
import RNFS from 'react-native-fs';

export interface SyncStatus {
  isOnline: boolean;
  lastSyncTime: number;
  pendingOperations: number;
  syncInProgress: boolean;
  lastError?: string;
}

export interface OfflineOperation {
  id: string;
  type: 'CREATE' | 'UPDATE' | 'DELETE';
  endpoint: string;
  data: any;
  timestamp: number;
  retryCount: number;
  maxRetries: number;
}

export interface CachedData {
  key: string;
  data: any;
  timestamp: number;
  ttl: number; // Time to live in milliseconds
  version: string;
}

export interface SyncConfig {
  maxRetries: number;
  retryDelay: number;
  batchSize: number;
  syncInterval: number;
  maxCacheSize: number;
  defaultTTL: number;
}

export class OfflineStorageService {
  private static instance: OfflineStorageService;
  private isOnline = true;
  private syncInProgress = false;
  private pendingOperations: OfflineOperation[] = [];
  private syncConfig: SyncConfig;
  private syncTimer: NodeJS.Timeout | null = null;
  private listeners: ((status: SyncStatus) => void)[] = [];

  private constructor() {
    this.syncConfig = {
      maxRetries: 3,
      retryDelay: 5000, // 5 seconds
      batchSize: 10,
      syncInterval: 30000, // 30 seconds
      maxCacheSize: 50 * 1024 * 1024, // 50MB
      defaultTTL: 5 * 60 * 1000, // 5 minutes
    };
    
    this.initialize();
  }

  static getInstance(): OfflineStorageService {
    if (!OfflineStorageService.instance) {
      OfflineStorageService.instance = new OfflineStorageService();
    }
    return OfflineStorageService.instance;
  }

  private async initialize(): Promise<void> {
    try {
      // Load pending operations from storage
      await this.loadPendingOperations();
      
      // Set up network monitoring
      this.setupNetworkMonitoring();
      
      // Start sync timer
      this.startSyncTimer();
      
      console.log('Offline storage service initialized');
    } catch (error) {
      console.error('Failed to initialize offline storage service:', error);
    }
  }

  private setupNetworkMonitoring(): void {
    NetInfo.addEventListener(state => {
      const wasOnline = this.isOnline;
      this.isOnline = state.isConnected ?? false;
      
      console.log('Network status changed:', {
        isOnline: this.isOnline,
        type: state.type,
        isInternetReachable: state.isInternetReachable,
      });
      
      // If we just came online, trigger sync
      if (!wasOnline && this.isOnline) {
        this.syncPendingOperations();
      }
      
      this.notifyListeners();
    });
  }

  private startSyncTimer(): void {
    if (this.syncTimer) {
      clearInterval(this.syncTimer);
    }
    
    this.syncTimer = setInterval(() => {
      if (this.isOnline && !this.syncInProgress) {
        this.syncPendingOperations();
      }
    }, this.syncConfig.syncInterval);
  }

  // Cache Management
  async setCache(key: string, data: any, ttl?: number): Promise<void> {
    try {
      const cachedData: CachedData = {
        key,
        data,
        timestamp: Date.now(),
        ttl: ttl || this.syncConfig.defaultTTL,
        version: '1.0',
      };
      
      await AsyncStorage.setItem(`cache_${key}`, JSON.stringify(cachedData));
      
      // Clean up old cache entries periodically
      await this.cleanupCache();
    } catch (error) {
      console.error('Failed to set cache:', error);
    }
  }

  async getCache(key: string): Promise<any | null> {
    try {
      const cached = await AsyncStorage.getItem(`cache_${key}`);
      if (!cached) return null;
      
      const cachedData: CachedData = JSON.parse(cached);
      
      // Check if cache is expired
      if (Date.now() - cachedData.timestamp > cachedData.ttl) {
        await AsyncStorage.removeItem(`cache_${key}`);
        return null;
      }
      
      return cachedData.data;
    } catch (error) {
      console.error('Failed to get cache:', error);
      return null;
    }
  }

  async removeCache(key: string): Promise<void> {
    try {
      await AsyncStorage.removeItem(`cache_${key}`);
    } catch (error) {
      console.error('Failed to remove cache:', error);
    }
  }

  async clearCache(): Promise<void> {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const cacheKeys = keys.filter(key => key.startsWith('cache_'));
      await AsyncStorage.multiRemove(cacheKeys);
    } catch (error) {
      console.error('Failed to clear cache:', error);
    }
  }

  private async cleanupCache(): Promise<void> {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const cacheKeys = keys.filter(key => key.startsWith('cache_'));
      
      // Remove expired entries
      for (const key of cacheKeys) {
        const cached = await AsyncStorage.getItem(key);
        if (cached) {
          const cachedData: CachedData = JSON.parse(cached);
          if (Date.now() - cachedData.timestamp > cachedData.ttl) {
            await AsyncStorage.removeItem(key);
          }
        }
      }
      
      // Check cache size and remove oldest entries if needed
      await this.enforceMaxCacheSize();
    } catch (error) {
      console.error('Failed to cleanup cache:', error);
    }
  }

  private async enforceMaxCacheSize(): Promise<void> {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const cacheKeys = keys.filter(key => key.startsWith('cache_'));
      
      let totalSize = 0;
      const cacheEntries: { key: string; size: number; timestamp: number }[] = [];
      
      for (const key of cacheKeys) {
        const cached = await AsyncStorage.getItem(key);
        if (cached) {
          const size = new Blob([cached]).size;
          const cachedData: CachedData = JSON.parse(cached);
          totalSize += size;
          cacheEntries.push({
            key,
            size,
            timestamp: cachedData.timestamp,
          });
        }
      }
      
      if (totalSize > this.syncConfig.maxCacheSize) {
        // Sort by timestamp (oldest first)
        cacheEntries.sort((a, b) => a.timestamp - b.timestamp);
        
        // Remove oldest entries until under limit
        for (const entry of cacheEntries) {
          if (totalSize <= this.syncConfig.maxCacheSize) break;
          
          await AsyncStorage.removeItem(entry.key);
          totalSize -= entry.size;
        }
      }
    } catch (error) {
      console.error('Failed to enforce cache size limit:', error);
    }
  }

  // Offline Operations Management
  async addOfflineOperation(
    type: 'CREATE' | 'UPDATE' | 'DELETE',
    endpoint: string,
    data: any
  ): Promise<string> {
    const operation: OfflineOperation = {
      id: `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type,
      endpoint,
      data,
      timestamp: Date.now(),
      retryCount: 0,
      maxRetries: this.syncConfig.maxRetries,
    };
    
    this.pendingOperations.push(operation);
    await this.savePendingOperations();
    
    // Try to sync immediately if online
    if (this.isOnline) {
      this.syncPendingOperations();
    }
    
    this.notifyListeners();
    return operation.id;
  }

  async removeOfflineOperation(operationId: string): Promise<void> {
    this.pendingOperations = this.pendingOperations.filter(op => op.id !== operationId);
    await this.savePendingOperations();
    this.notifyListeners();
  }

  private async loadPendingOperations(): Promise<void> {
    try {
      const stored = await AsyncStorage.getItem('pending_operations');
      if (stored) {
        this.pendingOperations = JSON.parse(stored);
      }
    } catch (error) {
      console.error('Failed to load pending operations:', error);
      this.pendingOperations = [];
    }
  }

  private async savePendingOperations(): Promise<void> {
    try {
      await AsyncStorage.setItem('pending_operations', JSON.stringify(this.pendingOperations));
    } catch (error) {
      console.error('Failed to save pending operations:', error);
    }
  }

  // Data Synchronization
  async syncPendingOperations(): Promise<void> {
    if (!this.isOnline || this.syncInProgress || this.pendingOperations.length === 0) {
      return;
    }
    
    this.syncInProgress = true;
    this.notifyListeners();
    
    try {
      const batch = this.pendingOperations.slice(0, this.syncConfig.batchSize);
      const successfulOperations: string[] = [];
      
      for (const operation of batch) {
        try {
          await this.executeOperation(operation);
          successfulOperations.push(operation.id);
        } catch (error) {
          console.error('Failed to execute operation:', operation.id, error);
          
          // Increment retry count
          operation.retryCount++;
          
          // Remove operation if max retries exceeded
          if (operation.retryCount >= operation.maxRetries) {
            console.warn('Max retries exceeded for operation:', operation.id);
            successfulOperations.push(operation.id); // Remove from queue
          }
        }
      }
      
      // Remove successful operations
      this.pendingOperations = this.pendingOperations.filter(
        op => !successfulOperations.includes(op.id)
      );
      
      await this.savePendingOperations();
      await this.updateLastSyncTime();
      
    } catch (error) {
      console.error('Sync failed:', error);
    } finally {
      this.syncInProgress = false;
      this.notifyListeners();
    }
  }

  private async executeOperation(operation: OfflineOperation): Promise<void> {
    const API_BASE_URL = await AsyncStorage.getItem('API_BASE_URL') || 'http://localhost:8004';
    const authToken = await AsyncStorage.getItem('auth_token');
    
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    
    if (authToken) {
      headers['Authorization'] = `Bearer ${authToken}`;
    }
    
    let method: string;
    switch (operation.type) {
      case 'CREATE':
        method = 'POST';
        break;
      case 'UPDATE':
        method = 'PUT';
        break;
      case 'DELETE':
        method = 'DELETE';
        break;
      default:
        throw new Error(`Unknown operation type: ${operation.type}`);
    }
    
    const response = await fetch(`${API_BASE_URL}${operation.endpoint}`, {
      method,
      headers,
      body: operation.data ? JSON.stringify(operation.data) : undefined,
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    console.log('Operation executed successfully:', operation.id);
  }

  // File Storage for Large Data
  async storeFile(filename: string, data: string): Promise<string> {
    try {
      const documentsPath = Platform.OS === 'ios' 
        ? RNFS.DocumentDirectoryPath 
        : RNFS.ExternalDirectoryPath;
      
      const filePath = `${documentsPath}/ai-ots/${filename}`;
      
      // Ensure directory exists
      const dirPath = `${documentsPath}/ai-ots`;
      const dirExists = await RNFS.exists(dirPath);
      if (!dirExists) {
        await RNFS.mkdir(dirPath);
      }
      
      await RNFS.writeFile(filePath, data, 'utf8');
      return filePath;
    } catch (error) {
      console.error('Failed to store file:', error);
      throw error;
    }
  }

  async readFile(filename: string): Promise<string | null> {
    try {
      const documentsPath = Platform.OS === 'ios' 
        ? RNFS.DocumentDirectoryPath 
        : RNFS.ExternalDirectoryPath;
      
      const filePath = `${documentsPath}/ai-ots/${filename}`;
      const exists = await RNFS.exists(filePath);
      
      if (!exists) return null;
      
      return await RNFS.readFile(filePath, 'utf8');
    } catch (error) {
      console.error('Failed to read file:', error);
      return null;
    }
  }

  async deleteFile(filename: string): Promise<void> {
    try {
      const documentsPath = Platform.OS === 'ios' 
        ? RNFS.DocumentDirectoryPath 
        : RNFS.ExternalDirectoryPath;
      
      const filePath = `${documentsPath}/ai-ots/${filename}`;
      const exists = await RNFS.exists(filePath);
      
      if (exists) {
        await RNFS.unlink(filePath);
      }
    } catch (error) {
      console.error('Failed to delete file:', error);
    }
  }

  // Status and Monitoring
  async getSyncStatus(): Promise<SyncStatus> {
    const lastSyncTime = await this.getLastSyncTime();
    
    return {
      isOnline: this.isOnline,
      lastSyncTime,
      pendingOperations: this.pendingOperations.length,
      syncInProgress: this.syncInProgress,
    };
  }

  private async getLastSyncTime(): Promise<number> {
    try {
      const stored = await AsyncStorage.getItem('last_sync_time');
      return stored ? parseInt(stored, 10) : 0;
    } catch (error) {
      console.error('Failed to get last sync time:', error);
      return 0;
    }
  }

  private async updateLastSyncTime(): Promise<void> {
    try {
      await AsyncStorage.setItem('last_sync_time', Date.now().toString());
    } catch (error) {
      console.error('Failed to update last sync time:', error);
    }
  }

  // Event Listeners
  addSyncStatusListener(listener: (status: SyncStatus) => void): void {
    this.listeners.push(listener);
  }

  removeSyncStatusListener(listener: (status: SyncStatus) => void): void {
    this.listeners = this.listeners.filter(l => l !== listener);
  }

  private async notifyListeners(): Promise<void> {
    const status = await this.getSyncStatus();
    this.listeners.forEach(listener => {
      try {
        listener(status);
      } catch (error) {
        console.error('Error in sync status listener:', error);
      }
    });
  }

  // Utility Methods
  async getStorageInfo(): Promise<{
    cacheSize: number;
    pendingOperations: number;
    lastSyncTime: number;
    isOnline: boolean;
  }> {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const cacheKeys = keys.filter(key => key.startsWith('cache_'));
      
      let cacheSize = 0;
      for (const key of cacheKeys) {
        const cached = await AsyncStorage.getItem(key);
        if (cached) {
          cacheSize += new Blob([cached]).size;
        }
      }
      
      const lastSyncTime = await this.getLastSyncTime();
      
      return {
        cacheSize,
        pendingOperations: this.pendingOperations.length,
        lastSyncTime,
        isOnline: this.isOnline,
      };
    } catch (error) {
      console.error('Failed to get storage info:', error);
      return {
        cacheSize: 0,
        pendingOperations: 0,
        lastSyncTime: 0,
        isOnline: false,
      };
    }
  }

  async forcSync(): Promise<void> {
    if (this.isOnline) {
      await this.syncPendingOperations();
    }
  }

  // Cleanup
  cleanup(): void {
    if (this.syncTimer) {
      clearInterval(this.syncTimer);
      this.syncTimer = null;
    }
    this.listeners = [];
  }
}

// Singleton instance
export const offlineStorageService = OfflineStorageService.getInstance();

