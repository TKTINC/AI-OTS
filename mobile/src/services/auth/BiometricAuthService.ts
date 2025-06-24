import { Platform } from 'react-native';
import TouchID from 'react-native-touch-id';
import FingerprintScanner from 'react-native-fingerprint-scanner';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Keychain from 'react-native-keychain';

export enum BiometricType {
  NONE = 'none',
  TOUCH_ID = 'TouchID',
  FACE_ID = 'FaceID',
  FINGERPRINT = 'Fingerprint',
  FACE = 'Face',
  IRIS = 'Iris',
}

export interface BiometricAuthConfig {
  title: string;
  subtitle?: string;
  description?: string;
  fallbackLabel?: string;
  cancelLabel?: string;
  disableDeviceFallback?: boolean;
  passcodeFallback?: boolean;
}

export interface AuthenticationResult {
  success: boolean;
  error?: string;
  biometricType?: BiometricType;
  fallback?: boolean;
}

export interface BiometricCapabilities {
  isAvailable: boolean;
  biometricType: BiometricType;
  isEnrolled: boolean;
  error?: string;
}

export class BiometricAuthService {
  private static instance: BiometricAuthService;
  private isInitialized = false;
  private capabilities: BiometricCapabilities | null = null;

  private constructor() {}

  static getInstance(): BiometricAuthService {
    if (!BiometricAuthService.instance) {
      BiometricAuthService.instance = new BiometricAuthService();
    }
    return BiometricAuthService.instance;
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      this.capabilities = await this.checkBiometricCapabilities();
      this.isInitialized = true;
      console.log('Biometric auth service initialized:', this.capabilities);
    } catch (error) {
      console.error('Failed to initialize biometric auth service:', error);
      this.capabilities = {
        isAvailable: false,
        biometricType: BiometricType.NONE,
        isEnrolled: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  async checkBiometricCapabilities(): Promise<BiometricCapabilities> {
    try {
      if (Platform.OS === 'ios') {
        return await this.checkIOSCapabilities();
      } else {
        return await this.checkAndroidCapabilities();
      }
    } catch (error) {
      console.error('Error checking biometric capabilities:', error);
      return {
        isAvailable: false,
        biometricType: BiometricType.NONE,
        isEnrolled: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  private async checkIOSCapabilities(): Promise<BiometricCapabilities> {
    try {
      const biometryType = await TouchID.isSupported();
      
      if (biometryType) {
        let biometricType: BiometricType;
        
        switch (biometryType) {
          case 'FaceID':
            biometricType = BiometricType.FACE_ID;
            break;
          case 'TouchID':
            biometricType = BiometricType.TOUCH_ID;
            break;
          default:
            biometricType = BiometricType.TOUCH_ID;
        }

        return {
          isAvailable: true,
          biometricType,
          isEnrolled: true, // TouchID.isSupported() only returns true if enrolled
        };
      } else {
        return {
          isAvailable: false,
          biometricType: BiometricType.NONE,
          isEnrolled: false,
        };
      }
    } catch (error) {
      console.error('iOS biometric check error:', error);
      return {
        isAvailable: false,
        biometricType: BiometricType.NONE,
        isEnrolled: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  private async checkAndroidCapabilities(): Promise<BiometricCapabilities> {
    try {
      const isAvailable = await FingerprintScanner.isSensorAvailable();
      
      return {
        isAvailable: true,
        biometricType: BiometricType.FINGERPRINT,
        isEnrolled: true, // isSensorAvailable() checks enrollment
      };
    } catch (error: any) {
      console.error('Android biometric check error:', error);
      
      let biometricType = BiometricType.NONE;
      let isAvailable = false;
      
      // Parse Android-specific errors
      if (error.name === 'FingerprintScannerNotAvailable') {
        isAvailable = false;
      } else if (error.name === 'FingerprintScannerNotEnrolled') {
        isAvailable = true;
        biometricType = BiometricType.FINGERPRINT;
      }

      return {
        isAvailable,
        biometricType,
        isEnrolled: false,
        error: error.message || 'Unknown error',
      };
    }
  }

  async authenticate(config?: BiometricAuthConfig): Promise<AuthenticationResult> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    if (!this.capabilities?.isAvailable) {
      return {
        success: false,
        error: 'Biometric authentication not available',
      };
    }

    if (!this.capabilities.isEnrolled) {
      return {
        success: false,
        error: 'No biometric credentials enrolled',
      };
    }

    try {
      if (Platform.OS === 'ios') {
        return await this.authenticateIOS(config);
      } else {
        return await this.authenticateAndroid(config);
      }
    } catch (error) {
      console.error('Biometric authentication error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Authentication failed',
      };
    }
  }

  private async authenticateIOS(config?: BiometricAuthConfig): Promise<AuthenticationResult> {
    const defaultConfig: BiometricAuthConfig = {
      title: 'Authenticate',
      subtitle: 'Use your biometric to access AI-OTS',
      description: 'Place your finger on the Touch ID sensor or look at the camera for Face ID',
      fallbackLabel: 'Use Passcode',
      cancelLabel: 'Cancel',
      disableDeviceFallback: false,
      passcodeFallback: true,
    };

    const authConfig = { ...defaultConfig, ...config };

    try {
      const result = await TouchID.authenticate(authConfig.description || '', {
        title: authConfig.title,
        fallbackLabel: authConfig.fallbackLabel,
        cancelLabel: authConfig.cancelLabel,
        disableDeviceFallback: authConfig.disableDeviceFallback,
        passcodeFallback: authConfig.passcodeFallback,
      });

      return {
        success: true,
        biometricType: this.capabilities?.biometricType,
      };
    } catch (error: any) {
      console.error('iOS authentication error:', error);
      
      // Handle specific iOS errors
      if (error.name === 'LAErrorUserCancel') {
        return {
          success: false,
          error: 'User cancelled authentication',
        };
      } else if (error.name === 'LAErrorUserFallback') {
        return {
          success: false,
          error: 'User chose fallback authentication',
          fallback: true,
        };
      } else if (error.name === 'LAErrorSystemCancel') {
        return {
          success: false,
          error: 'System cancelled authentication',
        };
      } else if (error.name === 'LAErrorPasscodeNotSet') {
        return {
          success: false,
          error: 'Passcode not set on device',
        };
      } else if (error.name === 'LAErrorBiometryNotAvailable') {
        return {
          success: false,
          error: 'Biometry not available',
        };
      } else if (error.name === 'LAErrorBiometryNotEnrolled') {
        return {
          success: false,
          error: 'No biometric credentials enrolled',
        };
      } else if (error.name === 'LAErrorBiometryLockout') {
        return {
          success: false,
          error: 'Biometry locked out due to too many failed attempts',
        };
      }

      return {
        success: false,
        error: error.message || 'Authentication failed',
      };
    }
  }

  private async authenticateAndroid(config?: BiometricAuthConfig): Promise<AuthenticationResult> {
    const defaultConfig: BiometricAuthConfig = {
      title: 'Biometric Authentication',
      subtitle: 'Use your fingerprint to access AI-OTS',
      description: 'Place your finger on the fingerprint sensor',
      cancelLabel: 'Cancel',
    };

    const authConfig = { ...defaultConfig, ...config };

    try {
      const result = await FingerprintScanner.authenticate({
        title: authConfig.title,
        subTitle: authConfig.subtitle,
        description: authConfig.description,
        cancelButton: authConfig.cancelLabel,
      });

      return {
        success: true,
        biometricType: BiometricType.FINGERPRINT,
      };
    } catch (error: any) {
      console.error('Android authentication error:', error);
      
      // Handle specific Android errors
      if (error.name === 'UserCancel') {
        return {
          success: false,
          error: 'User cancelled authentication',
        };
      } else if (error.name === 'UserFallback') {
        return {
          success: false,
          error: 'User chose fallback authentication',
          fallback: true,
        };
      } else if (error.name === 'SystemCancel') {
        return {
          success: false,
          error: 'System cancelled authentication',
        };
      } else if (error.name === 'PasscodeNotSet') {
        return {
          success: false,
          error: 'Passcode not set on device',
        };
      } else if (error.name === 'FingerprintScannerNotAvailable') {
        return {
          success: false,
          error: 'Fingerprint scanner not available',
        };
      } else if (error.name === 'FingerprintScannerNotEnrolled') {
        return {
          success: false,
          error: 'No fingerprints enrolled',
        };
      } else if (error.name === 'DeviceLocked') {
        return {
          success: false,
          error: 'Device locked due to too many failed attempts',
        };
      }

      return {
        success: false,
        error: error.message || 'Authentication failed',
      };
    } finally {
      // Clean up Android fingerprint scanner
      if (Platform.OS === 'android') {
        FingerprintScanner.release();
      }
    }
  }

  async storeCredentials(username: string, password: string): Promise<boolean> {
    try {
      const biometricType = this.capabilities?.biometricType || BiometricType.NONE;
      
      const options: Keychain.Options = {
        service: 'ai-ots-app',
        accessControl: Keychain.ACCESS_CONTROL.BIOMETRY_ANY,
        authenticationType: Keychain.AUTHENTICATION_TYPE.DEVICE_PASSCODE_OR_BIOMETRICS,
        accessGroup: undefined, // iOS only
        showModal: true,
        kLocalizedFallbackTitle: 'Use Passcode',
      };

      // Adjust options based on platform and biometric type
      if (Platform.OS === 'ios') {
        if (biometricType === BiometricType.FACE_ID) {
          options.accessControl = Keychain.ACCESS_CONTROL.BIOMETRY_ANY;
        } else if (biometricType === BiometricType.TOUCH_ID) {
          options.accessControl = Keychain.ACCESS_CONTROL.BIOMETRY_ANY;
        }
      }

      await Keychain.setInternetCredentials(
        'ai-ots-credentials',
        username,
        password,
        options
      );

      // Store flag indicating biometric auth is enabled
      await AsyncStorage.setItem('biometric_auth_enabled', 'true');
      
      return true;
    } catch (error) {
      console.error('Failed to store credentials:', error);
      return false;
    }
  }

  async getStoredCredentials(): Promise<{ username: string; password: string } | null> {
    try {
      const credentials = await Keychain.getInternetCredentials('ai-ots-credentials');
      
      if (credentials && credentials.username && credentials.password) {
        return {
          username: credentials.username,
          password: credentials.password,
        };
      }
      
      return null;
    } catch (error) {
      console.error('Failed to get stored credentials:', error);
      return null;
    }
  }

  async removeStoredCredentials(): Promise<boolean> {
    try {
      await Keychain.resetInternetCredentials('ai-ots-credentials');
      await AsyncStorage.removeItem('biometric_auth_enabled');
      return true;
    } catch (error) {
      console.error('Failed to remove stored credentials:', error);
      return false;
    }
  }

  async isBiometricAuthEnabled(): Promise<boolean> {
    try {
      const enabled = await AsyncStorage.getItem('biometric_auth_enabled');
      return enabled === 'true';
    } catch (error) {
      console.error('Failed to check biometric auth status:', error);
      return false;
    }
  }

  async setBiometricAuthEnabled(enabled: boolean): Promise<void> {
    try {
      if (enabled) {
        await AsyncStorage.setItem('biometric_auth_enabled', 'true');
      } else {
        await AsyncStorage.removeItem('biometric_auth_enabled');
        await this.removeStoredCredentials();
      }
    } catch (error) {
      console.error('Failed to set biometric auth status:', error);
    }
  }

  getCapabilities(): BiometricCapabilities | null {
    return this.capabilities;
  }

  getBiometricTypeDisplayName(): string {
    if (!this.capabilities) return 'Biometric';
    
    switch (this.capabilities.biometricType) {
      case BiometricType.FACE_ID:
        return 'Face ID';
      case BiometricType.TOUCH_ID:
        return 'Touch ID';
      case BiometricType.FINGERPRINT:
        return 'Fingerprint';
      case BiometricType.FACE:
        return 'Face Recognition';
      case BiometricType.IRIS:
        return 'Iris Recognition';
      default:
        return 'Biometric';
    }
  }

  async authenticateForLogin(): Promise<AuthenticationResult> {
    const biometricName = this.getBiometricTypeDisplayName();
    
    return await this.authenticate({
      title: 'Login to AI-OTS',
      subtitle: `Use ${biometricName} to access your account`,
      description: `Authenticate with ${biometricName} to securely access your trading account`,
      fallbackLabel: 'Use Password',
      cancelLabel: 'Cancel',
      passcodeFallback: true,
    });
  }

  async authenticateForTransaction(): Promise<AuthenticationResult> {
    const biometricName = this.getBiometricTypeDisplayName();
    
    return await this.authenticate({
      title: 'Confirm Transaction',
      subtitle: `Use ${biometricName} to confirm this trade`,
      description: `Authenticate with ${biometricName} to execute this trading order`,
      fallbackLabel: 'Use Password',
      cancelLabel: 'Cancel',
      passcodeFallback: true,
    });
  }

  async authenticateForSettings(): Promise<AuthenticationResult> {
    const biometricName = this.getBiometricTypeDisplayName();
    
    return await this.authenticate({
      title: 'Access Settings',
      subtitle: `Use ${biometricName} to access sensitive settings`,
      description: `Authenticate with ${biometricName} to modify account settings`,
      fallbackLabel: 'Use Password',
      cancelLabel: 'Cancel',
      passcodeFallback: true,
    });
  }

  // Cleanup method
  cleanup(): void {
    if (Platform.OS === 'android') {
      try {
        FingerprintScanner.release();
      } catch (error) {
        console.error('Error releasing fingerprint scanner:', error);
      }
    }
  }
}

// Singleton instance
export const biometricAuthService = BiometricAuthService.getInstance();

