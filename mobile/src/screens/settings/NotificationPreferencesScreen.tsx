import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Switch,
  TouchableOpacity,
  Alert,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { notificationService, NotificationPreferences } from '../services/notifications/NotificationService';

interface SettingsSectionProps {
  title: string;
  children: React.ReactNode;
}

const SettingsSection: React.FC<SettingsSectionProps> = ({ title, children }) => (
  <View style={styles.section}>
    <Text style={styles.sectionTitle}>{title}</Text>
    <View style={styles.sectionContent}>
      {children}
    </View>
  </View>
);

interface SettingsRowProps {
  title: string;
  subtitle?: string;
  value?: boolean;
  onValueChange?: (value: boolean) => void;
  onPress?: () => void;
  showArrow?: boolean;
  disabled?: boolean;
}

const SettingsRow: React.FC<SettingsRowProps> = ({
  title,
  subtitle,
  value,
  onValueChange,
  onPress,
  showArrow = false,
  disabled = false,
}) => (
  <TouchableOpacity
    style={[styles.settingsRow, disabled && styles.disabledRow]}
    onPress={onPress}
    disabled={disabled || (!onPress && !onValueChange)}
    activeOpacity={0.7}
  >
    <View style={styles.settingsRowContent}>
      <View style={styles.settingsRowText}>
        <Text style={[styles.settingsRowTitle, disabled && styles.disabledText]}>
          {title}
        </Text>
        {subtitle && (
          <Text style={[styles.settingsRowSubtitle, disabled && styles.disabledText]}>
            {subtitle}
          </Text>
        )}
      </View>
      <View style={styles.settingsRowControl}>
        {onValueChange && (
          <Switch
            value={value}
            onValueChange={onValueChange}
            disabled={disabled}
            trackColor={{ false: '#E5E5E5', true: '#007AFF' }}
            thumbColor={Platform.OS === 'android' ? '#FFFFFF' : undefined}
          />
        )}
        {showArrow && (
          <Icon name="chevron-right" size={24} color="#C7C7CC" />
        )}
      </View>
    </View>
  </TouchableOpacity>
);

interface TimePickerProps {
  title: string;
  time: string;
  onTimeChange: (time: string) => void;
}

const TimePicker: React.FC<TimePickerProps> = ({ title, time, onTimeChange }) => {
  const showTimePicker = () => {
    // This would show a time picker modal
    // For now, we'll show an alert with preset options
    Alert.alert(
      title,
      'Select time',
      [
        { text: '20:00', onPress: () => onTimeChange('20:00') },
        { text: '21:00', onPress: () => onTimeChange('21:00') },
        { text: '22:00', onPress: () => onTimeChange('22:00') },
        { text: '23:00', onPress: () => onTimeChange('23:00') },
        { text: 'Cancel', style: 'cancel' },
      ]
    );
  };

  return (
    <SettingsRow
      title={title}
      subtitle={time}
      onPress={showTimePicker}
      showArrow
    />
  );
};

export const NotificationPreferencesScreen: React.FC = () => {
  const navigation = useNavigation();
  const [preferences, setPreferences] = useState<NotificationPreferences | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadPreferences();
  }, []);

  const loadPreferences = async () => {
    try {
      const prefs = await notificationService.getPreferences();
      setPreferences(prefs);
    } catch (error) {
      console.error('Failed to load preferences:', error);
      Alert.alert('Error', 'Failed to load notification preferences');
    } finally {
      setLoading(false);
    }
  };

  const updatePreferences = async (updates: Partial<NotificationPreferences>) => {
    if (!preferences) return;

    try {
      const newPreferences = { ...preferences, ...updates };
      await notificationService.updatePreferences(updates);
      setPreferences(newPreferences);
    } catch (error) {
      console.error('Failed to update preferences:', error);
      Alert.alert('Error', 'Failed to update notification preferences');
    }
  };

  const requestPermissions = async () => {
    try {
      const granted = await notificationService.requestPermissions();
      if (granted) {
        Alert.alert('Success', 'Notification permissions granted');
        await updatePreferences({ enabled: true });
      } else {
        Alert.alert(
          'Permissions Required',
          'Please enable notifications in your device settings to receive trading alerts.',
          [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Open Settings', onPress: () => {
              // This would open device settings
              console.log('Open device settings');
            }},
          ]
        );
      }
    } catch (error) {
      console.error('Failed to request permissions:', error);
      Alert.alert('Error', 'Failed to request notification permissions');
    }
  };

  const testNotification = async () => {
    try {
      await notificationService.testNotification();
      Alert.alert('Test Sent', 'A test notification has been sent');
    } catch (error) {
      console.error('Failed to send test notification:', error);
      Alert.alert('Error', 'Failed to send test notification');
    }
  };

  if (loading || !preferences) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <Text style={styles.loadingText}>Loading preferences...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Icon name="arrow-back" size={24} color="#007AFF" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Notification Settings</Text>
        <TouchableOpacity
          style={styles.testButton}
          onPress={testNotification}
        >
          <Text style={styles.testButtonText}>Test</Text>
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        {/* Master Toggle */}
        <SettingsSection title="General">
          <SettingsRow
            title="Enable Notifications"
            subtitle="Master switch for all notifications"
            value={preferences.enabled}
            onValueChange={(value) => {
              if (value) {
                requestPermissions();
              } else {
                updatePreferences({ enabled: value });
              }
            }}
          />
        </SettingsSection>

        {/* Notification Types */}
        <SettingsSection title="Notification Types">
          <SettingsRow
            title="Trading Signals"
            subtitle="New trading opportunities and signals"
            value={preferences.signalAlerts}
            onValueChange={(value) => updatePreferences({ signalAlerts: value })}
            disabled={!preferences.enabled}
          />
          <SettingsRow
            title="Risk Alerts"
            subtitle="Portfolio risk and compliance warnings"
            value={preferences.riskAlerts}
            onValueChange={(value) => updatePreferences({ riskAlerts: value })}
            disabled={!preferences.enabled}
          />
          <SettingsRow
            title="Portfolio Updates"
            subtitle="Position changes and P&L updates"
            value={preferences.portfolioUpdates}
            onValueChange={(value) => updatePreferences({ portfolioUpdates: value })}
            disabled={!preferences.enabled}
          />
          <SettingsRow
            title="Market News"
            subtitle="Important market news and events"
            value={preferences.marketNews}
            onValueChange={(value) => updatePreferences({ marketNews: value })}
            disabled={!preferences.enabled}
          />
        </SettingsSection>

        {/* Priority Levels */}
        <SettingsSection title="Priority Levels">
          <SettingsRow
            title="Critical Alerts"
            subtitle="Emergency situations requiring immediate action"
            value={preferences.priorities.critical}
            onValueChange={(value) => 
              updatePreferences({ 
                priorities: { ...preferences.priorities, critical: value }
              })
            }
            disabled={!preferences.enabled}
          />
          <SettingsRow
            title="High Priority"
            subtitle="Important alerts requiring attention"
            value={preferences.priorities.high}
            onValueChange={(value) => 
              updatePreferences({ 
                priorities: { ...preferences.priorities, high: value }
              })
            }
            disabled={!preferences.enabled}
          />
          <SettingsRow
            title="Medium Priority"
            subtitle="Standard trading signals and updates"
            value={preferences.priorities.medium}
            onValueChange={(value) => 
              updatePreferences({ 
                priorities: { ...preferences.priorities, medium: value }
              })
            }
            disabled={!preferences.enabled}
          />
          <SettingsRow
            title="Low Priority"
            subtitle="Informational updates and minor alerts"
            value={preferences.priorities.low}
            onValueChange={(value) => 
              updatePreferences({ 
                priorities: { ...preferences.priorities, low: value }
              })
            }
            disabled={!preferences.enabled}
          />
        </SettingsSection>

        {/* Delivery Channels */}
        <SettingsSection title="Delivery Options">
          <SettingsRow
            title="Push Notifications"
            subtitle="System notifications on lock screen"
            value={preferences.channels.push}
            onValueChange={(value) => 
              updatePreferences({ 
                channels: { ...preferences.channels, push: value }
              })
            }
            disabled={!preferences.enabled}
          />
          <SettingsRow
            title="In-App Notifications"
            subtitle="Notifications within the app"
            value={preferences.channels.inApp}
            onValueChange={(value) => 
              updatePreferences({ 
                channels: { ...preferences.channels, inApp: value }
              })
            }
            disabled={!preferences.enabled}
          />
          <SettingsRow
            title="Sound"
            subtitle="Play notification sounds"
            value={preferences.channels.sound}
            onValueChange={(value) => 
              updatePreferences({ 
                channels: { ...preferences.channels, sound: value }
              })
            }
            disabled={!preferences.enabled}
          />
          <SettingsRow
            title="Vibration"
            subtitle="Vibrate for notifications"
            value={preferences.channels.vibration}
            onValueChange={(value) => 
              updatePreferences({ 
                channels: { ...preferences.channels, vibration: value }
              })
            }
            disabled={!preferences.enabled}
          />
        </SettingsSection>

        {/* Quiet Hours */}
        <SettingsSection title="Quiet Hours">
          <SettingsRow
            title="Enable Quiet Hours"
            subtitle="Reduce notifications during specified hours"
            value={preferences.quietHours.enabled}
            onValueChange={(value) => 
              updatePreferences({ 
                quietHours: { ...preferences.quietHours, enabled: value }
              })
            }
            disabled={!preferences.enabled}
          />
          {preferences.quietHours.enabled && (
            <>
              <TimePicker
                title="Start Time"
                time={preferences.quietHours.start}
                onTimeChange={(time) => 
                  updatePreferences({ 
                    quietHours: { ...preferences.quietHours, start: time }
                  })
                }
              />
              <TimePicker
                title="End Time"
                time={preferences.quietHours.end}
                onTimeChange={(time) => 
                  updatePreferences({ 
                    quietHours: { ...preferences.quietHours, end: time }
                  })
                }
              />
              <View style={styles.quietHoursNote}>
                <Text style={styles.noteText}>
                  Note: Critical alerts will still be delivered during quiet hours
                </Text>
              </View>
            </>
          )}
        </SettingsSection>

        {/* Additional Options */}
        <SettingsSection title="Advanced">
          <SettingsRow
            title="Notification History"
            subtitle="View and manage notification history"
            onPress={() => {
              // Navigate to notification history screen
              console.log('Navigate to notification history');
            }}
            showArrow
          />
          <SettingsRow
            title="Reset to Defaults"
            subtitle="Reset all notification settings"
            onPress={() => {
              Alert.alert(
                'Reset Settings',
                'Are you sure you want to reset all notification settings to defaults?',
                [
                  { text: 'Cancel', style: 'cancel' },
                  { 
                    text: 'Reset', 
                    style: 'destructive',
                    onPress: async () => {
                      try {
                        const defaultPrefs = {
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
                        await notificationService.updatePreferences(defaultPrefs);
                        setPreferences({ ...preferences, ...defaultPrefs });
                        Alert.alert('Success', 'Settings reset to defaults');
                      } catch (error) {
                        Alert.alert('Error', 'Failed to reset settings');
                      }
                    }
                  },
                ]
              );
            }}
          />
        </SettingsSection>

        <View style={styles.bottomPadding} />
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F2F2F7',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    fontSize: 16,
    color: '#8E8E93',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#FFFFFF',
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#C6C6C8',
  },
  backButton: {
    padding: 8,
    marginLeft: -8,
  },
  headerTitle: {
    fontSize: 17,
    fontWeight: '600',
    color: '#000000',
  },
  testButton: {
    padding: 8,
    marginRight: -8,
  },
  testButtonText: {
    fontSize: 17,
    color: '#007AFF',
    fontWeight: '400',
  },
  scrollView: {
    flex: 1,
  },
  section: {
    marginTop: 32,
  },
  sectionTitle: {
    fontSize: 13,
    fontWeight: '400',
    color: '#6D6D72',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginHorizontal: 16,
    marginBottom: 8,
  },
  sectionContent: {
    backgroundColor: '#FFFFFF',
    borderTopWidth: StyleSheet.hairlineWidth,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderColor: '#C6C6C8',
  },
  settingsRow: {
    backgroundColor: '#FFFFFF',
  },
  settingsRowContent: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    minHeight: 44,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#C6C6C8',
  },
  settingsRowText: {
    flex: 1,
    marginRight: 12,
  },
  settingsRowTitle: {
    fontSize: 17,
    fontWeight: '400',
    color: '#000000',
    marginBottom: 2,
  },
  settingsRowSubtitle: {
    fontSize: 13,
    fontWeight: '400',
    color: '#8E8E93',
    lineHeight: 18,
  },
  settingsRowControl: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  disabledRow: {
    opacity: 0.5,
  },
  disabledText: {
    color: '#8E8E93',
  },
  quietHoursNote: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#F2F2F7',
  },
  noteText: {
    fontSize: 13,
    color: '#8E8E93',
    fontStyle: 'italic',
  },
  bottomPadding: {
    height: 32,
  },
});

export default NotificationPreferencesScreen;

