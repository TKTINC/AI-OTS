/**
 * App Navigator
 * Main navigation structure for the mobile application
 */

import React from 'react';
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createDrawerNavigator } from '@react-navigation/drawer';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { useSelector } from 'react-redux';

import { RootState } from '../store/store';
import { AuthNavigator } from './AuthNavigator';
import { TradingNavigator } from './TradingNavigator';
import { PortfolioNavigator } from './PortfolioNavigator';
import { AnalyticsNavigator } from './AnalyticsNavigator';
import { SettingsNavigator } from './SettingsNavigator';

// Navigation types
export type RootStackParamList = {
  Auth: undefined;
  Main: undefined;
  Modal: { screen: string; params?: any };
};

export type MainTabParamList = {
  Trading: undefined;
  Portfolio: undefined;
  Analytics: undefined;
  Settings: undefined;
};

const RootStack = createStackNavigator<RootStackParamList>();
const MainTab = createBottomTabNavigator<MainTabParamList>();
const Drawer = createDrawerNavigator();

// Main Tab Navigator
const MainTabNavigator: React.FC = () => {
  return (
    <MainTab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: string;

          switch (route.name) {
            case 'Trading':
              iconName = focused ? 'trending-up' : 'trending-up';
              break;
            case 'Portfolio':
              iconName = focused ? 'account-balance-wallet' : 'account-balance-wallet';
              break;
            case 'Analytics':
              iconName = focused ? 'analytics' : 'analytics';
              break;
            case 'Settings':
              iconName = focused ? 'settings' : 'settings';
              break;
            default:
              iconName = 'help';
          }

          return <Icon name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: '#007AFF',
        tabBarInactiveTintColor: '#8E8E93',
        tabBarStyle: {
          backgroundColor: '#FFFFFF',
          borderTopWidth: 1,
          borderTopColor: '#E5E5EA',
          paddingBottom: 5,
          paddingTop: 5,
          height: 60,
        },
        headerShown: false,
      })}
    >
      <MainTab.Screen 
        name="Trading" 
        component={TradingNavigator}
        options={{ tabBarLabel: 'Trading' }}
      />
      <MainTab.Screen 
        name="Portfolio" 
        component={PortfolioNavigator}
        options={{ tabBarLabel: 'Portfolio' }}
      />
      <MainTab.Screen 
        name="Analytics" 
        component={AnalyticsNavigator}
        options={{ tabBarLabel: 'Analytics' }}
      />
      <MainTab.Screen 
        name="Settings" 
        component={SettingsNavigator}
        options={{ tabBarLabel: 'Settings' }}
      />
    </MainTab.Navigator>
  );
};

// Drawer Navigator (for additional navigation)
const DrawerNavigator: React.FC = () => {
  return (
    <Drawer.Navigator
      screenOptions={{
        drawerStyle: {
          backgroundColor: '#FFFFFF',
          width: 280,
        },
        drawerActiveTintColor: '#007AFF',
        drawerInactiveTintColor: '#8E8E93',
        headerShown: false,
      }}
    >
      <Drawer.Screen 
        name="MainTabs" 
        component={MainTabNavigator}
        options={{
          drawerLabel: 'Trading Dashboard',
          drawerIcon: ({ color, size }) => (
            <Icon name="dashboard" size={size} color={color} />
          ),
        }}
      />
    </Drawer.Navigator>
  );
};

// Main App Navigator
export const AppNavigator: React.FC = () => {
  const isAuthenticated = useSelector((state: RootState) => state.auth.isAuthenticated);

  return (
    <RootStack.Navigator
      screenOptions={{
        headerShown: false,
        gestureEnabled: true,
        cardStyleInterpolator: ({ current, layouts }) => {
          return {
            cardStyle: {
              transform: [
                {
                  translateX: current.progress.interpolate({
                    inputRange: [0, 1],
                    outputRange: [layouts.screen.width, 0],
                  }),
                },
              ],
            },
          };
        },
      }}
    >
      {!isAuthenticated ? (
        <RootStack.Screen name="Auth" component={AuthNavigator} />
      ) : (
        <RootStack.Screen name="Main" component={DrawerNavigator} />
      )}
    </RootStack.Navigator>
  );
};

