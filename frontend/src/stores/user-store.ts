import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { User, UserPreferences, UserRole } from '../types';

interface UserState {
  // State
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;

  // Actions
  setUser: (user: User) => void;
  updatePreferences: (preferences: Partial<UserPreferences>) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  logout: () => void;
  reset: () => void;
}

const initialState = {
  user: null,
  isAuthenticated: false,
  isLoading: false,
  error: null,
};

export const useUserStore = create<UserState>()(
  devtools(
    persist(
      (set, get) => ({
        ...initialState,

        setUser: (user: User) =>
          set(
            { 
              user, 
              isAuthenticated: true, 
              error: null 
            },
            false,
            'setUser'
          ),

        updatePreferences: (preferences: Partial<UserPreferences>) =>
          set(
            (state) => ({
              user: state.user
                ? {
                    ...state.user,
                    preferences: {
                      ...state.user.preferences,
                      ...preferences,
                    },
                  }
                : null,
            }),
            false,
            'updatePreferences'
          ),

        setLoading: (isLoading: boolean) =>
          set({ isLoading }, false, 'setLoading'),

        setError: (error: string | null) =>
          set({ error }, false, 'setError'),

        logout: () =>
          set(
            {
              user: null,
              isAuthenticated: false,
              error: null,
            },
            false,
            'logout'
          ),

        reset: () => set(initialState, false, 'reset'),
      }),
      {
        name: 'derslens-user-store',
        partialize: (state) => ({
          user: state.user,
          isAuthenticated: state.isAuthenticated,
        }),
      }
    ),
    {
      name: 'user-store',
    }
  )
);

// Selectors
export const useAuth = () => {
  const { user, isAuthenticated, isLoading, error } = useUserStore();
  return { user, isAuthenticated, isLoading, error };
};

export const useUserRole = (): UserRole | null => {
  const user = useUserStore((state) => state.user);
  return user?.role || null;
};

export const useUserPreferences = () => {
  const user = useUserStore((state) => state.user);
  return user?.preferences || null;
};
