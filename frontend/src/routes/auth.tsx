import { createFileRoute, useSearch, useNavigate, redirect } from '@tanstack/react-router'
import { useState } from 'react'
import { z } from 'zod'
import { useMutation } from '@tanstack/react-query'
import axios from 'axios'

// Validation schemas
const emailSchema = z.string()
  .min(1, 'Email is required')
  .email('Invalid email address')

const passwordSchema = z.string()
  .min(8, 'Password must be at least 8 characters')
  .regex(/[A-Z]/, 'Password must contain at least one uppercase letter')
  .regex(/[a-z]/, 'Password must contain at least one lowercase letter')
  .regex(/[0-9]/, 'Password must contain at least one number')
  .regex(/[^A-Za-z0-9]/, 'Password must contain at least one special character')

// Authentication API functions
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

interface AuthCredentials {
  email: string;
  password: string;
}

interface RegisterCredentials extends AuthCredentials {
  confirmPassword?: string;
}

const loginUser = async (credentials: AuthCredentials) => {
  const response = await axios.post(`${API_URL}/api/login`, credentials);
  return response.data;
}

const registerUser = async (credentials: RegisterCredentials) => {
  // Remove confirmPassword before sending to API
  const { confirmPassword, ...apiCredentials } = credentials;
  const response = await axios.post(`${API_URL}/api/register`, apiCredentials);
  return response.data;
}

export const Route = createFileRoute('/auth')({
  component: AuthComponent,
  validateSearch: (search: Record<string, unknown>) => {
    return {
      mode: search.mode === 'register' ? 'register' : 'login'
    }
  },
  beforeLoad: async () => {
    // Check if the user is already authenticated
    const token = localStorage.getItem('auth_token')
    if (token) {
      // Redirect to home if already authenticated
      throw redirect({
        to: '/',
        replace: true,
      })
    }
    return {}
  }
})

function AuthComponent() {
  const navigate = useNavigate();
  const { mode } = useSearch({ from: '/auth' })
  const [isLogin, setIsLogin] = useState(mode === 'login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [errors, setErrors] = useState<{ 
    email?: string; 
    password?: string;
    confirmPassword?: string;
    auth?: string;
  }>({})

  // Login mutation
  const loginMutation = useMutation({
    mutationFn: loginUser,
    onSuccess: (data) => {
      // Store auth token or user data
      localStorage.setItem('auth_token', data.token);
      // Redirect to dashboard or home
      navigate({ to: '/' });
    },
    onError: (error: any) => {
      setErrors(prev => ({ 
        ...prev, 
        auth: error.response?.data?.message || 'Login failed. Please try again.' 
      }));
    }
  });

  // Register mutation
  const registerMutation = useMutation({
    mutationFn: registerUser,
    onSuccess: (data) => {
      // Store auth token or user data
      localStorage.setItem('auth_token', data.token);
      // Redirect to dashboard or home
      navigate({ to: '/' });
    },
    onError: (error: any) => {
      setErrors(prev => ({ 
        ...prev, 
        auth: error.response?.data?.message || 'Registration failed. Please try again.' 
      }));
    }
  });

  const validateForm = () => {
    const newErrors: { 
      email?: string; 
      password?: string;
      confirmPassword?: string;
    } = {}
    
    try {
      emailSchema.parse(email)
    } catch (error) {
      if (error instanceof z.ZodError) {
        newErrors.email = error.errors[0].message
      }
    }

    // Only validate password complexity for registration, not for login
    if (!isLogin) {
      try {
        passwordSchema.parse(password)
      } catch (error) {
        if (error instanceof z.ZodError) {
          newErrors.password = error.errors[0].message
        }
      }
      
      if (password !== confirmPassword) {
        newErrors.confirmPassword = 'Passwords do not match'
      }
    } else {
      // For login, just check if password is not empty
      if (!password) {
        newErrors.password = 'Password is required'
      }
    }

    setErrors(prev => ({ ...prev, ...newErrors }))
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    const credentials = { email, password };
    
    if (isLogin) {
      loginMutation.mutate(credentials);
    } else {
      registerMutation.mutate({ ...credentials, confirmPassword });
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 min-w-screen">
      <div className="flex w-full max-w-6xl mx-auto shadow-2xl rounded-xl overflow-hidden">
        {/* Left side - Image */}
        <div className="hidden lg:block w-1/2 bg-indigo-600">
          <img
            src="https://images.unsplash.com/photo-1496917756835-20cb06e75b4e?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1908&q=80"
            alt="Authentication background"
            className="w-full h-full object-cover"
          />
        </div>
        
        {/* Right side - Form */}
        <div className="w-full lg:w-1/2 bg-white px-8 py-12">
          <div className="max-w-md mx-auto">
            <div>
              <h2 className="text-3xl font-extrabold text-gray-900">
                {isLogin ? 'Sign in to your account' : 'Create new account'}
              </h2>
              <p className="mt-2 text-sm text-gray-600">
                Or{' '}
                <button
                  onClick={() => {
                    setIsLogin(!isLogin)
                    setErrors({})
                    setPassword('')
                    setConfirmPassword('')
                  }}
                  className="font-medium text-indigo-600 hover:text-indigo-500"
                >
                  {isLogin ? 'create a new account' : 'sign in to existing account'}
                </button>
              </p>
            </div>
            <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
              <div className="space-y-4">
                <div>
                  <label htmlFor="email-address" className="block text-sm font-medium text-gray-700">
                    Email address
                  </label>
                  <input
                    id="email-address"
                    name="email"
                    type="text"
                    autoComplete="email"
                    required
                    className={`mt-1 block w-full px-3 py-2 border rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm ${
                      errors.email ? 'border-red-500' : 'border-gray-300'
                    }`}
                    placeholder="Enter your email"
                    value={email}
                    onChange={(e) => {
                      setEmail(e.target.value)
                      setErrors(prev => ({ ...prev, email: undefined }))
                    }}
                  />
                  {errors.email && (
                    <p className="mt-1 text-sm text-red-500">{errors.email}</p>
                  )}
                </div>
                <div>
                  <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                    Password
                  </label>
                  <input
                    id="password"
                    name="password"
                    type="password"
                    autoComplete={isLogin ? "current-password" : "new-password"}
                    required
                    className={`mt-1 block w-full px-3 py-2 border rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm ${
                      errors.password ? 'border-red-500' : 'border-gray-300'
                    }`}
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => {
                      setPassword(e.target.value)
                      setErrors(prev => ({ ...prev, password: undefined, confirmPassword: undefined }))
                    }}
                  />
                  {errors.password ? (
                    <p className="mt-1 text-sm text-red-500">{errors.password}</p>
                  ) : (
                    // Only show password requirements for registration
                    !isLogin && (
                      <p className="mt-1 text-sm text-gray-500">
                        Password must be at least 8 characters and contain uppercase, lowercase, number, and special character
                      </p>
                    )
                  )}
                </div>
                {!isLogin && (
                  <div>
                    <label htmlFor="confirm-password" className="block text-sm font-medium text-gray-700">
                      Confirm Password
                    </label>
                    <input
                      id="confirm-password"
                      name="confirm-password"
                      type="password"
                      autoComplete="new-password"
                      required
                      className={`mt-1 block w-full px-3 py-2 border rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm ${
                        errors.confirmPassword ? 'border-red-500' : 'border-gray-300'
                      }`}
                      placeholder="Confirm your password"
                      value={confirmPassword}
                      onChange={(e) => {
                        setConfirmPassword(e.target.value)
                        setErrors(prev => ({ ...prev, confirmPassword: undefined }))
                      }}
                    />
                    {errors.confirmPassword && (
                      <p className="mt-1 text-sm text-red-500">{errors.confirmPassword}</p>
                    )}
                  </div>
                )}
              </div>

              {errors.auth && (
                <div className="rounded-md bg-red-50 p-4">
                  <div className="flex">
                    <div className="ml-3">
                      <h3 className="text-sm font-medium text-red-800">Authentication Error</h3>
                      <div className="mt-2 text-sm text-red-700">
                        <p>{errors.auth}</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div>
                <button
                  type="submit"
                  disabled={loginMutation.isPending || registerMutation.isPending}
                  className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                >
                  {(loginMutation.isPending || registerMutation.isPending) ? (
                    <span>Processing...</span>
                  ) : (
                    <span>{isLogin ? 'Sign in' : 'Register'}</span>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  )
}
