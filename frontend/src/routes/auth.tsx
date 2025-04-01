import { createFileRoute, useSearch, useNavigate, redirect } from '@tanstack/react-router'
import { useState, useEffect, useRef } from 'react'
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

const verificationCodeSchema = z.string()
  .length(6, 'Verification code must be 6 digits')
  .regex(/^\d+$/, 'Verification code must contain only numbers')

// Authentication API functions
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

interface AuthCredentials {
  email: string;
  password: string;
}

interface RegisterCredentials extends AuthCredentials {
  username?: string;
}

interface VerifyCredentials {
  email: string;
  token: string;
  username: string;
}

interface ResendCodeCredentials {
  email: string;
  password: string;
}

const loginUser = async (credentials: AuthCredentials) => {
  const response = await axios.post(`${API_URL}/api/login`, credentials);
  return response.data;
}

const registerUser = async (credentials: RegisterCredentials) => {
  const response = await axios.post(`${API_URL}/api/register`, credentials);
  return response.data;
}

const resendVerificationCode = async (credentials: ResendCodeCredentials) => {
  const response = await axios.post(`${API_URL}/api/register`, credentials);
  return response.data;
}

const verifyUser = async (credentials: VerifyCredentials) => {
  const response = await axios.post(`${API_URL}/api/verify`, credentials);
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
  const [registerStep, setRegisterStep] = useState(1)
  const [email, setEmail] = useState('')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [verificationCode, setVerificationCode] = useState('')
  const [timer, setTimer] = useState(0)
  const [isResendDisabled, setIsResendDisabled] = useState(false)
  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const [errors, setErrors] = useState<{ 
    email?: string; 
    username?: string;
    password?: string;
    confirmPassword?: string;
    verificationCode?: string;
    auth?: string;
  }>({})

  useEffect(() => {
    return () => {
      // Clean up timer on unmount
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  const startTimer = () => {
    setTimer(60);
    setIsResendDisabled(true);
    
    if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    
    timerRef.current = setInterval(() => {
      setTimer((prevTimer) => {
        if (prevTimer <= 1) {
          clearInterval(timerRef.current as NodeJS.Timeout);
          setIsResendDisabled(false);
          return 0;
        }
        return prevTimer - 1;
      });
    }, 1000);
  };

  // Register step 1 mutation
  const registerStep1Mutation = useMutation({
    mutationFn: registerUser,
    onSuccess: () => {
      // Move to step 2 and start timer
      setRegisterStep(2);
      startTimer();
      setErrors(prev => ({ ...prev, auth: undefined }));
    },
    onError: (error: any) => {
      setErrors(prev => ({ 
        ...prev, 
        auth: error.response?.data?.message || 'Registration failed. Please check your details and try again.' 
      }));
    }
  });

  // Resend verification code mutation
  const resendCodeMutation = useMutation({
    mutationFn: resendVerificationCode,
    onSuccess: () => {
      startTimer();
      setErrors(prev => ({ ...prev, auth: undefined }));
    },
    onError: (error: any) => {
      setErrors(prev => ({ 
        ...prev, 
        auth: error.response?.data?.message || 'Failed to resend verification code. Please try again.' 
      }));
    }
  });

  // Verify user mutation (final step of registration)
  const verifyUserMutation = useMutation({
    mutationFn: (credentials: VerifyCredentials) => verifyUser(credentials),
    onSuccess: (data) => {
      // Store auth token or user data
      localStorage.setItem('auth_token', data.token);
      // Redirect to dashboard or home
      navigate({ to: '/' });
    },
    onError: (error: any) => {
      setErrors(prev => ({ 
        ...prev, 
        auth: error.response?.data?.message || 'Verification failed. Please check your code and try again.' 
      }));
    }
  });

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

  const validateStep1 = () => {
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

    setErrors(prev => ({ ...prev, ...newErrors }))
    return Object.keys(newErrors).length === 0
  }

  const validateStep2 = () => {
    const newErrors: { 
      verificationCode?: string;
      username?: string;
    } = {}
    
    if (!verificationCode.trim() || verificationCode.length !== 6) {
      newErrors.verificationCode = 'Valid verification code is required (6 digits)'
    }

    if (!username.trim()) {
      newErrors.username = 'Username is required'
    }

    setErrors(prev => ({ ...prev, ...newErrors }))
    return Object.keys(newErrors).length === 0
  }

  const validateForm = () => {
    if (isLogin) {
      // For login, just check if email and password are not empty
      const newErrors: { email?: string; password?: string } = {}
      
      try {
        emailSchema.parse(email)
      } catch (error) {
        if (error instanceof z.ZodError) {
          newErrors.email = error.errors[0].message
        }
      }

      if (!password) {
        newErrors.password = 'Password is required'
      }

      setErrors(prev => ({ ...prev, ...newErrors }))
      return Object.keys(newErrors).length === 0
    } else {
      // For registration, use the appropriate step validation
      return registerStep === 1 ? validateStep1() : validateStep2()
    }
  }

  const handleNextStep = () => {
    if (validateStep1()) {
      // Send registration with email and password
      registerStep1Mutation.mutate({ email, password });
      // Note: The step change now happens in the mutation's onSuccess callback
    }
  };

  const handleResendCode = () => {
    resendCodeMutation.mutate({ email, password });
  };

  const handlePrevStep = () => {
    setRegisterStep(1);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    if (isLogin) {
      const credentials = { email, password };
      loginMutation.mutate(credentials);
    } else if (registerStep === 2) {
      // Send verification request with email, token and username
      verifyUserMutation.mutate({
        email,
        token: verificationCode,
        username
      });
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
                    setRegisterStep(1)
                  }}
                  className="font-medium text-indigo-600 hover:text-indigo-500"
                >
                  {isLogin ? 'create a new account' : 'sign in to existing account'}
                </button>
              </p>
            </div>
            <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
              <div className="space-y-4">
                {isLogin ? (
                  <>
                    {/* Login Form */}
                    <div>
                      <label htmlFor="email-address" className="block text-sm font-medium text-gray-700">
                        Email address
                      </label>
                      <div className="mt-1">
                        <input
                          id="email-address"
                          name="email"
                          type="text"
                          autoComplete="email"
                          required
                          className={`block w-full px-3 py-2 border rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm ${
                            errors.email ? 'border-red-500' : 'border-gray-300'
                          }`}
                          placeholder="Enter your email"
                          value={email}
                          onChange={(e) => {
                            setEmail(e.target.value)
                            setErrors(prev => ({ ...prev, email: undefined }))
                          }}
                        />
                      </div>
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
                        autoComplete="current-password"
                        required
                        className={`mt-1 block w-full px-3 py-2 border rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm ${
                          errors.password ? 'border-red-500' : 'border-gray-300'
                        }`}
                        placeholder="Enter your password"
                        value={password}
                        onChange={(e) => {
                          setPassword(e.target.value)
                          setErrors(prev => ({ ...prev, password: undefined }))
                        }}
                      />
                      {errors.password && (
                        <p className="mt-1 text-sm text-red-500">{errors.password}</p>
                      )}
                    </div>
                  </>
                ) : (
                  <>
                    {/* Registration Form - Step 1 */}
                    {registerStep === 1 && (
                      <>
                        {/* Row 1: Email */}
                        <div>
                          <label htmlFor="email-address" className="block text-sm font-medium text-gray-700">
                            Email address
                          </label>
                          <div className="mt-1">
                            <input
                              id="email-address"
                              name="email"
                              type="text"
                              autoComplete="email"
                              required
                              className={`block w-full px-3 py-2 border rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm ${
                                errors.email ? 'border-red-500' : 'border-gray-300'
                              }`}
                              placeholder="Enter your email"
                              value={email}
                              onChange={(e) => {
                                setEmail(e.target.value)
                                setErrors(prev => ({ ...prev, email: undefined }))
                              }}
                            />
                          </div>
                          {errors.email && (
                            <p className="mt-1 text-sm text-red-500">{errors.email}</p>
                          )}
                        </div>

                        {/* Row 2: Password */}
                        <div>
                          <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                            Password
                          </label>
                          <input
                            id="password"
                            name="password"
                            type="password"
                            autoComplete="new-password"
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
                            <p className="mt-1 text-sm text-gray-500">
                              Password must be at least 8 characters and contain uppercase, lowercase, number, and special character
                            </p>
                          )}
                        </div>

                        {/* Row 3: Confirm Password */}
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
                      </>
                    )}

                    {/* Registration Form - Step 2 */}
                    {registerStep === 2 && (
                      <>
                        {/* Row 1: Verification Code */}
                        <div>
                          <label htmlFor="verification-code" className="block text-sm font-medium text-gray-700">
                            Verification Code
                          </label>
                          <div className="flex mt-1 items-center">
                            <input
                              id="verification-code"
                              name="verification-code"
                              type="text"
                              maxLength={6}
                              inputMode="numeric"
                              pattern="[0-9]{6}"
                              className={`block flex-1 px-3 py-2 border rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm ${
                                errors.verificationCode ? 'border-red-500' : 'border-gray-300'
                              }`}
                              placeholder="6-digit code"
                              value={verificationCode}
                              onChange={(e) => {
                                // Only allow numeric input
                                const value = e.target.value.replace(/[^0-9]/g, '');
                                setVerificationCode(value);
                                setErrors(prev => ({ ...prev, verificationCode: undefined }));
                              }}
                            />
                            <button 
                              type="button"
                              onClick={handleResendCode}
                              disabled={isResendDisabled || resendCodeMutation.isPending}
                              className="ml-2 inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 whitespace-nowrap"
                            >
                              {isResendDisabled ? `Resend (${timer}s)` : 'Resend'}
                            </button>
                          </div>
                          {errors.verificationCode && (
                            <p className="mt-1 text-sm text-red-500">{errors.verificationCode}</p>
                          )}
                          <p className="mt-1 text-xs text-gray-500">
                            Enter the 6-digit verification code sent to your email
                          </p>
                        </div>

                        {/* Row 2: Username */}
                        <div>
                          <label htmlFor="username" className="block text-sm font-medium text-gray-700">
                            Username
                          </label>
                          <input
                            id="username"
                            name="username"
                            type="text"
                            autoComplete="username"
                            required
                            className={`mt-1 block w-full px-3 py-2 border rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm ${
                              errors.username ? 'border-red-500' : 'border-gray-300'
                            }`}
                            placeholder="Choose a username"
                            value={username}
                            onChange={(e) => {
                              setUsername(e.target.value)
                              setErrors(prev => ({ ...prev, username: undefined }))
                            }}
                          />
                          {errors.username && (
                            <p className="mt-1 text-sm text-red-500">{errors.username}</p>
                          )}
                        </div>
                      </>
                    )}
                  </>
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
                {isLogin ? (
                  // Login Button
                  <button
                    type="submit"
                    disabled={loginMutation.isPending}
                    className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                  >
                    {loginMutation.isPending ? <span>Processing...</span> : <span>Sign in</span>}
                  </button>
                ) : (
                  // Registration Buttons - Step 1: Next Button, Step 2: Register Button
                  <div className="flex space-x-3">
                    {registerStep === 2 && (
                      <button
                        type="button"
                        onClick={handlePrevStep}
                        className="flex-1 flex justify-center py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                      >
                        Back
                      </button>
                    )}
                    {registerStep === 1 ? (
                      <button
                        type="button"
                        onClick={handleNextStep}
                        className="flex-1 flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                      >
                        Next
                      </button>
                    ) : (
                      <button
                        type="submit"
                        disabled={verifyUserMutation.isPending}
                        className="flex-1 flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                      >
                        {verifyUserMutation.isPending ? <span>Processing...</span> : <span>Register</span>}
                      </button>
                    )}
                  </div>
                )}
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  )
}
