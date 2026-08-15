const rawApiUrl = import.meta.env.VITE_API_URL || ''
export const API_BASE_URL = rawApiUrl.replace(/\/+$/, '')

export const getApiEndpoint = (path: string): string => {
  const cleanPath = path.startsWith('/') ? path : `/${path}`
  return API_BASE_URL ? `${API_BASE_URL}${cleanPath}` : cleanPath
}
