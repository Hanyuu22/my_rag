import { useState, useEffect } from 'react'

export interface Settings {
  // 检索参数
  retrieverTopK: number
  rerankerTopN: number
  scoreThreshold: number
  maxRetry: number
  // 生成参数
  temperature: number
  // 流程控制
  rerankerEnabled: boolean       // BGE Reranker 精排
  fallbackEnabled: boolean       // Fallback 兜底路由
  fallbackMethod: 'auto' | 'llm' | 'web'  // Fallback 方式
  // 头像
  userAvatar: string
  botAvatar: string
}

export const DEFAULT_SETTINGS: Settings = {
  retrieverTopK: 10,
  rerankerTopN: 3,
  scoreThreshold: 0.5,
  maxRetry: 2,
  temperature: 0.1,
  rerankerEnabled: true,
  fallbackEnabled: true,
  fallbackMethod: 'auto',
  userAvatar: '/avatar_user.png',
  botAvatar: '/avatar_bot.png',
}

const STORAGE_KEY = 'rag_settings'

export function useSettings() {
  const [settings, setSettings] = useState<Settings>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) {
        return { ...DEFAULT_SETTINGS, ...JSON.parse(stored) }
      }
    } catch {}
    return DEFAULT_SETTINGS
  })

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
  }, [settings])

  const update = (patch: Partial<Settings>) => {
    setSettings(prev => ({ ...prev, ...patch }))
  }

  const reset = () => {
    setSettings(DEFAULT_SETTINGS)
    localStorage.removeItem(STORAGE_KEY)
  }

  return { settings, update, reset }
}
